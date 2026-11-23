"""
Semantic semi-join baselines on BioDEX:  articles ⋉ R,  R = a fixed set of 10 reactions (liver or haematologic).

semi-join is computed by definition as a projection of the join onto the left columns:
    L ⋉ R = π_cols(L)(L ⋈ R)   (with duplicate elimination on the left row id)

Baselines
  - join_full:    sem_join without cascade (every pair goes to the LLM), then projection
  - join_cascade: sem_join with the LOTUS cascade (same settings as the BioDEX Join pipeline), then projection
  - early_stop:   per article, evaluate R in list order with the same pairwise predicate and stop at the first True
                  (exact semi-join semantics: same answer as join_full given a deterministic LM, fewer calls)

Ground truth is label-based: an article is in the answer iff reactions_list ∩ R ≠ ∅.
join_full is also used as the oracle for join_cascade.

usage: [REACTION_SET=liver|haematologic] python semijoin_baseline.py [n_articles] [full|cascade|early|both]
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from datasets import load_dataset

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.sem_ops.sem_join import sem_join as pairwise_join
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS

RESULTS_DIR = "semijoin_results"
TRUNCATION_LIMIT = 8000

LIVER_REACTIONS = [
    "Drug-induced liver injury",
    "Hepatic function abnormal",
    "Hepatotoxicity",
    "Liver injury",
    "Hepatic failure",
    "Hepatic enzyme increased",
    "Acute hepatic failure",
    "Hepatitis",
    "Immune-mediated hepatitis",
    "Hepatic necrosis",
]

HAEMATOLOGIC_REACTIONS = [
    "Neutropenia",
    "Febrile neutropenia",
    "Anaemia",
    "Pancytopenia",
    "Leukopenia",
    "Thrombocytopenia",
    "Haemolytic anaemia",
    "Heparin-induced thrombocytopenia",
    "Aplastic anaemia",
    "Normocytic anaemia",
]

# right table R; liver: 10/250 gold articles, haematologic: 32/250
REACTION_SETS = {"liver": LIVER_REACTIONS, "haematologic": HAEMATOLOGIC_REACTIONS}
REACTION_SET = os.environ.get("REACTION_SET", "liver")

# PROMPT=paper: same pairwise predicate as the BioDEX Join pipeline (biodex_tester.py)
# PROMPT=adr:   asks whether the condition is a reported adverse drug reaction, which is what BioDEX labels encode
#               (the paper predicate only asks whether it is described, so background mentions, disease-caused
#               findings and normal lab values count as matches)
# baselines and the optimized semi-join use the same predicate, so they differ only in the algorithm
PAPER_JOIN_INSTRUCTION = """Can the following condition be found in the following medical article?
        Medical article: {patient_description}

        Condition we are looking for: {reaction}


        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""

ADR_JOIN_INSTRUCTION = """Is the following condition reported in the following case report as an adverse reaction experienced by the patient, plausibly caused by a drug or treatment?
        Medical article: {patient_description}

        Condition we are looking for: {reaction}


        Answer True only if {reaction} (or a clinically equivalent finding) actually occurred in the patient as an adverse reaction to a drug or treatment. Answer False if it is only mentioned as background, literature, a differential diagnosis or a risk, if it is caused by the underlying disease rather than a drug, if it was pre-existing, or if the reported lab values are within the normal range."""

PROMPT = os.environ.get("PROMPT", "paper")
JOIN_INSTRUCTION = {"paper": PAPER_JOIN_INSTRUCTION, "adr": ADR_JOIN_INSTRUCTION}[PROMPT]

MAP_INSTRUCTION = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."


class CountingLM(LM):
    """LM that counts requests (lotus' LMStats only tracks tokens)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = 0

    def __call__(self, messages, *args, **kwargs):
        self.n_calls += len(messages)
        return super().__call__(messages, *args, **kwargs)


LLM_MODEL = os.environ.get("LLM_MODEL", "hosted_vllm/llama3.1-8b")


def configure() -> CountingLM:
    if LLM_MODEL.startswith("hosted_vllm/"):
        # local llama-server
        endpoint = {
            "api_base": os.environ.get("LLM_API_BASE", "http://localhost:8080/v1"),
            "max_batch_size": int(os.environ.get("LLM_BATCH", "4")),  # match llama-server --parallel
        }
    else:
        # hosted API (e.g. gpt-4o-mini-2024-07-18), reads OPENAI_API_KEY
        endpoint = {"max_batch_size": 64}
    lm = CountingLM(model=LLM_MODEL, temperature=0.0, max_tokens=256, **endpoint)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=64, device="cpu")
    lotus.settings.configure(lm=lm, rm=rm, vs=FaissVS())
    return lm


def load_articles(n: int) -> pd.DataFrame:
    df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()[:n].reset_index(drop=True)
    df["reactions_list"] = df["reactions"].apply(lambda x: [r.strip() for r in x.split(",")])
    df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[:TRUNCATION_LIMIT])
    df["_qid"] = df.index
    return df[["_qid", "title", "patient_description", "reactions_list"]]


def semi_join(articles: pd.DataFrame, reactions: pd.DataFrame, cascade_args: CascadeArgs | None):
    left_cols = list(articles.columns)
    joined = articles.sem_join(reactions, JOIN_INSTRUCTION, cascade_args=cascade_args)
    # projection onto the left columns + duplicate elimination on the left row id
    return joined[left_cols].drop_duplicates(subset="_qid"), joined


CHUNK = 10  # articles per join_full checkpoint
SHARD = os.environ.get("SHARD", "0/1")  # "i/k": this process runs chunks with index % k == i


def run_full_chunks(articles: pd.DataFrame, reactions: pd.DataFrame, lm: CountingLM, chunk_dir: str) -> None:
    """join_full in article chunks; each finished chunk is saved, so a crash or node expiry loses at most one chunk."""
    os.makedirs(chunk_dir, exist_ok=True)
    shard_id, n_shards = map(int, SHARD.split("/"))
    for ci, start in enumerate(range(0, len(articles), CHUNK)):
        meta_path = os.path.join(chunk_dir, f"{start:04d}.json")
        if ci % n_shards != shard_id or os.path.exists(meta_path):
            continue
        calls_before = lm.n_calls
        t = time.time()
        _, joined = semi_join(articles.iloc[start : start + CHUNK], reactions, None)
        elapsed = time.time() - t
        joined[["_qid", "title", "reaction"]].to_csv(os.path.join(chunk_dir, f"{start:04d}.csv"), index=False)
        with open(meta_path, "w") as f:  # written last: marks the chunk as done
            json.dump({"start": start, "lm_calls": lm.n_calls - calls_before, "time_s": round(elapsed, 1)}, f)
        print(f"join_full chunk {start}-{start + CHUNK - 1}: {lm.n_calls - calls_before} calls, {elapsed:.0f}s", flush=True)


def merge_full_chunks(n: int, chunk_dir: str) -> tuple[pd.DataFrame, int, float] | None:
    """(joined pairs, lm calls, summed time) once every chunk exists, else None."""
    starts = range(0, n, CHUNK)
    if not all(os.path.exists(os.path.join(chunk_dir, f"{s:04d}.json")) for s in starts):
        return None
    metas = [json.load(open(os.path.join(chunk_dir, f"{s:04d}.json"))) for s in starts]
    joined = pd.concat([pd.read_csv(os.path.join(chunk_dir, f"{s:04d}.csv")) for s in starts], ignore_index=True)
    return joined, sum(m["lm_calls"] for m in metas), sum(m["time_s"] for m in metas)


EARLY_WORKERS = int(os.environ.get("EARLY_WORKERS", "4"))  # concurrent articles (match llama-server --parallel)


def run_early_stop(articles: pd.DataFrame, reaction_list: list[str], lm: CountingLM, path: str):
    """Early-stopping semi-join. One JSONL line per finished article, so reruns resume where they stopped.

    Articles run concurrently; within an article reactions are sequential, which also lets llama-server reuse the
    cached article prefix (the prompt is system + article + reaction)."""
    done = {}
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            done[rec["_qid"]] = rec
    lock = threading.Lock()
    reactions = pd.Series(reaction_list, name="reaction")

    def one(row) -> None:
        t = time.time()
        matched, calls = None, 0
        for k in range(len(reaction_list)):
            out = pairwise_join(
                pd.Series([row["patient_description"]], name="patient_description"),
                reactions.iloc[[k]],
                [row["_qid"]],
                [k],
                "patient_description",
                "reaction",
                lm,
                JOIN_INSTRUCTION,
                show_progress_bar=False,
            )
            calls += 1
            if out.join_results:
                matched = reaction_list[k]
                break
        rec = {"_qid": int(row["_qid"]), "matched": matched, "lm_calls": calls, "time_s": round(time.time() - t, 1)}
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[rec["_qid"]] = rec
            print(f"early_stop {len(done)}/{len(articles)}: qid={rec['_qid']} calls={calls} matched={matched}", flush=True)

    todo = [r for r in articles.to_dict("records") if r["_qid"] not in done]
    with ThreadPoolExecutor(EARLY_WORKERS) as ex:
        list(ex.map(one, todo))
    recs = [done[q] for q in articles["_qid"]]
    joined = pd.DataFrame(
        [{"_qid": r["_qid"], "reaction": r["matched"]} for r in recs if r["matched"] is not None],
        columns=["_qid", "reaction"],
    )
    return joined, sum(r["lm_calls"] for r in recs)


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "n_pred": len(pred)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    which = sys.argv[2] if len(sys.argv) > 2 else "both"

    lm = configure()
    articles = load_articles(n)
    reaction_list = REACTION_SETS[REACTION_SET]
    reactions = pd.DataFrame({"reaction": reaction_list})
    gold = set(articles.loc[articles["reactions_list"].apply(lambda rl: bool(set(rl) & set(reaction_list))), "_qid"])
    print(f"{n} articles x {len(reactions)} reactions = {n * len(reactions)} pairs, {len(gold)} gold articles")

    out_dir = os.path.join(RESULTS_DIR, LLM_MODEL.split("/")[-1], f"n={n}")
    if REACTION_SET != "liver":  # liver results predate reaction sets and live directly under the model dir
        out_dir = os.path.join(RESULTS_DIR, LLM_MODEL.split("/")[-1], REACTION_SET, f"n={n}")
    if PROMPT != "paper":  # e.g. semijoin_results/llama3.1-70b/adr/n=500
        out_dir = os.path.join(os.path.dirname(out_dir), PROMPT, os.path.basename(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    runs = {}
    oracle = None

    configs = {
        "join_full": None,
        "join_cascade": CascadeArgs(
            recall_target=0.9,
            precision_target=0.9,
            failure_probability=0.2,
            sampling_percentage=min(1.0, 100 / (n * len(reactions))),  # ~100 oracle samples
            map_instruction=MAP_INSTRUCTION,
            cascade_IS_random_seed=42,
            min_join_cascade_size=0,
        ),
        "early_stop": None,
    }
    if which == "both":
        configs.pop("early_stop")
    elif which == "early":
        configs = {"early_stop": None}
    else:
        configs = {f"join_{which}": configs[f"join_{which}"]}
    chunk_dir = os.path.join(out_dir, "join_full_chunks")
    if "join_full" not in configs and (merged := merge_full_chunks(n, chunk_dir)) is not None:
        oracle = set(merged[0]["_qid"])  # join_full finished earlier (possibly on other nodes)

    for name, cascade_args in configs.items():
        print(f"\n=== {name} ===")
        if name == "join_full":
            run_full_chunks(articles, reactions, lm, chunk_dir)
            merged = merge_full_chunks(n, chunk_dir)
            if merged is None:
                print(f"join_full: shard {SHARD} done, other chunks still missing in {chunk_dir}")
                continue
            joined, calls, elapsed = merged
        elif name == "early_stop":
            start = time.time()
            joined, calls = run_early_stop(articles, reaction_list, lm, os.path.join(out_dir, "early_stop_progress.jsonl"))
            elapsed = time.time() - start  # wall clock of this invocation only (resumed runs undercount)
            joined["title"] = joined["_qid"].map(articles.set_index("_qid")["title"])
        else:
            calls_before = lm.n_calls
            start = time.time()
            _, joined = semi_join(articles, reactions, cascade_args)
            calls, elapsed = lm.n_calls - calls_before, time.time() - start
        # projection onto the left row id (same as semi_join(), kept as ids so chunked results merge too)
        semi = articles[articles["_qid"].isin(set(joined["_qid"]))]

        pred = set(semi["_qid"])
        res = {
            "method": name,
            "model": LLM_MODEL,
            "n_articles": n,
            "n_reactions": len(reactions),
            "n_gold": len(gold),
            "vs_labels": prf(pred, gold),
            "lm_calls": calls,
            "time_s": round(elapsed, 1),
            "join_rows": len(joined),
        }
        if name == "join_full":
            oracle = pred
        elif oracle is not None:
            res["vs_join_full"] = prf(pred, oracle)
        runs[name] = res
        print(json.dumps(res, indent=2))

        joined.drop(columns=["patient_description"], errors="ignore").to_csv(os.path.join(out_dir, f"{name}_join.csv"), index=False)
        semi[["_qid", "title"]].to_csv(os.path.join(out_dir, f"{name}_semi.csv"), index=False)

    summary_name = "summary.json" if which == "both" else f"summary_{which}.json"
    with open(os.path.join(out_dir, summary_name), "w") as f:
        json.dump({"reaction_set": REACTION_SET, "prompt": PROMPT, "reactions": reaction_list, "runs": runs}, f, indent=2)
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
