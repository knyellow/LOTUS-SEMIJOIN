"""
Semantic semi-join baselines on BioDEX:  articles ⋉ R,  R = a fixed set of liver-related reactions.

semi-join is computed by definition as a projection of the join onto the left columns:
    L ⋉ R = π_cols(L)(L ⋈ R)   (with duplicate elimination on the left row id)

Baselines
  - join_full:    sem_join without cascade (every pair goes to the LLM), then projection
  - join_cascade: sem_join with the LOTUS cascade (same settings as the BioDEX Join pipeline), then projection

Ground truth is label-based: an article is in the answer iff reactions_list ∩ R ≠ ∅.
join_full is also used as the oracle for join_cascade.

usage: python semijoin_baseline.py [n_articles] [full|cascade|both]
"""

import json
import os
import sys
import time

import pandas as pd
from datasets import load_dataset

import lotus
from lotus.models import LM, SentenceTransformersRM
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

# same pairwise predicate as the BioDEX Join pipeline (biodex_tester.py), so baselines and
# the optimized semi-join differ only in the algorithm, not in the prompt
JOIN_INSTRUCTION = """Can the following condition be found in the following medical article?
        Medical article: {patient_description}

        Condition we are looking for: {reaction}


        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""

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
        # local llama-server, single slot
        endpoint = {"api_base": os.environ.get("LLM_API_BASE", "http://localhost:8080/v1"), "max_batch_size": 4}
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
    reactions = pd.DataFrame({"reaction": LIVER_REACTIONS})
    gold = set(articles.loc[articles["reactions_list"].apply(lambda rl: bool(set(rl) & set(LIVER_REACTIONS))), "_qid"])
    print(f"{n} articles x {len(reactions)} reactions = {n * len(reactions)} pairs, {len(gold)} gold articles")

    out_dir = os.path.join(RESULTS_DIR, LLM_MODEL.split("/")[-1], f"n={n}")
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
    }
    if which != "both":
        configs = {f"join_{which}": configs[f"join_{which}"]}

    for name, cascade_args in configs.items():
        print(f"\n=== {name} ===")
        calls_before = lm.n_calls
        start = time.time()
        semi, joined = semi_join(articles, reactions, cascade_args)
        elapsed = time.time() - start

        pred = set(semi["_qid"])
        res = {
            "method": name,
            "model": LLM_MODEL,
            "n_articles": n,
            "n_reactions": len(reactions),
            "n_gold": len(gold),
            "vs_labels": prf(pred, gold),
            "lm_calls": lm.n_calls - calls_before,
            "time_s": round(elapsed, 1),
            "join_rows": len(joined),
        }
        if name == "join_full":
            oracle = pred
        elif oracle is not None:
            res["vs_join_full"] = prf(pred, oracle)
        runs[name] = res
        print(json.dumps(res, indent=2))

        joined.drop(columns=["patient_description"]).to_csv(os.path.join(out_dir, f"{name}_join.csv"), index=False)
        semi[["_qid", "title"]].to_csv(os.path.join(out_dir, f"{name}_semi.csv"), index=False)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"reactions": LIVER_REACTIONS, "runs": runs}, f, indent=2)
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
