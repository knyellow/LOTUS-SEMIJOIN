"""
Find articles where the pairwise semantic join misses but a match-only (existential) judgment succeeds.

  (A) pairwise sem_join:  one LLM call per (article, reaction) pair with the BioDEX join_instruction;
                          the article is selected iff some pair is True  (= join + projection)
  (B) match-only:         one LLM call per article asking whether ANY of the reactions is described

Target cases: gold-positive articles (reactions_list ∩ R ≠ ∅) where every (A) pair is False but (B) is True.
The reverse direction ((B) false positives on gold-negative articles) is reported too.

Parse failures are tracked separately: an output is a parse failure when lotus' postprocessor
gives different answers with default=True and default=False.

usage: python failure_cases.py [n_pos] [n_neg]
"""

import json
import os
import sys
import time

import pandas as pd
from datasets import load_dataset
from semijoin_baseline import JOIN_INSTRUCTION, LIVER_REACTIONS, TRUNCATION_LIMIT, configure

from lotus.sem_ops.postprocessors import filter_postprocess
from lotus.sem_ops.sem_filter import sem_filter
from lotus.sem_ops.sem_join import sem_join
from lotus.templates.task_instructions import df2multimodal_info

OUT_DIR = "failure_results"

# existential version of JOIN_INSTRUCTION: same wording, but over the whole reaction set
MATCH_INSTRUCTION = (
    """Can at least one of the following conditions be found in the following medical article?
        Medical article: {patient_description}

        Conditions we are looking for: """
    + ", ".join(LIVER_REACTIONS)
    + """


        Determine if at least one of the conditions is described in the medical article, considering the context and meaning beyond just the presence of individual words."""
)


def load_articles(n_pos: int, n_neg: int) -> pd.DataFrame:
    df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    df["reactions_list"] = df["reactions"].apply(lambda x: [r.strip() for r in x.split(",")])
    df["gold_matches"] = df["reactions_list"].apply(lambda rl: sorted(set(rl) & set(LIVER_REACTIONS)))
    df["gold"] = df["gold_matches"].apply(bool)
    df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[:TRUNCATION_LIMIT])
    # first n_pos gold-positive and first n_neg gold-negative articles in test-set order
    picked = pd.concat([df[df["gold"]][:n_pos], df[~df["gold"]][:n_neg]])
    picked["test_idx"] = picked.index
    return picked.reset_index(drop=True)


def parse(raw_outputs: list[str], lm) -> tuple[list[bool], list[bool]]:
    """Strict answers (parse failure -> False) and a parse-failure mask."""
    as_true = filter_postprocess(raw_outputs, lm, default=True).outputs
    as_false = filter_postprocess(raw_outputs, lm, default=False).outputs
    return as_false, [t != f for t, f in zip(as_true, as_false)]


def main():
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    os.makedirs(OUT_DIR, exist_ok=True)

    lm = configure()
    articles = load_articles(n_pos, n_neg)
    reactions = pd.DataFrame({"reaction": LIVER_REACTIONS})
    print(f"{int(articles['gold'].sum())} gold-positive + {int((~articles['gold']).sum())} gold-negative articles")

    # (B) match-only: one call per article (cheap, run first so it is saved early)
    start = time.time()
    docs = df2multimodal_info(articles, ["patient_description"])
    b_raw = sem_filter(docs, lm, MATCH_INSTRUCTION, default=False, progress_bar_desc="match-only").raw_outputs
    articles["B"], articles["B_parse_fail"] = parse(b_raw, lm)
    articles["B_raw"] = b_raw
    b_time = time.time() - start
    articles.drop(columns=["patient_description", "fulltext_processed"]).to_json(
        os.path.join(OUT_DIR, "match_only.jsonl"), orient="records", lines=True
    )
    print(f"(B) done in {b_time:.0f}s")

    # (A) pairwise sem_join over the same articles; outputs are ordered article-major
    start = time.time()
    out = sem_join(
        articles["patient_description"],
        reactions["reaction"],
        list(articles.index),
        list(reactions.index),
        "patient_description",
        "reaction",
        lm,
        JOIN_INSTRUCTION,
        default=False,
    )
    a_time = time.time() - start
    pair_true, pair_fail = parse(out.all_raw_outputs, lm)
    k = len(reactions)
    articles["A_true_reactions"] = [
        [r for r, t in zip(LIVER_REACTIONS, pair_true[i * k : (i + 1) * k]) if t] for i in range(len(articles))
    ]
    articles["A"] = articles["A_true_reactions"].apply(bool)
    articles["A_parse_fails"] = [sum(pair_fail[i * k : (i + 1) * k]) for i in range(len(articles))]
    articles["A_raw"] = [out.all_raw_outputs[i * k : (i + 1) * k] for i in range(len(articles))]

    def counts(df):
        return {
            "n": len(df),
            "A_true": int(df["A"].sum()),
            "B_true": int(df["B"].sum()),
            "A_miss_B_hit": int((~df["A"] & df["B"]).sum()),
            "A_hit_B_miss": int((df["A"] & ~df["B"]).sum()),
        }

    pos, neg = articles[articles["gold"]], articles[~articles["gold"]]
    cases = pos[~pos["A"] & pos["B"]]
    summary = {
        "reactions": LIVER_REACTIONS,
        "gold_positive": counts(pos),
        "gold_negative": counts(neg),
        "parse_failures": {"A_pairs": int(sum(pair_fail)), "A_pairs_total": len(pair_fail), "B": int(articles["B_parse_fail"].sum())},
        "lm_calls": {"A": len(pair_fail), "B": len(b_raw)},
        "time_s": {"A": round(a_time, 1), "B": round(b_time, 1)},
        "target_cases": cases["test_idx"].tolist(),
    }
    print(json.dumps(summary, indent=2))

    keep = ["test_idx", "title", "gold", "gold_matches", "A", "A_true_reactions", "A_parse_fails", "B", "B_parse_fail", "A_raw", "B_raw"]
    articles[keep].to_json(os.path.join(OUT_DIR, "all.jsonl"), orient="records", lines=True)
    cases[keep + ["patient_description"]].to_json(os.path.join(OUT_DIR, "cases.jsonl"), orient="records", lines=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
