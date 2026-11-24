"""Reproduce the exact BioDEX abstract-in-LLM-input check without LOTUS or model imports."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

DATASET = "BioDEX/BioDEX-Reactions"
REVISION = "01a5dacdabd144a120af04931a11a99febd48432"
FILENAME = "data/test-00000-of-00001-2449439e9130546c.parquet"
PARQUET_SHA256 = "7ea93550ba12993ad87d47d0afd665574d42ec14218bca1bfdba13fe3d4ff749"
EXPECTED_ROWS = 4249
REFERENCE = Path(__file__).parent / "evidence" / "abstract_containment"


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def baseline_settings() -> tuple[int, dict]:
    # Read the existing literal settings without importing LOTUS, torch, or datasets.
    source = Path(__file__).with_name("semijoin_baseline.py").read_text(encoding="utf-8")
    names = {"TRUNCATION_LIMIT", "LIVER_REACTIONS", "HAEMATOLOGIC_REACTIONS"}
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    limit = values["TRUNCATION_LIMIT"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("The baseline truncation limit must be a positive integer.")
    return limit, {"liver": values["LIVER_REACTIONS"], "haematologic": values["HAEMATOLOGIC_REACTIONS"]}


def check_article(row: dict, qid: int, limit: int, panels: dict) -> dict:
    abstract, full = row["abstract"] or "", row["fulltext_processed"] or ""
    has_abstract, has_fulltext = bool(abstract.strip()), bool(full.strip())
    start = full.find(abstract) if has_abstract else -1
    contained = has_abstract and abstract in full[:limit]
    if not has_abstract:
        status = "empty_abstract"
    elif not has_fulltext:
        status = "empty_fulltext"
    elif start == -1:
        status = "not_found_in_fulltext"
    elif not contained:
        status = "outside_prefix"
    else:
        status = "contained"
    labels = {label.strip() for label in (row["reactions"] or "").split(",")}
    return {
        "_qid": qid,
        "pmid": row["pmid"],
        "abstract_characters": len(abstract),
        "fulltext_characters": len(full),
        "has_abstract": has_abstract,
        "has_fulltext": has_fulltext,
        "exact_prefix": contained,
        "exact_full": has_abstract and start >= 0,
        "abstract_start": start if start >= 0 else None,
        "abstract_end_exclusive": start + len(abstract) if start >= 0 else None,
        "status": status,
        **{f"{name}_positive": bool(labels.intersection(reactions)) for name, reactions in panels.items()},
    }


def summarize(rows: list[dict], limit: int) -> dict:
    starts = [row["abstract_start"] for row in rows if row["exact_full"]]
    ends = [row["abstract_end_exclusive"] for row in rows if row["exact_full"]]
    return {
        "n_articles": len(rows),
        "n_empty_abstract": sum(not row["has_abstract"] for row in rows),
        "n_empty_fulltext": sum(not row["has_fulltext"] for row in rows),
        "n_abstract_over_limit": sum(row["abstract_characters"] > limit for row in rows),
        "exact_prefix": sum(row["exact_prefix"] for row in rows),
        "exact_full": sum(row["exact_full"] for row in rows),
        "n_not_contained": sum(not row["exact_prefix"] for row in rows),
        "n_full_but_not_prefix": sum(row["exact_full"] and not row["exact_prefix"] for row in rows),
        "max_abstract_characters": max((row["abstract_characters"] for row in rows), default=0),
        "min_abstract_start": min(starts, default=None),
        "max_abstract_start": max(starts, default=None),
        "max_abstract_end_exclusive": max(ends, default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--check-reference", action="store_true", help="Compare both outputs to the committed evidence."
    )
    args = parser.parse_args()
    if args.output_dir.resolve() == REFERENCE.resolve():
        parser.error("Choose a separate output directory; do not overwrite the committed evidence.")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=REVISION,
            filename=FILENAME,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    if file_sha256(path) != PARQUET_SHA256:
        raise ValueError("Dataset file SHA-256 differs from the pinned evidence artifact.")
    data = pq.read_table(path, columns=["pmid", "abstract", "fulltext_processed", "reactions"]).to_pylist()
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} test rows; found {len(data)}.")
    limit, panels = baseline_settings()
    rows = [check_article(row, index, limit, panels) for index, row in enumerate(data)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_article.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "filename": FILENAME,
        "parquet_sha256": PARQUET_SHA256,
        "truncation_limit_characters": limit,
        "llm_input": "fulltext_processed[:TRUNCATION_LIMIT]",
        "method": "bool(abstract.strip()) and abstract in fulltext_processed[:TRUNCATION_LIMIT]",
        "reaction_panels": panels,
        "per_article_csv_sha256": file_sha256(csv_path),
        "first_250": summarize(rows[:250], limit),
        "all_test": summarize(rows, limit),
        "panels_first_250": {
            name: summarize([row for row in rows[:250] if row[f"{name}_positive"]], limit) for name in panels
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.check_reference:
        expected = json.loads((REFERENCE / "summary.json").read_text(encoding="utf-8"))
        # Git may use CRLF on Windows; compare the same CSV content with universal newlines.
        if summary != expected or csv_path.read_text(encoding="utf-8") != (REFERENCE / "per_article.csv").read_text(
            encoding="utf-8"
        ):
            raise ValueError("Reproduced summary or per-article CSV differs from the committed evidence.")
        print("Reference comparison passed: summary.json and every per_article.csv row match.")
    return int(any(not row["exact_prefix"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
