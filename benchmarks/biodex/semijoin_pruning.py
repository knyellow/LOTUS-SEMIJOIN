"""BioLORD article-level negative pruning using the existing early-stop executor."""

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

MODEL = "FremyCompany/BioLORD-2023"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def distinct_output(output: Path, inputs: list[Path]) -> None:
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must differ from the input and source result files.")


def article_digest(articles) -> str:
    return digest(articles[["_qid", "patient_description", "reactions_list"]].to_dict("records"))


def ordering(scores: list[float | None]) -> list[int]:
    # Invalid embeddings keep the article and its original candidate order.
    if any(score is None or not math.isfinite(score) for score in scores):
        return list(range(len(scores)))
    return sorted(range(len(scores)), key=lambda index: -scores[index])


def maximum(scores: list[float | None]) -> float | None:
    if not scores or any(score is None or not math.isfinite(score) for score in scores):
        return None
    return max(scores)


def keep_article(scores: list[float | None], threshold: float) -> bool:
    score = maximum(scores)
    return score is None or score >= threshold


def load_scores(path: Path) -> dict:
    data = read_json(path)
    rows, reactions = data["articles"], data["reactions"]
    if not rows or not reactions or len(reactions) != len(set(reactions)):
        raise ValueError("Scores require articles and distinct reaction labels.")
    if [row["_qid"] for row in rows] != list(range(len(rows))):
        raise ValueError("Scores must cover every original article in order.")
    for row in rows:
        if len(row["scores"]) != len(reactions):
            raise ValueError("Every article must have a score for every reaction.")
        if any(index not in range(len(reactions)) for index in row["gold_matches"]):
            raise ValueError("Gold reaction index is outside the candidate list.")
    return data


def make_scores(args) -> None:
    import semijoin_baseline as baseline

    from lotus.models import SentenceTransformersRM

    articles = baseline.load_articles(args.n_articles, split=args.split)
    reactions = baseline.REACTION_SETS[baseline.REACTION_SET]
    if articles.empty:
        raise ValueError("The requested article set is empty.")
    started = time.perf_counter()
    rm = SentenceTransformersRM(model=MODEL, device=args.device, normalize_embeddings=True)
    # Use the model's native input length; record it so ordering and pruning share it.
    vectors = rm(articles["patient_description"].tolist())
    labels = rm(reactions)
    similarities = vectors @ labels.T
    # A zero vector has no defined cosine similarity, even after normalization.
    invalid_articles = ~np.isfinite(vectors).all(axis=1) | (np.linalg.norm(vectors, axis=1) == 0)
    invalid_labels = ~np.isfinite(labels).all(axis=1) | (np.linalg.norm(labels, axis=1) == 0)
    similarities[invalid_articles, :] = np.nan
    similarities[:, invalid_labels] = np.nan
    elapsed = time.perf_counter() - started
    rows = []
    for row, scores in zip(articles.to_dict("records"), similarities):
        rows.append(
            {
                "_qid": row["_qid"],
                "scores": [float(score) if np.isfinite(score) else None for score in scores],
                "gold_matches": [index for index, label in enumerate(reactions) if label in row["reactions_list"]],
            }
        )
    save_json(
        args.output,
        {
            "scorer": {
                "model": MODEL,
                "revision": getattr(rm.transformer[0].auto_model.config, "_commit_hash", None),
                "max_seq_length": rm.transformer.max_seq_length,
                "normalize_embeddings": True,
                "representation": "article_prefix",
                "truncation_limit": baseline.TRUNCATION_LIMIT,
            },
            "split": args.split,
            "input_sha256": article_digest(articles),
            "reactions": reactions,
            "embedding_time_s": elapsed,
            "embedding_device": args.device,
            "articles": rows,
        },
    )


def calibrate(scores: dict, target: float) -> dict:
    positives = [row for row in scores["articles"] if row["gold_matches"]]
    finite_positive_scores = sorted(value for row in positives if (value := maximum(row["scores"])) is not None)
    if not positives or not finite_positive_scores:
        raise ValueError("Calibration requires positive articles with finite scores.")
    # An observed score boundary avoids interpolated quantiles exceeding the miss budget.
    allowed_misses = len(positives) - math.ceil(target * len(positives))
    threshold = finite_positive_scores[min(allowed_misses, len(finite_positive_scores) - 1)]
    pruned = [row for row in scores["articles"] if not keep_article(row["scores"], threshold)]
    missed = sum(bool(row["gold_matches"]) for row in pruned)
    return {
        "threshold": threshold,
        "target_retention": target,
        "n_positive": len(positives),
        "n_pruned": len(pruned),
        "n_positive_pruned": missed,
        "observed_positive_retention": 1 - missed / len(positives),
    }


def calibrate_file(args) -> None:
    distinct_output(args.output, [args.scores])
    scores = load_scores(args.scores)
    if scores["split"] != "validation":
        raise ValueError("Choose thresholds on validation scores before evaluating test data.")
    save_json(
        args.output,
        {
            "scorer": scores["scorer"],
            "reactions": scores["reactions"],
            "validation_input_sha256": scores["input_sha256"],
            "policies": {"conservative": calibrate(scores, 1.0), "retain_95": calibrate(scores, 0.95)},
        },
    )


def threshold_from_args(args, scores: dict) -> tuple[float, dict]:
    if args.calibration is None:
        threshold = args.threshold
        source = {"kind": "manual"}
    else:
        calibration = read_json(args.calibration)
        if calibration["scorer"] != scores["scorer"] or calibration["reactions"] != scores["reactions"]:
            raise ValueError("Calibration must use the same model, representation, and reaction panel.")
        threshold = calibration["policies"][args.policy]["threshold"]
        source = {
            "kind": "validation_calibration",
            "policy": args.policy,
            "calibration_sha256": digest(calibration),
            "validation_input_sha256": calibration["validation_input_sha256"],
        }
    if threshold is None or not math.isfinite(threshold):
        raise ValueError("A finite threshold or calibration file is required.")
    return threshold, source


def read_progress(path: Path, repair: bool = True) -> list[dict]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair or index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Progress contains a corrupt completed record.") from None
            # A killed writer may leave only its final record incomplete.
            with path.open("r+b") as stream:
                stream.truncate(len(payload) - len(line))
            return records
    if repair and payload and not payload.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records


def validate_records(scores: dict, records: list[dict]) -> dict:
    rows = {row["_qid"]: row for row in scores["articles"]}
    by_id = {}
    for record in records:
        qid = record["_qid"]
        if qid in by_id or qid not in rows:
            raise ValueError("Progress contains duplicate or unknown article IDs.")
        calls, matched = record["lm_calls"], record["matched"]
        if record.get("pruned", False):
            if calls != 0 or matched is not None:
                raise ValueError(f"Pruned article {qid} must have no model evaluations.")
        else:
            order = ordering(rows[qid]["scores"])
            if type(calls) is not int or not 1 <= calls <= len(order):
                raise ValueError(f"Invalid call count for article {qid}.")
            if matched is None and calls != len(order):
                raise ValueError(f"Article {qid} stopped without evaluating all candidates.")
            if matched is not None and matched != scores["reactions"][order[calls - 1]]:
                raise ValueError(f"Article {qid} does not match the recorded BioLORD order.")
        by_id[qid] = record
    return by_id


def apply_gate(scores: dict, records: list[dict], threshold: float) -> list[dict]:
    by_id = validate_records(scores, records)
    output = []
    for row in scores["articles"]:
        qid = row["_qid"]
        pruned = not keep_article(row["scores"], threshold)
        record = by_id.get(qid)
        if not pruned:
            if record is None or record.get("pruned", False):
                raise ValueError(f"Article {qid} has no evaluated result for this threshold.")
            calls = record["lm_calls"]
            matched = record["matched"]
        else:
            calls, matched = 0, None
        gold_ranks = [rank for rank, index in enumerate(ordering(row["scores"]), 1) if index in row["gold_matches"]]
        output.append(
            {
                "_qid": qid,
                "matched": matched,
                "lm_calls": calls,
                "pruned": pruned,
                "max_score": maximum(row["scores"]),
                "gold_positive": bool(row["gold_matches"]),
                "first_gold_witness_rank": min(gold_ranks) if gold_ranks else None,
            }
        )
    return output


def summarize(records: list[dict]) -> dict:
    tp = sum(row["gold_positive"] and row["matched"] is not None for row in records)
    selected = sum(row["matched"] is not None for row in records)
    positives = sum(row["gold_positive"] for row in records)
    ranks = [row["lm_calls"] for row in records if row["matched"] is not None]
    missed = sum(row["gold_positive"] and row["pruned"] for row in records)
    return {
        "n_articles": len(records),
        "n_positive": positives,
        "n_selected": selected,
        "n_pruned": sum(row["pruned"] for row in records),
        "n_positive_pruned": missed,
        "positive_retention": 1 - missed / positives if positives else None,
        "tp": tp,
        "fp": selected - tp,
        "fn": positives - tp,
        "precision": tp / selected if selected else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": 2 * tp / (selected + positives) if selected + positives else 0.0,
        "lm_calls": sum(row["lm_calls"] for row in records),
        "mean_first_llm_witness_rank": sum(ranks) / len(ranks) if ranks else None,
    }


def set_file_lock(stream, release: bool) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def output_lock(directory: Path):
    # The OS releases this lock when a server job exits, including after a crash.
    with (directory / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            set_file_lock(stream, release=False)
        except OSError as error:
            raise ValueError("Another process is using this output directory.") from error
        try:
            yield
        finally:
            set_file_lock(stream, release=True)


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output_dir):
        run_locked(args)


def run_locked(args) -> None:
    import semijoin_baseline as baseline

    started = time.perf_counter()
    scores = load_scores(args.scores)
    threshold, threshold_source = threshold_from_args(args, scores)
    if scores["split"] != "test":
        raise ValueError("Run the LLM evaluation on test scores.")
    articles = baseline.load_articles(len(scores["articles"]), split=scores["split"])
    if (
        baseline.REACTION_SETS[baseline.REACTION_SET] != scores["reactions"]
        or article_digest(articles) != scores["input_sha256"]
    ):
        raise ValueError("The loaded dataset or reaction panel differs from the score file.")
    model_config = baseline.lm_config()
    provider = model_config["model"].split("/", 1)[0].upper() if "/" in model_config["model"] else "OPENAI"
    if model_config["model"].startswith("claude"):
        provider = "ANTHROPIC"
    provider_endpoints = {
        key: os.environ.get(key)
        for key in (f"{provider}_API_BASE", f"{provider}_BASE_URL", f"{provider}_API_VERSION")
        if "api_base" not in model_config
    }
    manifest = {
        "scores_sha256": digest(scores),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "model": model_config["model"],
        "instruction": baseline.JOIN_INSTRUCTION,
        "max_tokens": model_config["max_tokens"],
        "temperature": model_config["temperature"],
        "workers": baseline.EARLY_WORKERS,
        "llm_batch": model_config["max_batch_size"],
        "endpoint_sha256": digest(
            {"explicit": model_config.get("api_base"), "provider_environment": provider_endpoints}
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Output directory belongs to a different configuration; choose a new directory.")
    else:
        if any(path.name != ".run.lock" for path in args.output_dir.iterdir()):
            raise ValueError("A new output directory must be empty.")
        save_json(manifest_path, manifest)
    progress_path = args.output_dir / "progress.jsonl"
    previous = read_progress(progress_path)
    validate_records(scores, previous)
    kept = [row["_qid"] for row in scores["articles"] if keep_article(row["scores"], threshold)]
    if any(record["_qid"] not in kept or record.get("pruned", False) for record in previous):
        raise ValueError("Live progress must contain only evaluated, retained articles.")
    summary_path, articles_path = args.output_dir / "summary.json", args.output_dir / "articles.json"
    complete = len(previous) == len(kept)
    if summary_path.exists():
        if not complete:
            raise ValueError("Completed results exist but their progress is incomplete; restore the progress file.")
        records = apply_gate(scores, previous, threshold)
        summary = read_json(summary_path)
        if summary.get("threshold") != threshold or any(
            summary.get(key) != value for key, value in summarize(records).items()
        ):
            raise ValueError("Saved result artifacts disagree with the completed progress.")
        if articles_path.exists() and read_json(articles_path) != records:
            raise ValueError("Saved article results disagree with the completed progress.")
        if not articles_path.exists():
            save_json(articles_path, records)
        print(json.dumps(summary, indent=2))
        return
    orders = {row["_qid"]: ordering(row["scores"]) for row in scores["articles"]}
    if not complete:
        lm = baseline.configure(with_retriever=False)
        baseline.run_early_stop(
            articles[articles["_qid"].isin(kept)], scores["reactions"], lm, str(progress_path), reaction_orders=orders
        )
    records = apply_gate(scores, read_progress(progress_path), threshold)
    elapsed = time.perf_counter() - started
    summary = summarize(records)
    summary.update(
        {
            "threshold": threshold,
            "embedding_time_s": scores["embedding_time_s"],
            "invocation_time_s": elapsed,
            "resumed": bool(previous),
            "estimated_embedding_plus_run_time_s": None if previous else scores["embedding_time_s"] + elapsed,
            "lm_calls_kind": "logical_pair_evaluations",
        }
    )
    save_json(args.output_dir / "articles.json", records)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def replay(args) -> None:
    inputs = [
        args.scores,
        *(args.source_dir / name for name in ["manifest.json", "progress.jsonl", "articles.json", "summary.json"]),
    ]
    if args.calibration is not None:
        inputs.append(args.calibration)
    distinct_output(args.output, inputs)
    scores = load_scores(args.scores)
    manifest = read_json(args.source_dir / "manifest.json")
    if manifest["scores_sha256"] != digest(scores):
        raise ValueError("Replay requires exactly the same scores as the source run.")
    source_threshold = manifest["threshold"]
    source_progress = read_progress(args.source_dir / "progress.jsonl", repair=False)
    expected_ids = {row["_qid"] for row in scores["articles"] if keep_article(row["scores"], source_threshold)}
    if {row["_qid"] for row in source_progress} != expected_ids or any(
        row.get("pruned", False) for row in source_progress
    ):
        raise ValueError("Replay requires the completed source run's evaluated progress.")
    source = apply_gate(scores, source_progress, source_threshold)
    source_summary = read_json(args.source_dir / "summary.json")
    if (
        read_json(args.source_dir / "articles.json") != source
        or source_summary.get("threshold") != source_threshold
        or any(source_summary.get(key) != value for key, value in summarize(source).items())
    ):
        raise ValueError("Source results disagree with their original threshold or completed progress.")
    threshold, threshold_source = threshold_from_args(args, scores)
    records = apply_gate(scores, source, threshold)
    summary = summarize(records)
    summary.update(
        {
            "threshold": threshold,
            "threshold_source": threshold_source,
            "replay": True,
            "estimated_embedding_plus_run_time_s": None,
            "lm_calls_kind": "replayed_logical_pair_evaluations",
        }
    )
    save_json(args.output, {"source_manifest": manifest, "summary": summary, "articles": records})
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Embed each article and target label once; no LLM calls.")
    score.add_argument("--split", choices=["validation", "test"], required=True)
    score.add_argument("--n-articles", type=int)
    score.add_argument("--device", default="cpu")
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(action=make_scores)
    calibration = commands.add_parser("calibrate", help="Choose two thresholds from validation labels; no LLM calls.")
    calibration.add_argument("scores", type=Path)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(action=calibrate_file)
    for name, action in [("run", run), ("replay", replay)]:
        command = commands.add_parser(name)
        command.add_argument("scores", type=Path)
        thresholds = command.add_mutually_exclusive_group(required=True)
        thresholds.add_argument("--threshold", type=float)
        thresholds.add_argument("--calibration", type=Path)
        command.add_argument("--policy", choices=["conservative", "retain_95"], default="conservative")
        if name == "run":
            command.add_argument("--output-dir", type=Path, required=True)
        else:
            command.add_argument("--source-dir", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        command.set_defaults(action=action)
    args = parser.parse_args()
    if getattr(args, "n_articles", None) is not None and args.n_articles <= 0:
        parser.error("--n-articles must be positive.")
    args.action(args)


if __name__ == "__main__":
    main()
