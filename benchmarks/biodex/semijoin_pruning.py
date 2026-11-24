"""Abstract-chunk E5/BioLORD ordering and BioLORD pruning with the existing early-stop executor."""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODEL = "FremyCompany/BioLORD-2023"
MODELS = {"e5": "intfloat/e5-base-v2", "biolord": MODEL}
CHUNK_TOKENS = 128
REPRESENTATION = "abstract_chunk_max"


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
    columns = ["_qid", "patient_description", "reactions_list"]
    if "abstract" in articles.columns:
        columns.append("abstract")
    return digest(articles[columns].to_dict("records"))


def ordering(scores: list[float | None]) -> list[int]:
    # Invalid embeddings keep the article and its original candidate order.
    if any(score is None or not math.isfinite(score) for score in scores):
        return list(range(len(scores)))
    return sorted(range(len(scores)), key=lambda index: -scores[index])


def maximum(scores: list[float | None]) -> float | None:
    if not scores or any(score is None or not math.isfinite(score) for score in scores):
        return None
    return max(scores)


def keep_article(scores: list[float | None], threshold: float | None) -> bool:
    score = maximum(scores)
    return threshold is None or score is None or score >= threshold


def load_scores(path: Path) -> dict:
    data = read_json(path)
    if data["scorer"].get("representation") != REPRESENTATION or data["scorer"]["model"] not in MODELS.values():
        raise ValueError("Regenerate scores with the shared abstract-chunk representation.")
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


def compare_baselines(scores: dict, paths: list[Path], baseline) -> dict:
    """Compare recorded settings, without treating missing historical provenance as verified."""
    expected = {
        "reaction_set": baseline.REACTION_SET,
        "prompt": baseline.PROMPT,
        "reactions": scores["reactions"],
        "n_articles": len(scores["articles"]),
        "n_reactions": len(scores["reactions"]),
        "n_gold": sum(bool(row["gold_matches"]) for row in scores["articles"]),
        "model": baseline.lm_config()["model"],
    }
    methods, sources = set(), {}
    for path in paths:
        summary = read_json(path)
        runs = summary.get("runs") if isinstance(summary, dict) else None
        if not isinstance(runs, dict) or not {"join_full", "early_stop"}.intersection(runs):
            raise ValueError(f"{path}: no join_full or early_stop result; cascade is not experiment 2.")
        checked, missing = [], []

        def check(values, key, target, prefix=""):
            field = prefix + key
            if key not in values:
                missing.append(field)
            elif type(values[key]) is not type(target) or values[key] != target:
                raise ValueError(f"{path}: {field} differs: recorded {values[key]!r}, current {target!r}.")
            else:
                checked.append(field)

        for key in ("reaction_set", "prompt", "reactions"):
            check(summary, key, expected[key])
        found = sorted({"join_full", "early_stop"}.intersection(runs))
        for method in found:
            values = runs[method]
            if not isinstance(values, dict) or type(values.get("n_articles")) is not int or values["n_articles"] <= 0:
                raise ValueError(f"{path}: {method} must record a positive integer n_articles.")
            check(values, "method", method, f"runs.{method}.")
            for key in ("n_articles", "n_reactions", "n_gold", "model"):
                check(values, key, expected[key], f"runs.{method}.")
        methods.update(found)
        sources[digest(summary)] = {
            "summary_sha256": digest(summary),
            "methods": found,
            "checked_fields": checked,
            "missing_fields": missing,
            "ignored_methods": sorted(set(runs) - set(found)),
        }
    return {
        "status": "reported_fields_match_only",
        "current": expected,
        "sources": [sources[key] for key in sorted(sources)],
        "missing_methods": sorted({"join_full", "early_stop"} - methods),
        "unverified": [
            "historical_article_ids_and_exact_input_text",
            "historical_dataset_revision",
            "historical_prompt_text_and_parser_settings",
            "historical_model_weights_and_server_configuration",
            "historical_timing_scope_and_cache_state",
        ],
    }


def check_baselines(args) -> None:
    import semijoin_baseline as baseline

    distinct_output(args.output, [args.scores, *args.baseline_summary])
    if args.output.exists():
        raise ValueError("Comparison output already exists; choose a new path.")
    scores = load_scores(args.scores)
    if scores["split"] != "test" or scores["reactions"] != baseline.REACTION_SETS[baseline.REACTION_SET]:
        raise ValueError("Compare baselines against test scores for the current reaction panel.")
    report = compare_baselines(scores, args.baseline_summary, baseline)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


def fits_chunk(text: str, tokenizers: dict) -> bool:
    return all(
        len(
            tokenizer.encode(
                ("passage: " if name == "e5" else "") + text.strip(),
                truncation=False,
                add_special_tokens=True,
                verbose=False,
            )
        )
        <= CHUNK_TOKENS
        for name, tokenizer in tokenizers.items()
    )


def tokenizer_digest(tokenizer) -> str:
    # Fast tokenizer state includes vocabulary, normalization, and special-token handling.
    state = json.loads(tokenizer.backend_tokenizer.to_str())
    for key in ("padding", "truncation"):
        state.pop(key, None)  # Mutable batching settings do not change our untruncated encoding.
    return digest(state)


def abstract_chunks(text: str, tokenizers: dict) -> list[str]:
    """Preserve the full abstract, using identical spans that fit both embedding models."""
    if not text.strip():
        return []
    chunks, start = [], 0
    while start < len(text):
        low, high, end = start + 1, len(text), start
        while low <= high:
            middle = (low + high) // 2
            if fits_chunk(text[start:middle], tokenizers):
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError("Even a single character exceeds the shared embedding token budget.")
        # Prefer a word boundary, but recheck because token counts need not be monotonic.
        boundary = text.rfind(" ", start, end) + 1 if end < len(text) else end
        if boundary > start and text[start:boundary].strip() and fits_chunk(text[start:boundary], tokenizers):
            end = boundary
        chunks.append(text[start:end])
        start = end
    return chunks


def make_scores(args) -> None:
    if args.output.exists():
        raise ValueError("Score output already exists; reuse it or choose a new path.")
    import semijoin_baseline as baseline
    from transformers import AutoTokenizer

    from lotus.models import SentenceTransformersRM

    articles = baseline.load_articles(args.n_articles, split=args.split, include_abstract=True)
    reactions = baseline.REACTION_SETS[baseline.REACTION_SET]
    if articles.empty:
        raise ValueError("The requested article set is empty.")
    started = time.perf_counter()
    model_name = getattr(args, "model", "biolord")
    rm = SentenceTransformersRM(model=MODELS[model_name], device=args.device, normalize_embeddings=True)
    if rm.transformer.max_seq_length < CHUNK_TOKENS:
        raise ValueError("The embedding model input limit is smaller than the common chunk budget.")
    tokenizers = {
        name: rm.transformer.tokenizer if name == model_name else AutoTokenizer.from_pretrained(model)
        for name, model in MODELS.items()
    }
    chunks = [abstract_chunks(text, tokenizers) for text in articles["abstract"]]
    flat_chunks = [text for group in chunks for text in group]
    label_texts = [("query: " if model_name == "e5" else "") + label for label in reactions]
    if any(
        len(tokenizers[model_name].encode(text, add_special_tokens=True, truncation=False))
        > rm.transformer.max_seq_length
        for text in label_texts
    ):
        raise ValueError("A reaction label exceeds the embedding model input limit.")
    labels = rm(label_texts)
    invalid_labels = ~np.isfinite(labels).all(axis=1) | (np.linalg.norm(labels, axis=1) == 0)
    similarities = np.full((len(articles), len(reactions)), np.nan)
    if flat_chunks:
        vectors = rm([("passage: " if model_name == "e5" else "") + text.strip() for text in flat_chunks])
        chunk_scores = vectors @ labels.T
        # Missing evidence in any chunk must not become a reason to prune an article.
        invalid_chunks = ~np.isfinite(vectors).all(axis=1) | (np.linalg.norm(vectors, axis=1) == 0)
        offset = 0
        for index, group in enumerate(chunks):
            stop = offset + len(group)
            if group and not invalid_chunks[offset:stop].any():
                similarities[index] = chunk_scores[offset:stop].max(axis=0)
            offset = stop
        similarities[:, invalid_labels] = np.nan
    elapsed = time.perf_counter() - started
    rows = []
    for row, scores, group in zip(articles.to_dict("records"), similarities, chunks):
        rows.append(
            {
                "_qid": row["_qid"],
                "scores": [float(score) if np.isfinite(score) else None for score in scores],
                "n_chunks": len(group),
                "gold_matches": [index for index, label in enumerate(reactions) if label in row["reactions_list"]],
            }
        )
    save_json(
        args.output,
        {
            "scorer": {
                "model": MODELS[model_name],
                "revision": getattr(rm.transformer[0].auto_model.config, "_commit_hash", None),
                "max_seq_length": rm.transformer.max_seq_length,
                "normalize_embeddings": True,
                "representation": REPRESENTATION,
                "chunking": {
                    "version": 1,
                    "token_budget": CHUNK_TOKENS,
                    "overlap": 0,
                    "tokenizers": {
                        name: {"model": MODELS[name], "sha256": tokenizer_digest(tokenizer)}
                        for name, tokenizer in tokenizers.items()
                    },
                },
                "document_prefix": "passage: " if model_name == "e5" else "",
                "label_prefix": "query: " if model_name == "e5" else "",
            },
            "split": args.split,
            "input_sha256": article_digest(articles),
            "chunks_sha256": digest(chunks),
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
    if scores["scorer"]["model"] != MODEL:
        raise ValueError("Experiment 5 calibration requires BioLORD scores.")
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


def threshold_from_args(args, scores: dict) -> tuple[float | None, dict]:
    experiment = getattr(args, "experiment", 5)
    expected_model = MODELS["e5"] if experiment == 3 else MODEL
    if scores["scorer"]["model"] != expected_model:
        raise ValueError(f"Experiment {experiment} requires {expected_model} scores.")
    if experiment in (3, 4):
        if args.calibration is not None or args.threshold is not None:
            raise ValueError("Experiments 3 and 4 use ordering only; do not supply a threshold or calibration.")
        return None, {"kind": "disabled"}
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
                raise ValueError(f"Article {qid} does not match the recorded proxy order.")
        by_id[qid] = record
    return by_id


def apply_gate(scores: dict, records: list[dict], threshold: float | None) -> list[dict]:
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
    articles = baseline.load_articles(len(scores["articles"]), split=scores["split"], include_abstract=True)
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
        "experiment": getattr(args, "experiment", 5),
        "scores_sha256": digest(scores),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "model": model_config["model"],
        "llm_server_id": os.environ.get("LLM_SERVER_ID"),
        "instruction": baseline.JOIN_INSTRUCTION,
        "max_tokens": model_config["max_tokens"],
        "temperature": model_config["temperature"],
        "workers": baseline.EARLY_WORKERS,
        "llm_batch": model_config["max_batch_size"],
        "endpoint_sha256": digest(
            {"explicit": model_config.get("api_base"), "provider_environment": provider_endpoints}
        ),
    }
    if getattr(args, "baseline_summary", None):
        manifest["baseline_comparison"] = compare_baselines(scores, args.baseline_summary, baseline)
    manifest_path = args.output_dir / "manifest.json"
    resumed = manifest_path.exists()
    if resumed:
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
        if (
            summary.get("manifest_sha256") != digest(manifest)
            or summary.get("experiment") != manifest["experiment"]
            or summary.get("threshold") != threshold
            or any(summary.get(key) != value for key, value in summarize(records).items())
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
            "manifest_sha256": digest(manifest),
            "experiment": manifest["experiment"],
            "threshold": threshold,
            "embedding_time_s": scores["embedding_time_s"],
            "invocation_time_s": elapsed,
            "resumed": resumed,
            "estimated_embedding_plus_run_time_s": None if resumed else scores["embedding_time_s"] + elapsed,
            "lm_calls_kind": "logical_pair_evaluations",
        }
    )
    save_json(args.output_dir / "articles.json", records)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def completed_run(scores: dict, directory: Path) -> tuple[dict, list[dict], dict]:
    """Read a completed live run without repairing or modifying its artifacts."""
    manifest = read_json(directory / "manifest.json")
    if manifest["scores_sha256"] != digest(scores):
        raise ValueError("Replay requires exactly the same scores as the source run.")
    source_threshold = manifest["threshold"]
    source_progress = read_progress(directory / "progress.jsonl", repair=False)
    expected_ids = {row["_qid"] for row in scores["articles"] if keep_article(row["scores"], source_threshold)}
    if {row["_qid"] for row in source_progress} != expected_ids or any(
        row.get("pruned", False) for row in source_progress
    ):
        raise ValueError("Replay requires the completed source run's evaluated progress.")
    source = apply_gate(scores, source_progress, source_threshold)
    source_summary = read_json(directory / "summary.json")
    if (
        read_json(directory / "articles.json") != source
        or source_summary.get("manifest_sha256") != digest(manifest)
        or source_summary.get("experiment") != manifest["experiment"]
        or source_summary.get("threshold") != source_threshold
        or any(source_summary.get(key) != value for key, value in summarize(source).items())
    ):
        raise ValueError("Source results disagree with their original threshold or completed progress.")
    return manifest, source, source_summary


def replay(args) -> None:
    inputs = [
        args.scores,
        *(args.source_dir / name for name in ["manifest.json", "progress.jsonl", "articles.json", "summary.json"]),
    ]
    if args.calibration is not None:
        inputs.append(args.calibration)
    distinct_output(args.output, inputs)
    scores = load_scores(args.scores)
    manifest, source, _ = completed_run(scores, args.source_dir)
    threshold, threshold_source = threshold_from_args(args, scores)
    records = apply_gate(scores, source, threshold)
    summary = summarize(records)
    summary.update(
        {
            "experiment": 5,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "replay": True,
            "estimated_embedding_plus_run_time_s": None,
            "lm_calls_kind": "replayed_logical_pair_evaluations",
        }
    )
    save_json(args.output, {"source_manifest": manifest, "summary": summary, "articles": records})
    print(json.dumps(summary, indent=2))


def comparison_prf(pred: set[int], reference: set[int]) -> dict:
    import semijoin_baseline as baseline

    tp = len(pred & reference)
    return {
        **baseline.prf(pred, reference),
        "tp": tp,
        "fp": len(pred - reference),
        "fn": len(reference - pred),
    }


def report(args) -> None:
    """Standardize saved results after evaluation; never call the LLM or load the dataset."""
    import semijoin_baseline as baseline

    inputs = [args.scores, *args.baseline_summary, *args.replay]
    if args.output.exists():
        raise ValueError("Report output already exists; choose a new path.")
    reference = load_scores(args.scores)
    if reference["split"] != "test":
        raise ValueError("Report requires test scores as the gold reference.")
    universe = {row["_qid"] for row in reference["articles"]}
    gold = {row["_qid"] for row in reference["articles"] if row["gold_matches"]}

    def cohort(scores):
        return (
            scores["split"], scores["input_sha256"], scores["chunks_sha256"],
            scores["reactions"], scores["scorer"]["chunking"],
            [(row["_qid"], row["gold_matches"]) for row in scores["articles"]],
        )

    def selected(records):
        return {row["_qid"] for row in records if row["matched"] is not None}

    # Each run is paired with its original score file; E5 and BioLORD hashes differ.
    live = []
    for score_path, directory in args.runs:
        inputs.extend([score_path, *(directory / name for name in
                                    ("manifest.json", "progress.jsonl", "articles.json", "summary.json"))])
        scores = load_scores(score_path)
        if cohort(scores) != cohort(reference):
            raise ValueError("Report score files have different articles, gold labels, reactions, or chunks.")
        manifest, records, summary = completed_run(scores, directory)
        threshold_from_args(SimpleNamespace(experiment=manifest["experiment"], threshold=manifest["threshold"],
                                            calibration=None), scores)
        live.append((manifest, records, summary, directory, scores))
    keys = ("model", "instruction", "temperature", "max_tokens", "workers", "llm_batch",
            "endpoint_sha256", "llm_server_id")
    if live and any(any(m.get(key) != live[0][0].get(key) for key in keys) for m, *_ in live):
        raise ValueError("Live runs use different LLM, prompt, server, or execution settings.")
    if len({m["scores_sha256"] for m, *_ in live if m["experiment"] in (4, 5)}) > 1:
        raise ValueError("Experiments 4 and 5 must reuse exactly the same BioLORD score file.")
    summaries = [read_json(path) for path in args.baseline_summary]

    def recorded_value(key, values):
        found = [value[key] for value in values if key in value]
        if found and any(value != found[0] for value in found):
            raise ValueError(f"Historical summaries disagree on {key}.")
        return found[0] if found else None

    if live:
        model = live[0][0]["model"]
        prompts = {baseline.PAPER_JOIN_INSTRUCTION: "paper", baseline.ADR_JOIN_INSTRUCTION: "adr"}
        if live[0][0]["instruction"] not in prompts:
            raise ValueError("The saved prompt is not a supported baseline instruction.")
        prompt = prompts[live[0][0]["instruction"]]
    else:
        model = recorded_value("model", [run for item in summaries for name, run in item["runs"].items()
                                          if name in ("join_full", "early_stop")])
        prompt = recorded_value("prompt", summaries)
    panel = next((name for name, labels in baseline.REACTION_SETS.items() if labels == reference["reactions"]), None)
    if panel is None:
        raise ValueError("The reference reaction panel is not a baseline panel.")
    context = SimpleNamespace(REACTION_SET=panel, PROMPT=prompt, lm_config=lambda: {"model": model})
    comparison = compare_baselines(reference, args.baseline_summary, context)
    summary_hashes = {digest(item) for item in summaries}
    for manifest, *_ in live:
        bound = manifest.get("baseline_comparison")
        if bound and not summary_hashes.issubset({item["summary_sha256"] for item in bound["sources"]}):
            raise ValueError("Historical summaries differ from the live run's recorded baseline references.")

    historic = {}

    def check_saved_metrics(saved, pred, reference_ids, source):
        if type(saved.get("n_pred")) is not int or saved["n_pred"] != len(pred):
            raise ValueError(f"{source}: selected IDs disagree with saved n_pred.")
        # Validate against the original writer's floating-point formula before normalizing it.
        expected = baseline.prf(pred, reference_ids)
        for key in ("precision", "recall", "f1"):
            if key in saved and (type(saved[key]) not in (int, float) or saved[key] != expected[key]):
                raise ValueError(f"{source}: selected IDs disagree with saved {key}.")

    for path, source in zip(args.baseline_summary, summaries):
        for method, experiment in (("join_full", 1), ("early_stop", 2)):
            if method not in source["runs"]:
                continue
            summary = source["runs"][method]
            csv_path = path.parent / f"{method}_semi.csv"
            if method == "join_full" and not csv_path.exists():
                csv_path = path.parent / "join_full_join.csv"
            inputs.append(csv_path)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if "_qid" not in (reader.fieldnames or []):
                    raise ValueError(f"{csv_path}: missing _qid column.")
                pred = {int(row["_qid"]) for row in reader}
            if not pred.issubset(universe):
                raise ValueError(f"{csv_path}: selected article IDs are outside the reference cohort.")
            check_saved_metrics(summary.get("vs_labels", {}), pred, gold, csv_path)
            value = (digest(source), pred, summary, path)
            if experiment in historic and historic[experiment][:3] != value[:3]:
                raise ValueError(f"Conflicting experiment {experiment} results were supplied.")
            historic[experiment] = value
    full = historic[1][1] if 1 in historic else None
    if full is not None and 2 in historic and "vs_join_full" in historic[2][2]:
        check_saved_metrics(historic[2][2]["vs_join_full"], historic[2][1], full, historic[2][3])
    results = []

    def add_result(experiment, mode, pred, summary, source, threshold=None, threshold_source=None):
        calls = summary.get("lm_calls")
        if calls is not None and (type(calls) is not int or calls < 0):
            raise ValueError("LLM calls must be a nonnegative integer or absent.")
        results.append({
            "experiment": experiment, "mode": mode, "source": str(source.resolve()), "threshold": threshold,
            "threshold_source": threshold_source,
            "n_articles": len(universe), "n_positive": len(gold), "n_selected": len(pred),
            "lm_calls": calls, "lm_calls_kind": "replayed_logical_pair_evaluations" if mode == "replay"
            else "logical_pair_evaluations",
            "vs_labels": comparison_prf(pred, gold),
            "vs_join_full": comparison_prf(pred, full) if full is not None else None,
        })

    for experiment, (_, pred, summary, path) in sorted(historic.items()):
        add_result(experiment, "live", pred, summary, path)
    for manifest, records, summary, directory, _ in live:
        add_result(manifest["experiment"], "live", selected(records), summary, directory, manifest["threshold"],
                   manifest.get("threshold_source"))
    for path in args.replay:
        saved = read_json(path)
        sources = [item for item in live if item[0] == saved["source_manifest"]]
        if not sources:
            raise ValueError("Replay report requires its verified source live run via --run.")
        if any(item[1] != sources[0][1] for item in sources):
            raise ValueError("Replay source is ambiguous; supply only its original live run.")
        manifest, records, _, _, scores = sources[0]
        summary = saved["summary"]
        threshold_from_args(SimpleNamespace(experiment=5, threshold=summary["threshold"], calibration=None), scores)
        replayed = apply_gate(scores, records, summary["threshold"])
        if (saved["articles"] != replayed or summary.get("experiment") != 5 or summary.get("replay") is not True
                or summary.get("lm_calls_kind") != "replayed_logical_pair_evaluations"
                or any(summary.get(key) != value for key, value in summarize(replayed).items())):
            raise ValueError("Replay results disagree with their verified source decisions.")
        add_result(5, "replay", selected(replayed), summary, path, summary["threshold"], summary.get("threshold_source"))
    if not results:
        raise ValueError("Supply at least one baseline summary or completed --run.")
    distinct_output(args.output, inputs)
    save_json(args.output, {
        "reference_scores_sha256": digest(reference),
        "results": results,
        "missing_experiments": sorted(set(range(1, 6)) - {row["experiment"] for row in results}),
        "baseline_comparison": comparison,
        "notes": [
            "All precision/recall/F1 values are recomputed from article ID sets and rounded once to four decimals.",
            "vs_join_full is null when experiment 1 is not supplied; an empty supplied full result is not missing.",
            "Historical row IDs/counts do not prove identical source articles, prompts, weights, or server state.",
            "Timing is not standardized by this report; retain original scope-labelled timing artifacts.",
            "Replay LLM counts are simulated policy costs; aggregation and replay make no additional LLM calls.",
        ],
    })
    print(json.dumps(read_json(args.output), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Embed abstract chunks and target labels; no LLM calls.")
    score.add_argument("--model", choices=list(MODELS), default="biolord")
    score.add_argument("--split", choices=["validation", "test"], required=True)
    score.add_argument("--n-articles", type=int)
    score.add_argument("--device", default="cpu")
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(action=make_scores)
    calibration = commands.add_parser("calibrate", help="Choose two thresholds from validation labels; no LLM calls.")
    calibration.add_argument("scores", type=Path)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(action=calibrate_file)
    comparison = commands.add_parser("check-baselines", help="Compare recorded experiment 1/2 settings; no LLM calls.")
    comparison.add_argument("scores", type=Path)
    comparison.add_argument("--baseline-summary", type=Path, action="append", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.set_defaults(action=check_baselines)
    reporting = commands.add_parser("report", help="Recompute common experiment 1-5 metrics from saved artifacts; no LLM calls.")
    reporting.add_argument("--scores", type=Path, required=True)
    reporting.add_argument("--baseline-summary", type=Path, action="append", default=[])
    reporting.add_argument("--run", dest="runs", type=Path, nargs=2, action="append", default=[], metavar=("SCORES", "DIR"))
    reporting.add_argument("--replay", type=Path, action="append", default=[])
    reporting.add_argument("--output", type=Path, required=True)
    reporting.set_defaults(action=report)
    for name, action in [("run", run), ("replay", replay)]:
        command = commands.add_parser(name)
        command.add_argument("scores", type=Path)
        thresholds = command.add_mutually_exclusive_group()
        thresholds.add_argument("--threshold", type=float)
        thresholds.add_argument("--calibration", type=Path)
        command.add_argument("--policy", choices=["conservative", "retain_95"], default="conservative")
        if name == "run":
            command.add_argument("--experiment", type=int, choices=[3, 4, 5], default=5)
            command.add_argument("--baseline-summary", type=Path, action="append")
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
