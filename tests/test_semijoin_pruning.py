"""Offline regressions for pruning boundaries, ordering, replay, and resume identity."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "biodex"
spec = importlib.util.spec_from_file_location("semijoin_pruning", BENCHMARK / "semijoin_pruning.py")
pruning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pruning)


def load_baseline(real_lotus: bool = False):
    # Replace only external model/dataset imports; exercise the actual executor.
    modules = {
        "datasets": SimpleNamespace(load_dataset=Mock()),
        "lotus": SimpleNamespace(),
        "lotus.models": SimpleNamespace(LM=object, SentenceTransformersRM=Mock()),
        "lotus.sem_ops.sem_join": SimpleNamespace(sem_join=Mock()),
        "lotus.types": SimpleNamespace(CascadeArgs=object),
        "lotus.vector_store": SimpleNamespace(FaissVS=Mock()),
    }
    if real_lotus:
        modules = {"datasets": modules["datasets"]}
    module_spec = importlib.util.spec_from_file_location("pruning_baseline_test", BENCHMARK / "semijoin_baseline.py")
    baseline = importlib.util.module_from_spec(module_spec)
    with patch.dict(sys.modules, modules):
        module_spec.loader.exec_module(baseline)
    return baseline


class FakeTokenizer:
    def __init__(self, characters_per_token=1):
        self.characters_per_token = characters_per_token
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({"step": characters_per_token}))

    def encode(self, text, **kwargs):
        return [0] * (2 + (len(text) + self.characters_per_token - 1) // self.characters_per_token)


class TestSemijoinPruning(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "split": "test",
            "scorer": {"model": pruning.MODEL, "representation": pruning.REPRESENTATION},
            "reactions": ["A", "B"],
            "articles": [
                {"_qid": 0, "scores": [0.2, 0.8], "gold_matches": [1]},
                {"_qid": 1, "scores": [0.1, 0.3], "gold_matches": [0]},
            ],
            "embedding_time_s": 1.0,
        }

    def baseline_summary(self, baseline):
        return {
            "reaction_set": baseline.REACTION_SET,
            "prompt": baseline.PROMPT,
            "reactions": self.scores["reactions"],
            "runs": {
                method: {
                    "method": method,
                    "n_articles": 2,
                    "n_reactions": 2,
                    "n_gold": 2,
                    "model": baseline.lm_config()["model"],
                }
                for method in ("join_full", "early_stop")
            },
        }

    def report_fixture(self, root):
        baseline = load_baseline()
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        articles = pd.DataFrame([
            {"_qid": 0, "patient_description": "first", "reactions_list": ["B"]},
            {"_qid": 1, "patient_description": "second", "reactions_list": ["A"]},
            {"_qid": 2, "patient_description": "third", "reactions_list": []},
        ])
        baseline.load_articles = Mock(return_value=articles)
        baseline.configure = Mock(return_value=object())

        def pair(*args, **kwargs):
            qid, label = args[2][0], args[3][0]
            return SimpleNamespace(join_results=[(qid, label, None)] if (qid, label) in {(0, 1), (2, 0)} else [])

        baseline.pairwise_join = Mock(side_effect=pair)
        runs = []
        with patch.dict(sys.modules, {"semijoin_baseline": baseline}), contextlib.redirect_stdout(io.StringIO()):
            for experiment in (3, 4, 5):
                scores = json.loads(json.dumps(self.scores))
                scores["input_sha256"] = pruning.article_digest(articles)
                scores["chunks_sha256"] = pruning.digest(["first", "second", "third"])
                scores["scorer"]["chunking"] = {"version": 1, "token_budget": 128, "overlap": 0}
                scores["articles"].append({"_qid": 2, "scores": [0.6, 0.1], "gold_matches": []})
                scores["embedding_time_s"] = float(experiment)
                scores["embedding_device"] = "cpu"
                for row in scores["articles"]:
                    row["n_chunks"] = 1
                if experiment == 3:
                    scores["scorer"]["model"] = pruning.MODELS["e5"]
                    scores["articles"][0]["scores"] = [0.9, 0.1]
                score_path = root / {3: "e5.json", 4: "bio4.json", 5: "bio5.json"}[experiment]
                pruning.save_json(score_path, scores)
                directory = root / f"run{experiment}"
                pruning.run(SimpleNamespace(scores=score_path, output_dir=directory, experiment=experiment,
                                            threshold=0.7 if experiment == 5 else None, calibration=None))
                runs.append((score_path, directory))
        legacy = root / "legacy"
        legacy.mkdir()
        summary = self.baseline_summary(baseline)
        for method, value in summary["runs"].items():
            value.update(n_articles=3, lm_calls=6 if method == "join_full" else 4,
                         vs_labels=baseline.prf({0, 2}, {0, 1}))
            (legacy / f"{method}_semi.csv").write_text("_qid,title\n0,first\n2,third\n", encoding="utf-8")
        pruning.save_json(legacy / "summary.json", summary)
        args = SimpleNamespace(scores=root / "bio4.json", runs=runs, replay=[],
                               baseline_summary=[legacy / "summary.json"], output=root / "comparison.json")
        baseline.configure.reset_mock()
        baseline.load_articles.reset_mock()
        baseline.pairwise_join.reset_mock()
        return baseline, args

    def test_report_unifies_all_five_and_replay_without_lm_or_dataset(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, args = self.report_fixture(root)
            args.replay = [root / "replay.json"]
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.replay(SimpleNamespace(scores=args.scores, source_dir=root / "run4", threshold=0.7,
                                               calibration=None, output=args.replay[0]))
                before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                # Workstation environment defaults must not replace stored experiment conditions.
                baseline.LLM_MODEL, baseline.PROMPT = "other-current-model", "other-current-prompt"
                pruning.report(args)
            result = pruning.read_json(args.output)
            rows = result["results"]
            self.assertEqual([row["experiment"] for row in rows], [1, 2, 3, 4, 5, 5])
            self.assertEqual(result["missing_experiments"], [])
            self.assertTrue(all(set(row) == set(rows[0]) for row in rows))
            self.assertEqual([row["lm_calls"] for row in rows], [6, 4, 5, 4, 1, 1])
            self.assertEqual([row["n_selected"] for row in rows], [2, 2, 2, 2, 1, 1])
            self.assertEqual(rows[0]["vs_labels"]["f1"], 0.5)
            self.assertEqual(rows[4]["vs_labels"], dict(tp=1, fp=0, fn=1, n_pred=1,
                                                      precision=1.0, recall=0.5, f1=0.6667))
            self.assertEqual(rows[3]["vs_join_full"]["f1"], 1.0)
            self.assertEqual(rows[4]["vs_join_full"]["recall"], 0.5)
            self.assertEqual(rows[5]["lm_calls_kind"], "replayed_logical_pair_evaluations")
            self.assertEqual(result["reference_scores_sha256"], pruning.digest(pruning.read_json(args.scores)))
            self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
            baseline.configure.assert_not_called()
            baseline.load_articles.assert_not_called()
            baseline.pairwise_join.assert_not_called()

    def test_report_missing_full_is_null_but_valid_empty_full_is_measured(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, args = self.report_fixture(root)
            summary_path = args.baseline_summary[0]
            summary = pruning.read_json(summary_path)
            full = summary["runs"].pop("join_full")
            pruning.save_json(summary_path, summary)
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.report(args)
                result = pruning.read_json(args.output)
                self.assertEqual(result["missing_experiments"], [1])
                self.assertTrue(all(row["vs_join_full"] is None for row in result["results"]))
                args.output = root / "empty-full-report.json"
                full["vs_labels"] = baseline.prf(set(), {0, 1})
                summary["runs"] = {"join_full": full}
                pruning.save_json(summary_path, summary)
                (summary_path.parent / "join_full_semi.csv").write_text("_qid,title\n", encoding="utf-8")
                pruning.report(args)
                result = pruning.read_json(args.output)
                self.assertEqual(result["results"][0]["n_selected"], 0)
                self.assertTrue(all(row["vs_join_full"]["f1"] == 0.0 for row in result["results"]))
                self.assertTrue(all(row["vs_join_full"] is not None for row in result["results"]))

    def test_report_rejects_stale_csv_and_incomplete_or_mixed_artifacts(self):
        for problem in ("count", "ids", "metrics", "progress", "gold", "replay", "binding"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                with contextlib.redirect_stdout(io.StringIO()):
                    root = Path(directory)
                    baseline, args = self.report_fixture(root)
                    summary_path = args.baseline_summary[0]
                    summary = pruning.read_json(summary_path)
                    if problem == "count":
                        summary["runs"]["join_full"]["vs_labels"].pop("n_pred")
                        pruning.save_json(summary_path, summary)
                    elif problem == "ids":
                        (summary_path.parent / "join_full_semi.csv").write_text("_qid,title\n99,outside\n", encoding="utf-8")
                    elif problem == "metrics":
                        (summary_path.parent / "join_full_semi.csv").write_text("_qid,title\n0,a\n1,b\n", encoding="utf-8")
                    elif problem == "progress":
                        (root / "run4/progress.jsonl").write_text("", encoding="utf-8")
                    elif problem == "gold":
                        scores = pruning.read_json(root / "e5.json")
                        scores["articles"][0]["gold_matches"] = []
                        pruning.save_json(root / "e5.json", scores)
                    elif problem == "replay":
                        args.replay = [root / "replay.json"]
                        with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                            pruning.replay(SimpleNamespace(scores=args.scores, source_dir=root / "run4", threshold=0.7,
                                                           calibration=None, output=args.replay[0]))
                        saved = pruning.read_json(args.replay[0])
                        saved["summary"]["lm_calls"] = 0
                        pruning.save_json(args.replay[0], saved)
                    else:
                        path = root / "run4/manifest.json"
                        manifest = pruning.read_json(path)
                        manifest["baseline_comparison"] = {"sources": [{"summary_sha256": "different"}]}
                        pruning.save_json(path, manifest)
                        saved = pruning.read_json(root / "run4/summary.json")
                        saved["manifest_sha256"] = pruning.digest(manifest)
                        pruning.save_json(root / "run4/summary.json", saved)
                    with patch.dict(sys.modules, {"semijoin_baseline": baseline}), self.assertRaises(ValueError):
                        pruning.report(args)
                    self.assertFalse(args.output.exists())
                    baseline.configure.assert_not_called()
                    baseline.load_articles.assert_not_called()

    def test_report_rejects_independent_biolord_content_differences(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, args = self.report_fixture(root)
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                for field in ("score", "scorer", "device", "n_chunks"):
                    with self.subTest(field=field):
                        scores = pruning.read_json(root / "bio5.json")
                        if field == "score":
                            scores["articles"][0]["scores"][0] = np.nextafter(0.2, 1.0)
                        elif field == "scorer":
                            scores["scorer"]["revision"] = "other"
                        elif field == "device":
                            scores["embedding_device"] = "cuda"
                        else:
                            scores["articles"][0]["n_chunks"] += 1
                        alternate, run_dir = root / f"{field}.json", root / f"run-{field}"
                        pruning.save_json(alternate, scores)
                        pruning.run(SimpleNamespace(scores=alternate, output_dir=run_dir, experiment=5,
                                                    threshold=0.7, calibration=None))
                        args.runs[-1] = (alternate, run_dir)
                        with self.assertRaisesRegex(ValueError, "identical BioLORD score files"):
                            pruning.report(args)

    def test_report_and_replay_require_each_live_runs_original_scores(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, args = self.report_fixture(root)
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                args.runs[-1] = (root / "bio4.json", root / "run5")
                with self.assertRaisesRegex(ValueError, "exactly the same scores"):
                    pruning.report(args)
                with self.assertRaisesRegex(ValueError, "exactly the same scores"):
                    pruning.replay(SimpleNamespace(scores=root / "bio5.json", source_dir=root / "run4",
                                                   threshold=0.7, calibration=None, output=root / "replay.json"))

    def test_report_uses_existing_prf_for_rounding_edge_cases(self):
        baseline = load_baseline()
        with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
            for positives, selected, true_positives in [(10, 246, 4), (32, 160, 9)]:
                pred = set(range(selected))
                gold = set(range(true_positives)) | set(range(selected, selected + positives - true_positives))
                metrics = pruning.comparison_prf(pred, gold)
                self.assertEqual({key: metrics[key] for key in baseline.prf(pred, gold)}, baseline.prf(pred, gold))

    def test_baseline_comparison_rejects_different_recorded_conditions(self):
        baseline = load_baseline()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            for key, value in [
                ("reaction_set", "other"), ("prompt", "other"), ("reactions", ["B", "A"]),
                ("n_articles", 25), ("n_reactions", 3), ("n_gold", 1), ("model", "other"),
                ("method", "join_cascade"),
            ]:
                with self.subTest(field=key):
                    summary = self.baseline_summary(baseline)
                    target = summary if key in ("reaction_set", "prompt", "reactions") else summary["runs"]["early_stop"]
                    target[key] = value
                    pruning.save_json(path, summary)
                    with self.assertRaisesRegex(ValueError, f"{key} differs"):
                        pruning.compare_baselines(self.scores, [path], baseline)

    def test_missing_baseline_fields_are_unknown_and_cascade_is_not_early(self):
        baseline = load_baseline()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            summary = self.baseline_summary(baseline)
            summary.pop("prompt")
            summary["runs"]["join_full"].pop("model")
            summary["runs"]["join_cascade"] = summary["runs"].pop("early_stop")
            pruning.save_json(path, summary)
            report = pruning.compare_baselines(self.scores, [path], baseline)
            self.assertEqual(report["status"], "reported_fields_match_only")
            self.assertEqual(report["missing_methods"], ["early_stop"])
            self.assertEqual(report["sources"][0]["missing_fields"], ["prompt", "runs.join_full.model"])
            self.assertEqual(report["sources"][0]["ignored_methods"], ["join_cascade"])
            self.assertIn("historical_article_ids_and_exact_input_text", report["unverified"])
            del summary["runs"]["join_full"]
            pruning.save_json(path, summary)
            with self.assertRaisesRegex(ValueError, "cascade is not experiment 2"):
                pruning.compare_baselines(self.scores, [path], baseline)
            for placeholder in [{}, {"method": "early_stop"}, {"n_articles": True}, {"n_articles": 2.0}]:
                with self.subTest(placeholder=placeholder):
                    summary["runs"] = {"early_stop": placeholder}
                    pruning.save_json(path, summary)
                    with self.assertRaisesRegex(ValueError, "positive integer n_articles"):
                        pruning.compare_baselines(self.scores, [path], baseline)

    def test_check_baselines_needs_no_model_and_is_stable_across_file_moves(self):
        baseline = load_baseline()
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        baseline.configure, baseline.load_articles = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            paths = [root / "full.json", root / "early.json"]
            for path, method in zip(paths, ("join_full", "early_stop")):
                summary = self.baseline_summary(baseline)
                summary["runs"] = {method: summary["runs"][method]}
                pruning.save_json(path, summary)
            pruning.save_json(root / "scores.json", self.scores)
            args = SimpleNamespace(scores=root / "scores.json", baseline_summary=paths, output=root / "check.json")
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.check_baselines(args)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    pruning.check_baselines(args)
            report = pruning.read_json(args.output)
            self.assertEqual(report["missing_methods"], [])
            moved = root / "moved.json"
            paths[0].replace(moved)
            self.assertEqual(report, pruning.compare_baselines(self.scores, [paths[1], moved], baseline))
            baseline.configure.assert_not_called()
            baseline.load_articles.assert_not_called()

    def test_run_checks_baselines_before_llm_and_binds_reference_contents(self):
        baseline = load_baseline()
        articles = pd.DataFrame([
            {"_qid": 0, "patient_description": "first", "reactions_list": ["B"]},
            {"_qid": 1, "patient_description": "second", "reactions_list": ["A"]},
        ])
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        baseline.load_articles = Mock(return_value=articles)
        baseline.configure = Mock(return_value=object())
        baseline.pairwise_join = Mock(return_value=SimpleNamespace(join_results=[]))
        self.scores["input_sha256"] = pruning.article_digest(articles)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pruning.save_json(root / "scores.json", self.scores)
            summary = self.baseline_summary(baseline)
            summary["runs"]["join_full"]["n_articles"] = 25
            pruning.save_json(root / "baseline.json", summary)
            args = SimpleNamespace(
                scores=root / "scores.json", experiment=4, threshold=None, calibration=None,
                baseline_summary=[root / "baseline.json"], output_dir=root / "run",
            )
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                with self.assertRaisesRegex(ValueError, "n_articles differs"):
                    pruning.run(args)
                baseline.configure.assert_not_called()
                baseline.pairwise_join.assert_not_called()
                self.assertFalse((args.output_dir / "manifest.json").exists())
                summary["runs"]["join_full"]["n_articles"] = 2
                pruning.save_json(root / "baseline.json", summary)
                pruning.run(args)
                manifest = pruning.read_json(args.output_dir / "manifest.json")
                self.assertEqual(manifest["baseline_comparison"]["missing_methods"], [])
                baseline.configure.reset_mock()
                summary["runs"]["join_full"]["time_s"] = 123
                pruning.save_json(root / "baseline.json", summary)
                with self.assertRaisesRegex(ValueError, "different configuration"):
                    pruning.run(args)
                baseline.configure.assert_not_called()

    def test_strict_threshold_and_invalid_scores_retain_article(self):
        self.assertTrue(pruning.keep_article([0.5, 0.1], 0.5))
        self.assertFalse(pruning.keep_article([0.49, 0.1], 0.5))
        for invalid in [None, float("nan"), float("inf")]:
            self.assertTrue(pruning.keep_article([invalid, 0.1], 0.5))
            self.assertEqual(pruning.ordering([invalid, 0.1]), [0, 1])
        self.assertEqual(pruning.ordering([0.2, 0.8, 0.8]), [1, 2, 0])

    def test_calibration_respects_integer_miss_budget(self):
        self.scores["articles"] = [{"_qid": index, "scores": [index / 20], "gold_matches": [0]} for index in range(10)]
        self.assertEqual(pruning.calibrate(self.scores, 0.95)["n_positive_pruned"], 0)
        self.scores["articles"] = [{"_qid": index, "scores": [index / 20], "gold_matches": [0]} for index in range(20)]
        self.assertEqual(pruning.calibrate(self.scores, 0.95)["n_positive_pruned"], 1)
        self.scores["articles"] = [{"_qid": 0, "scores": [0.1], "gold_matches": []}]
        with self.assertRaisesRegex(ValueError, "positive articles"):
            pruning.calibrate(self.scores, 1.0)

    def test_pruned_positives_remain_in_metric_denominator(self):
        records = [{"_qid": 0, "matched": "B", "lm_calls": 1}]
        output = pruning.apply_gate(self.scores, records, 0.5)
        summary = pruning.summarize(output)
        self.assertEqual((summary["n_articles"], summary["n_positive"]), (2, 2))
        self.assertEqual((summary["lm_calls"], summary["n_positive_pruned"]), (1, 1))
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(pruning.summarize(pruning.apply_gate(self.scores, [], 0.9))["lm_calls"], 0)

    def test_all_pruned_run_and_replay_require_no_progress_or_model(self):
        baseline = load_baseline()
        articles = pd.DataFrame(
            [
                {"_qid": 0, "patient_description": "first", "reactions_list": ["B"]},
                {"_qid": 1, "patient_description": "second", "reactions_list": ["A"]},
            ]
        )
        self.scores["input_sha256"] = pruning.article_digest(articles)
        baseline.load_articles = Mock(return_value=articles)
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        baseline.configure = Mock()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pruning.save_json(root / "scores.json", self.scores)
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.run(
                    SimpleNamespace(
                        scores=root / "scores.json", threshold=0.9, calibration=None, output_dir=root / "run"
                    )
                )
            baseline.configure.assert_not_called()
            self.assertFalse((root / "run" / "progress.jsonl").exists())
            pruning.replay(
                SimpleNamespace(
                    scores=root / "scores.json",
                    threshold=1.0,
                    calibration=None,
                    source_dir=root / "run",
                    output=root / "replay.json",
                )
            )
            self.assertEqual(pruning.read_json(root / "replay.json")["summary"]["lm_calls"], 0)

    def test_calibrated_threshold_records_validation_origin(self):
        self.scores["split"] = "validation"
        self.scores["input_sha256"] = "validation-example"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pruning.save_json(root / "scores.json", self.scores)
            pruning.calibrate_file(SimpleNamespace(scores=root / "scores.json", output=root / "calibration.json"))
            threshold, source = pruning.threshold_from_args(
                SimpleNamespace(calibration=root / "calibration.json", policy="conservative"), self.scores
            )
            self.assertEqual(threshold, 0.3)
            self.assertEqual(source["validation_input_sha256"], "validation-example")
            self.assertEqual(source["policy"], "conservative")
            with self.assertRaisesRegex(ValueError, "Output must differ"):
                pruning.calibrate_file(SimpleNamespace(scores=root / "scores.json", output=root / "scores.json"))

    def test_replay_rejects_missing_or_wrong_order_results(self):
        with self.assertRaisesRegex(ValueError, "no evaluated result"):
            pruning.apply_gate(self.scores, [], 0.5)
        with self.assertRaisesRegex(ValueError, "proxy order"):
            pruning.apply_gate(self.scores, [{"_qid": 0, "matched": "A", "lm_calls": 1}], 0.5)
        pruned = pruning.apply_gate(self.scores, [], 0.9)
        with self.assertRaisesRegex(ValueError, "no evaluated result"):
            pruning.apply_gate(self.scores, pruned, 0.5)

    def test_common_chunks_preserve_full_abstract_and_fit_both_tokenizers(self):
        tokenizers = {"e5": FakeTokenizer(), "biolord": FakeTokenizer(2)}
        text = "  Long abstract 한글🙂 " * 30 + "final evidence\n"
        chunks = pruning.abstract_chunks(text, tokenizers)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(pruning.fits_chunk(chunk, tokenizers) for chunk in chunks))
        self.assertIn("final evidence", chunks[-1])
        self.assertEqual(pruning.abstract_chunks(" \n", tokenizers), [])
        text = " " + "!," * 150
        chunks = pruning.abstract_chunks(text, tokenizers)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(chunk.strip() and pruning.fits_chunk(chunk, tokenizers) for chunk in chunks))
        original = pruning.tokenizer_digest(tokenizers["e5"])
        tokenizers["e5"].backend_tokenizer.to_str = lambda: json.dumps(
            {"step": 1, "padding": "batch", "truncation": 128}
        )
        self.assertEqual(pruning.tokenizer_digest(tokenizers["e5"]), original)
        self.assertNotEqual(pruning.tokenizer_digest(tokenizers["biolord"]), original)

    def test_chunk_scoring_prefixes_max_pool_and_invalid_evidence(self):
        baseline = load_baseline()
        baseline.load_articles = Mock(
            return_value=pd.DataFrame(
                [
                    {
                        "_qid": 0,
                        "patient_description": "LLM only",
                        "abstract": "head " * 80 + "TAIL",
                        "reactions_list": [],
                    },
                    {"_qid": 1, "patient_description": "second", "abstract": "bad", "reactions_list": ["A"]},
                    {"_qid": 2, "patient_description": "third", "abstract": "", "reactions_list": []},
                ]
            )
        )
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        for model in ["e5", "biolord"]:
            label_texts = [("query: " if model == "e5" else "") + label for label in ["A", "B"]]

            def embed(texts):
                if texts == label_texts:
                    return np.eye(2)
                if model == "e5":
                    self.assertTrue(all(text.startswith("passage: ") for text in texts))
                self.assertFalse(any("LLM only" in text for text in texts))
                return np.array(
                    [[0.0, 0.0] if "bad" in text else [0.0, 1.0] if "TAIL" in text else [1.0, 0.0] for text in texts]
                )

            rm = Mock(side_effect=embed)
            rm.transformer = Mock(max_seq_length=128, tokenizer=FakeTokenizer(1 if model == "e5" else 2))
            rm.transformer.__getitem__ = Mock(
                return_value=SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace()))
            )
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict(
                    sys.modules,
                    {
                        "semijoin_baseline": baseline,
                        "lotus.models": SimpleNamespace(SentenceTransformersRM=Mock(return_value=rm)),
                        "transformers": SimpleNamespace(
                            AutoTokenizer=SimpleNamespace(
                                from_pretrained=Mock(return_value=FakeTokenizer(2 if model == "e5" else 1))
                            )
                        ),
                    },
                ), contextlib.redirect_stdout(io.StringIO()) as score_log:
                    pruning.make_scores(
                        SimpleNamespace(
                            model=model, n_articles=None, split="validation", device="cpu", output=root / "scores.json"
                        )
                    )
                scores = pruning.load_scores(root / "scores.json")
                self.assertEqual(score_log.getvalue().strip(), "null_scores=4, affected_articles=2, qids=[1, 2]")
                self.assertEqual(rm.call_count, 2)
                self.assertEqual(rm.call_args_list[0].args[0], label_texts)
                self.assertEqual(scores["articles"][0]["scores"], [1.0, 1.0])
                self.assertGreater(scores["articles"][0]["n_chunks"], 1)
                for row in scores["articles"][1:]:
                    self.assertEqual(row["scores"], [None, None])
                    self.assertTrue(pruning.keep_article(row["scores"], 0.5))
                # The only gold positive has an invalid embedding: do not invent a cutoff.
                with self.assertRaisesRegex(ValueError, "finite scores"):
                    pruning.calibrate(scores, 1.0)

    def test_score_command_preserves_existing_artifact_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            original = b'{"existing": "score artifact"}'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "Score output already exists"):
                pruning.make_scores(SimpleNamespace(output=path))
            self.assertEqual(path.read_bytes(), original)

    def test_experiment_model_and_gate_selection_and_abstract_identity(self):
        no_gate = SimpleNamespace(experiment=4, threshold=None, calibration=None)
        self.assertEqual(pruning.threshold_from_args(no_gate, self.scores), (None, {"kind": "disabled"}))
        self.assertTrue(pruning.keep_article([0.1, 0.3], None))
        no_gate.threshold = 0.5
        with self.assertRaisesRegex(ValueError, "ordering only"):
            pruning.threshold_from_args(no_gate, self.scores)
        no_gate.experiment, no_gate.threshold = 3, None
        with self.assertRaisesRegex(ValueError, "Experiment 3 requires"):
            pruning.threshold_from_args(no_gate, self.scores)
        self.scores["scorer"]["model"] = pruning.MODELS["e5"]
        self.assertIsNone(pruning.threshold_from_args(no_gate, self.scores)[0])
        no_gate.experiment = 5
        with self.assertRaisesRegex(ValueError, "Experiment 5 requires"):
            pruning.threshold_from_args(no_gate, self.scores)
        articles = pd.DataFrame(
            [{"_qid": 0, "patient_description": "fixed LLM text", "reactions_list": [], "abstract": "old"}]
        )
        old_digest = pruning.article_digest(articles)
        articles.loc[0, "abstract"] = "new"
        self.assertNotEqual(pruning.article_digest(articles), old_digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.scores["scorer"]["representation"] = "article_prefix"
            pruning.save_json(root / "old.json", self.scores)
            with self.assertRaisesRegex(ValueError, "Regenerate scores"):
                pruning.load_scores(root / "old.json")

    def test_run_lock_and_interrupted_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with pruning.output_lock(root):
                with self.assertRaisesRegex(ValueError, "Another process"):
                    with pruning.output_lock(root):
                        self.fail("A second writer acquired the same run lock.")
            progress = root / "progress.jsonl"
            record = {"_qid": 0, "matched": "B", "lm_calls": 1}
            progress.write_text(json.dumps(record) + '\n{"_qid":', encoding="utf-8")
            with patch.object(Path, "write_bytes", side_effect=AssertionError("Completed records must stay in place.")):
                self.assertEqual(pruning.read_progress(progress), [record])
            self.assertTrue(progress.read_bytes().endswith(b"\n"))
            self.assertEqual(pruning.read_progress(progress), [record])

    def test_existing_executor_uses_supplied_order_and_keeps_original_ids(self):
        baseline = load_baseline()
        articles = pd.DataFrame([{"_qid": 7, "patient_description": "article"}])
        seen = []

        def evaluate(*args, **kwargs):
            seen.append((args[2][0], args[3][0]))
            return SimpleNamespace(join_results=[(7, 1, None)])

        baseline.pairwise_join = evaluate
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            joined, calls = baseline.run_early_stop(
                articles, ["A", "B"], object(), str(Path(directory) / "progress.jsonl"), {7: [1, 0]}
            )
        self.assertEqual(seen, [(7, 1)])
        self.assertEqual(calls, 1)
        self.assertEqual(joined.to_dict("records"), [{"_qid": 7, "reaction": "B"}])

    def test_loader_keeps_llm_prefix_and_exposes_complete_abstract_only_when_requested(self):
        baseline = load_baseline()
        raw = pd.DataFrame(
            {
                "title": ["paper"],
                "fulltext_processed": ["x" * 9000],
                "abstract": ["abstract tail"],
                "reactions": ["A, B"],
            }
        )
        baseline.load_dataset = Mock(return_value=SimpleNamespace(to_pandas=lambda: raw.copy()))
        self.assertNotIn("abstract", baseline.load_articles(1).columns)
        articles = baseline.load_articles(1, include_abstract=True)
        self.assertEqual(articles.loc[0, "abstract"], "abstract tail")
        self.assertEqual(articles.loc[0, "patient_description"], "x" * 8000)

    def test_ordering_runs_3_and_4_and_replay_5_share_the_existing_executor(self):
        baseline = load_baseline()
        articles = pd.DataFrame(
            [
                {"_qid": 0, "patient_description": "first", "abstract": "first abstract", "reactions_list": ["B"]},
                {"_qid": 1, "patient_description": "second", "abstract": "second abstract", "reactions_list": ["A"]},
            ]
        )
        self.scores["input_sha256"] = pruning.article_digest(articles)
        baseline.load_articles = Mock(return_value=articles)
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        baseline.configure = Mock(return_value=object())
        baseline.pairwise_join = lambda *args, **kwargs: SimpleNamespace(join_results=[(args[2][0], args[3][0], None)])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for experiment, model in [(3, pruning.MODELS["e5"]), (4, pruning.MODEL)]:
                self.scores["scorer"]["model"] = model
                path = root / f"scores{experiment}.json"
                pruning.save_json(path, self.scores)
                with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                    pruning.run(
                        SimpleNamespace(
                            scores=path,
                            experiment=experiment,
                            threshold=None,
                            calibration=None,
                            output_dir=root / f"run{experiment}",
                        )
                    )
                summary = pruning.read_json(root / f"run{experiment}" / "summary.json")
                self.assertEqual((summary["lm_calls"], summary["n_pruned"]), (2, 0))
                self.assertIsNone(summary["threshold"])
            # A prior attempt can have made partial pair calls without completing any article.
            retry_dir = root / "retry4"
            pruning.save_json(retry_dir / "manifest.json", pruning.read_json(root / "run4" / "manifest.json"))
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.run(
                    SimpleNamespace(
                        scores=root / "scores4.json",
                        experiment=4,
                        threshold=None,
                        calibration=None,
                        output_dir=retry_dir,
                    )
                )
            retry_summary = pruning.read_json(retry_dir / "summary.json")
            self.assertTrue(retry_summary["resumed"])
            self.assertIsNone(retry_summary["estimated_embedding_plus_run_time_s"])
            pruning.replay(
                SimpleNamespace(
                    scores=root / "scores4.json",
                    threshold=0.5,
                    calibration=None,
                    source_dir=root / "run4",
                    output=root / "replay5.json",
                )
            )
            replay = pruning.read_json(root / "replay5.json")
            self.assertEqual(replay["summary"]["experiment"], 5)
            self.assertEqual((replay["summary"]["lm_calls"], replay["summary"]["n_positive_pruned"]), (1, 1))
            baseline.configure.assert_called_with(with_retriever=False)
            manifest_path = root / "run4" / "manifest.json"
            mismatched = pruning.read_json(manifest_path)
            mismatched["model"] = "hosted_vllm/different-model"
            pruning.save_json(manifest_path, mismatched)
            with self.assertRaisesRegex(ValueError, "Source results disagree"):
                pruning.replay(
                    SimpleNamespace(
                        scores=root / "scores4.json",
                        threshold=0.5,
                        calibration=None,
                        source_dir=root / "run4",
                        output=root / "wrong-model-replay.json",
                    )
                )

    @unittest.skipUnless(importlib.util.find_spec("litellm"), "Real LOTUS dependencies are installed in CI.")
    def test_real_lotus_pair_formatting_parsing_and_early_stop(self):
        from lotus.models import LM
        from lotus.types import LMOutput

        baseline = load_baseline(real_lotus=True)
        articles = pd.DataFrame([{"_qid": 7, "patient_description": "Patient report"}])
        responses = [LMOutput(outputs=["False"]), LMOutput(outputs=["True"])]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(LM, "__call__", side_effect=responses) as model_call:
                lm = baseline.configure(with_retriever=False)
                joined, calls = baseline.run_early_stop(
                    articles, ["A", "B", "C"], lm, str(Path(directory) / "progress.jsonl"), {7: [2, 0, 1]}
                )
        self.assertEqual(calls, 2)
        self.assertEqual(lm.n_calls, 2)
        self.assertEqual(joined.to_dict("records"), [{"_qid": 7, "reaction": "A"}])
        prompts = [call.args[0][0][-1]["content"] for call in model_call.call_args_list]
        self.assertIn("Patient report", prompts[0])
        self.assertIn("«C»", prompts[0])
        self.assertIn("«A»", prompts[1])
        self.assertNotIn("reactions_list", prompts[0])

    def test_effective_model_configuration(self):
        baseline = load_baseline()
        with patch.dict("os.environ", {"LLM_BATCH": "7", "LLM_API_BASE": "http://test/v1"}):
            baseline.LLM_MODEL = "hosted_vllm/test"
            self.assertEqual(baseline.lm_config()["max_batch_size"], 7)
            self.assertEqual(baseline.lm_config()["api_base"], "http://test/v1")
            baseline.LLM_MODEL = "gpt-4o-mini"
            self.assertEqual(baseline.lm_config()["max_batch_size"], 64)
            self.assertNotIn("api_base", baseline.lm_config())

    def test_run_and_replay_without_model_or_dataset_access(self):
        baseline = load_baseline()
        articles = pd.DataFrame(
            [
                {"_qid": 0, "patient_description": "first", "reactions_list": ["B"]},
                {"_qid": 1, "patient_description": "second", "reactions_list": ["A"]},
            ]
        )
        self.scores["input_sha256"] = pruning.article_digest(articles)
        baseline.load_articles = Mock(return_value=articles)
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        baseline.configure = Mock(return_value=object())
        baseline.pairwise_join = Mock(return_value=SimpleNamespace(join_results=[(0, 1, None)]))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pruning.save_json(root / "scores.json", self.scores)
            args = SimpleNamespace(
                scores=root / "scores.json", threshold=0.5, calibration=None, output_dir=root / "run"
            )
            with patch.dict(sys.modules, {"semijoin_baseline": baseline}):
                pruning.run(args)
                baseline.pairwise_join.assert_called_once()
                baseline.configure.assert_called_once_with(with_retriever=False)
                measured = (args.output_dir / "summary.json").read_bytes()
                baseline.configure.reset_mock()
                with patch.dict(
                    "os.environ", {"DASHBOARD_BASE_URL": "http://unrelated", "OPENAI_BASE_URL": "http://unused"}
                ):
                    pruning.run(args)
                baseline.configure.assert_not_called()
                self.assertEqual((args.output_dir / "summary.json").read_bytes(), measured)
                with patch.dict("os.environ", {"LLM_SERVER_ID": "different-weights-or-server-settings"}):
                    with self.assertRaisesRegex(ValueError, "different configuration"):
                        pruning.run(args)
                baseline.configure.assert_not_called()
                altered_summary = pruning.read_json(args.output_dir / "summary.json")
                altered_summary["threshold"] = 0.6
                pruning.save_json(args.output_dir / "summary.json", altered_summary)
                with self.assertRaisesRegex(ValueError, "Saved result artifacts disagree"):
                    pruning.run(args)
                (args.output_dir / "summary.json").write_bytes(measured)
                progress_path = args.output_dir / "progress.jsonl"
                saved_progress = progress_path.read_bytes()
                progress_path.write_bytes(b"")
                with self.assertRaisesRegex(ValueError, "progress is incomplete"):
                    pruning.run(args)
                baseline.configure.assert_not_called()
                progress_path.write_bytes(saved_progress)
                args.threshold = 0.6
                with self.assertRaisesRegex(ValueError, "different configuration"):
                    pruning.run(args)
                # Invalid partial progress must fail before any unfinished article is evaluated.
                manifest = pruning.read_json(args.output_dir / "manifest.json")
                manifest["threshold"] = 0.2
                args.output_dir, args.threshold = root / "corrupt", 0.2
                pruning.save_json(args.output_dir / "manifest.json", manifest)
                (args.output_dir / "progress.jsonl").write_text(
                    json.dumps({"_qid": 0, "matched": "A", "lm_calls": 1}) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "proxy order"):
                    pruning.run(args)
                baseline.configure.assert_not_called()
                (args.output_dir / "progress.jsonl").write_text(
                    json.dumps({"_qid": 0, "matched": None, "lm_calls": 0, "pruned": True}) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "only evaluated, retained"):
                    pruning.run(args)
                baseline.configure.assert_not_called()
            pruning.replay(
                SimpleNamespace(
                    scores=args.scores,
                    threshold=0.9,
                    calibration=None,
                    source_dir=root / "run",
                    output=root / "replay.json",
                )
            )
            replay = json.loads((root / "replay.json").read_text(encoding="utf-8"))
            self.assertEqual(replay["summary"]["lm_calls"], 0)
            self.assertEqual(replay["summary"]["fn"], 2)
            self.assertIsNone(replay["summary"]["estimated_embedding_plus_run_time_s"])
            self.assertEqual(replay["summary"]["lm_calls_kind"], "replayed_logical_pair_evaluations")
            # Mixed or incomplete source artifacts cannot supply new evidence.
            saved_articles = (root / "run" / "articles.json").read_bytes()
            mixed = pruning.read_json(root / "run" / "articles.json")
            mixed[1].update(pruned=False, matched="B", lm_calls=1)
            pruning.save_json(root / "run" / "articles.json", mixed)
            replay_args = SimpleNamespace(
                scores=args.scores, threshold=0.0, calibration=None, source_dir=root / "run", output=root / "bad.json"
            )
            with self.assertRaisesRegex(ValueError, "Source results disagree"):
                pruning.replay(replay_args)
            (root / "run" / "articles.json").write_bytes(saved_articles)
            (root / "run" / "progress.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "completed source run"):
                pruning.replay(replay_args)
            replay_args.output = root / "run" / "articles.json"
            with self.assertRaisesRegex(ValueError, "Output must differ"):
                pruning.replay(replay_args)


class TestSemijoinExperimentRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        has_tools = all(shutil.which(cmd) for cmd in ("bash", "setsid")) and Path("/usr/bin/time").is_file()
        if os.name != "posix" or not has_tools:
            raise unittest.SkipTest("Runner requires Bash, setsid and GNU time on POSIX.")
        probe = subprocess.run(["/usr/bin/time", "-f", "%e", "-o", os.devnull, "true"], capture_output=True)
        if probe.returncode:
            raise unittest.SkipTest("Runner requires GNU time.")

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="semijoin study ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BENCHMARK)
        (self.root / "code_commit.txt").write_bytes(commit)
        (self.root / "settings.sh").write_text(
            'fresh() { for path do test ! -e "$path" || return 1; done; }\n'
            'preflight() { :; }\ncheck_embedding_context() { :; }\n', encoding="utf-8"
        )
        python = self.root / "python"
        python.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
if sys.argv[1] == "-":
    sys.stdin.read()
    sys.exit(0)
assert sys.argv[1] == "semijoin_pruning.py"
root = Path(os.environ["R"])
command, *args = sys.argv[2:]
if command == os.environ.get("SLEEP_STAGE"):
    (root / "sleeping_group").write_text(str(os.getpgrp()))
with (root / "calls.jsonl").open("a") as trace:
    trace.write(json.dumps([command, *[arg.removeprefix(str(root) + "/") for arg in args]]) + "\\n")
print(command, flush=True)
if command == os.environ.get("SLEEP_STAGE"):
    time.sleep(0.5)
if command == os.environ.get("FAIL_STAGE"):
    sys.exit(23)
if command == "run":
    output = Path(args[args.index("--output-dir") + 1])
    output.mkdir()
    for name in ("summary.json", "articles.json"):
        (output / name).write_text("{}")
else:
    Path(args[args.index("--output") + 1]).write_text("{}")
""", encoding="utf-8")
        python.chmod(0o755)
        self.env = dict(os.environ, R=str(self.root), BENCHMARK_DIR=str(BENCHMARK), REACTION_SET="liver",
                        PROMPT="paper", N_ARTICLES="250", EMBED_DEVICE="cpu", EMBED_GPU_UUID="mock",
                        PATH=str(self.root) + os.pathsep + os.environ["PATH"], FAIL_STAGE="", SEMIJOIN_TIMED_CHILD="")

    def launch(self, experiment, *args, **settings):
        return subprocess.run(["bash", BENCHMARK / "run_semijoin_experiment.sh", str(experiment), self.root, *args],
                              env={**self.env, **settings}, capture_output=True, text=True, timeout=20)

    def calls(self):
        return [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]

    def test_three_fresh_workflows_measure_their_own_scores_and_live_run(self):
        for experiment, name, stages in [
            (3, "experiment3", ["test_e5_score"]),
            (4, "experiment4", ["test_biolord_score"]),
            (5, "experiment5_conservative", ["validation_biolord", "calibration", "test_biolord_exp5_score"]),
        ]:
            result = self.launch(experiment)
            self.assertEqual(result.returncode, 0, result.stderr)
            workflow = self.root / f"{name}_workflow"
            self.assertEqual(pruning.read_json(workflow / "status.json"),
                             {"experiment": experiment, "status": "complete", "exit_code": 0})
            self.assertGreaterEqual(float((workflow / "total_wall_s.txt").read_text()), 0)
            for stage in [*stages, name]:
                self.assertGreaterEqual(float((self.root / f"{stage}_wall_s.txt").read_text()), 0)
                self.assertTrue((self.root / f"{stage}.log").read_text().strip())
        self.assertEqual(self.calls(), [command.split() for command in [
            "score --model e5 --split test --n-articles 250 --device cpu --output test_e5.json",
            "run test_e5.json --experiment 3 --output-dir experiment3",
            "score --model biolord --split test --n-articles 250 --device cpu --output test_biolord.json",
            "run test_biolord.json --experiment 4 --output-dir experiment4",
            "score --model biolord --split validation --device cpu --output validation_biolord.json",
            "calibrate validation_biolord.json --output thresholds.json",
            "score --model biolord --split test --n-articles 250 --device cpu --output test_biolord_exp5.json",
            "run test_biolord_exp5.json --experiment 5 --calibration thresholds.json --policy conservative "
            "--output-dir experiment5_conservative",
        ]])

    def test_failed_calibration_stops_following_stages_and_preserves_files(self):
        self.assertNotEqual(self.launch(3, N_ARTICLES="249").returncode, 0)
        self.assertFalse((self.root / "experiment3_workflow").exists())
        result = self.launch(5, FAIL_STAGE="calibrate")
        self.assertEqual(result.returncode, 23, result.stderr)
        workflow = self.root / "experiment5_conservative_workflow"
        self.assertEqual(pruning.read_json(workflow / "status.json"),
                         {"experiment": 5, "status": "failed", "exit_code": 23})
        self.assertEqual([call[0] for call in self.calls()], ["score", "calibrate"])
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for args in [(), ("--timed",)]:
            self.assertNotEqual(self.launch(5, *args).returncode, 0)
            self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_term_to_launcher_records_failed_workflow(self):
        trace = self.root / "calls.jsonl"
        with subprocess.Popen(["bash", BENCHMARK / "run_semijoin_experiment.sh", "5", self.root],
                              env={**self.env, "SLEEP_STAGE": "calibrate"}) as launcher:
            deadline = time.monotonic() + 10
            while not trace.exists() or '["calibrate",' not in trace.read_text():
                self.assertIsNone(launcher.poll(), "Launcher exited before calibration")
                self.assertLess(time.monotonic(), deadline, "Timed out waiting for calibration")
                time.sleep(0.01)
            launcher.terminate()
            self.assertEqual(launcher.wait(timeout=10), 143)
        self.assertEqual(pruning.read_json(self.root / "experiment5_conservative_workflow/status.json"),
                         {"experiment": 5, "status": "failed", "exit_code": 143})
        self.assertEqual([call[0] for call in self.calls()], ["score", "calibrate"])
        group = int((self.root / "sleeping_group").read_text())
        deadline = time.monotonic() + 3
        while True:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            self.assertLess(time.monotonic(), deadline, "Timed child process group survived TERM")
            time.sleep(0.02)


if __name__ == "__main__":
    unittest.main()
