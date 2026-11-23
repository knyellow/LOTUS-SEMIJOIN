"""Offline regressions for pruning boundaries, ordering, replay, and resume identity."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
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


class TestSemijoinPruning(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "split": "test",
            "scorer": {"model": pruning.MODEL},
            "reactions": ["A", "B"],
            "articles": [
                {"_qid": 0, "scores": [0.2, 0.8], "gold_matches": [1]},
                {"_qid": 1, "scores": [0.1, 0.3], "gold_matches": [0]},
            ],
            "embedding_time_s": 1.0,
        }

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
        with self.assertRaisesRegex(ValueError, "BioLORD order"):
            pruning.apply_gate(self.scores, [{"_qid": 0, "matched": "A", "lm_calls": 1}], 0.5)
        pruned = pruning.apply_gate(self.scores, [], 0.9)
        with self.assertRaisesRegex(ValueError, "no evaluated result"):
            pruning.apply_gate(self.scores, pruned, 0.5)

    def test_score_and_calibration_commands_keep_zero_vectors(self):
        baseline = load_baseline()
        baseline.load_articles = Mock(
            return_value=pd.DataFrame(
                [
                    {"_qid": 0, "patient_description": "first", "reactions_list": ["A"]},
                    {"_qid": 1, "patient_description": "second", "reactions_list": []},
                ]
            )
        )
        baseline.REACTION_SET, baseline.REACTION_SETS = "test", {"test": ["A", "B"]}
        rm = Mock(side_effect=[np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])])
        rm.transformer = Mock(max_seq_length=128)
        rm.transformer.__getitem__ = Mock(
            return_value=SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace()))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                sys.modules,
                {
                    "semijoin_baseline": baseline,
                    "lotus.models": SimpleNamespace(SentenceTransformersRM=Mock(return_value=rm)),
                },
            ):
                pruning.make_scores(
                    SimpleNamespace(n_articles=None, split="validation", device="cpu", output=root / "scores.json")
                )
            scores = pruning.load_scores(root / "scores.json")
            self.assertEqual(rm.call_count, 2)
            self.assertEqual(scores["articles"][0]["scores"], [None, None])
            self.assertTrue(pruning.keep_article(scores["articles"][1]["scores"], 0.5))
            # All positive scores were invalid, so a cutoff cannot be calibrated.
            with self.assertRaisesRegex(ValueError, "finite scores"):
                pruning.calibrate_file(SimpleNamespace(scores=root / "scores.json", output=root / "calibration.json"))

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
                with self.assertRaisesRegex(ValueError, "BioLORD order"):
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


if __name__ == "__main__":
    unittest.main()
