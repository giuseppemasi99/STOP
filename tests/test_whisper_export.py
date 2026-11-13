import importlib.util
import json
import math
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "src/whisper/save_json.py"
spec = importlib.util.spec_from_file_location("whisper_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def record(size, index=0):
    return {
        "demand_id": f"openslr/librispeech_asr/clean/validation/{index}",
        "dataset_index": index, "audio": f"{index}.flac", "audio_id": str(index),
        "ground_truth_transcript": "HELLO WORLD", "predicted_transcript": "hello world",
        "normalized_reference": "hello world", "normalized_prediction": "hello world",
        "duration_seconds": 12.25, "input_bytes": 3456, "audio_sha256": "abc",
        "sample_rate": 16000, "num_samples": 196000, "gflops": 10.0,
        "performance": 100.0, "wer": 0.0, "correct": True, "performance_metric": "word-accuracy",
        "inference_time_ms": 20.0, "long_form": False,
        "model": export.get_model_id(size),
        "benchmark_config": {"dataset_config": "clean", "split": "validation",
                             "multilingual": False, "performance_metric": "word-accuracy"},
    }


class WhisperExportTests(unittest.TestCase):
    def test_merge_by_stable_id_and_four_actual_sizes(self):
        records = {size: [record(size, 1), record(size, 0)] for size in export.MODEL_SIZES}
        records["medium"].reverse()
        merged = export.merge_predictions(records)
        self.assertEqual([r["dataset_index"] for r in merged], [0, 1])
        self.assertEqual(set(merged[0]["models"]), {"tiny", "small", "medium", "large"})
        self.assertEqual(merged[0]["duration_seconds"], 12.25)
        self.assertEqual(merged[0]["models"]["small"]["model"], "openai/whisper-small.en")

    def test_reject_incomplete_duplicate_or_inconsistent_exports(self):
        valid = {size: [record(size)] for size in export.MODEL_SIZES}
        cases = []
        missing = deepcopy(valid)
        del missing["large"]
        cases.append(missing)
        duplicate = deepcopy(valid)
        duplicate["tiny"].append(record("tiny"))
        cases.append(duplicate)
        incomplete = deepcopy(valid)
        incomplete["medium"].append(record("medium", 1))
        cases.append(incomplete)
        wrong_model = deepcopy(valid)
        wrong_model["small"][0]["model"] = "openai/whisper-base.en"
        cases.append(wrong_model)
        wrong_audio = deepcopy(valid)
        wrong_audio["large"][0]["input_bytes"] = 4000
        cases.append(wrong_audio)
        wrong_metric = deepcopy(valid)
        wrong_metric["small"][0]["benchmark_config"]["performance_metric"] = "exact-match"
        cases.append(wrong_metric)
        for value in [0, -1, math.nan, math.inf, None, True]:
            bad = deepcopy(valid)
            bad["large"][0]["gflops"] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                export.merge_predictions(bad)

    def test_performance_metrics_and_wer_above_one(self):
        partial = export.outcome_metrics("hello world", "hello", "word-accuracy", lambda r, p: 0.5)
        self.assertEqual(partial["performance"], 50)
        self.assertFalse(partial["correct"])
        insertions = export.outcome_metrics("hello", "many extra words", "word-accuracy", lambda r, p: 2)
        self.assertEqual(insertions["wer"], 2)
        self.assertEqual(insertions["performance"], 0)
        exact = export.outcome_metrics("hello world", "hello", "exact-match", lambda r, p: 0.5)
        self.assertEqual(exact["performance"], 0)
        self.assertEqual(export.outcome_metrics("", "", "word-accuracy", None)["performance"], 100)

    def test_long_form_boundary_and_greedy_options(self):
        calls = []
        def asr(data, **kwargs):
            calls.append(kwargs)
            return {"text": "hello"}
        _, short = export.transcribe_audio(asr, {}, 30, {"do_sample": False})
        _, long = export.transcribe_audio(asr, {}, 30.01, {"do_sample": False})
        self.assertFalse(short)
        self.assertTrue(long)
        self.assertFalse(calls[0]["return_timestamps"])
        self.assertTrue(calls[1]["return_timestamps"])
        self.assertFalse(calls[1]["generate_kwargs"]["do_sample"])

    def test_legacy_pipeline_receives_torch_dtype(self):
        calls = []
        def legacy_pipeline(task, model, device, torch_dtype=None, model_kwargs=None):
            calls.append((task, model, device, torch_dtype, model_kwargs))
            return "legacy pipeline"
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(pipeline=legacy_pipeline)}):
            self.assertEqual(export.create_asr_pipeline("whisper", 0, "float16"), "legacy pipeline")
        self.assertEqual(calls, [("automatic-speech-recognition", "whisper", 0, "float16",
                                 {"attn_implementation": "eager"})])

    def test_modern_pipeline_receives_dtype(self):
        calls = []
        def modern_pipeline(task, model, device, dtype=None, model_kwargs=None):
            calls.append((task, model, device, dtype, model_kwargs))
            return "modern pipeline"
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(pipeline=modern_pipeline)}):
            self.assertEqual(export.create_asr_pipeline("whisper", -1, "float32"), "modern pipeline")
        self.assertEqual(calls, [("automatic-speech-recognition", "whisper", -1, "float32",
                                 {"attn_implementation": "eager"})])

    def test_transcription_preserves_input_for_second_pass(self):
        waveform = object()
        original = {"array": waveform, "sampling_rate": 16000}
        def consuming_pipeline(data, **kwargs):
            self.assertIs(data.pop("array"), waveform)
            self.assertEqual(data.pop("sampling_rate"), 16000)
            return {"text": "hello"}
        export.transcribe_audio(consuming_pipeline, original, 12)
        export.transcribe_audio(consuming_pipeline, original, 12)
        self.assertEqual(original, {"array": waveform, "sampling_rate": 16000})

    def test_actual_torch_counter_full_query_and_units(self):
        import torch
        calls = []
        matrix = torch.ones(2, 3)
        weights = torch.ones(3, 4)
        def asr(data, **kwargs):
            # An encoder-like matmul and two decoder-like matmuls per query.
            calls.append(kwargs)
            for _ in range(3):
                torch.mm(matrix, weights)
            return {"text": "hello world"}
        prediction, long, elapsed, cost = export.benchmark_query(asr, {}, 12, {}, "cpu")
        self.assertEqual(prediction["text"], "hello world")
        self.assertFalse(long)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(cost, 3 * 2 * 2 * 3 * 4 / 1e9)

    def test_profile_rejects_different_outcome_or_zero_cost(self):
        import torch
        for texts in [["first", "second"], ["same", "same"]]:
            iterator = iter(texts)
            def asr(data, **kwargs):
                return {"text": next(iterator)}
            with self.subTest(texts=texts), self.assertRaises(RuntimeError):
                export.benchmark_query(asr, {}, 12, {}, "cpu")

    def test_resume_requires_matching_prefix(self):
        records = [record("tiny", i) for i in range(2)]
        dataset = [{"id": str(i), "text": "HELLO WORLD"} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_tiny.json"
            export.atomic_write_json(path, records)
            loaded = export.load_existing_results(path, records[0]["benchmark_config"], records[0]["model"], dataset, 2)
            self.assertEqual(len(loaded), 2)
            with self.assertRaises(ValueError):
                export.load_existing_results(path, records[0]["benchmark_config"], records[0]["model"], dataset, 1)
            records.reverse()
            export.atomic_write_json(path, records)
            with self.assertRaises(ValueError):
                export.load_existing_results(path, records[0]["benchmark_config"], records[0]["model"], dataset, 2)

    def test_encoded_bytes_and_path_sizes_and_mono_conversion(self):
        import numpy as np
        def read(source, **kwargs):
            self.assertEqual(source.read(), b"encoded audio")
            return np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32), 2
        with patch.dict(sys.modules, {"soundfile": SimpleNamespace(read=read)}):
            waveform, rate, metadata = export.extract_audio({"audio": {"bytes": b"encoded audio"}})
            self.assertEqual(list(waveform), [2.0, 3.0])
            self.assertEqual(rate, 2)
            self.assertEqual(metadata["duration_seconds"], 1)
            self.assertEqual(metadata["input_bytes"], len(b"encoded audio"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "audio.flac"
                path.write_bytes(b"encoded audio")
                _, _, from_path = export.extract_audio({"audio": {"path": str(path)}})
                self.assertEqual(metadata, from_path)

    def test_merge_only_without_transformers_or_dataset_load(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in export.MODEL_SIZES:
                export.atomic_write_json(Path(directory) / f"predictions_{size}.json", [record(size)])
            args = export.build_parser().parse_args(["--merge-only", "--output-dir", directory])
            export.launch(args)
            merged = json.loads((Path(directory) / "demands.json").read_text())
            self.assertEqual(merged[0]["models"]["tiny"]["performance"], 100)


if __name__ == "__main__":
    unittest.main()
