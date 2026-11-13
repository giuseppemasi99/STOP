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

SCRIPT = Path(__file__).resolve().parents[1] / "src/yolo/save_json.py"
spec = importlib.util.spec_from_file_location("yolo_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def prediction(image, correct=True):
    return {"image": image, "ground_truth_label": "Bear", "width": 32,
            "height": 16, "input_bytes": 123, "gflops": 0.25,
            "predicted_label": "Bear" if correct else "Dog", "correct": correct,
            "confidence": 0.9, "inference_time_ms": 1.2}


class ExportTests(unittest.TestCase):
    def test_merge_joins_paths_not_positions(self):
        records = {size: [prediction("b.jpg"), prediction("a.jpg")]
                   for size in export.MODEL_NAMES}
        records["m"].reverse()
        records["l"][0] = prediction("b.jpg", False)
        merged = export.merge_predictions(records)
        self.assertEqual([r["image"] for r in merged], ["a.jpg", "b.jpg"])
        self.assertEqual(set(merged[0]["models"]), {"nano", "medium", "large"})
        self.assertFalse(merged[1]["models"]["large"]["correct"])
        self.assertEqual(merged[0]["input_bytes"], 123)

    def test_merge_rejects_missing_duplicate_and_inconsistent_records(self):
        valid = {size: [prediction("a.jpg")] for size in export.MODEL_NAMES}
        cases = []
        missing_size = deepcopy(valid)
        del missing_size["l"]
        cases.append(missing_size)
        duplicate = deepcopy(valid)
        duplicate["n"].append(prediction("a.jpg"))
        cases.append(duplicate)
        missing_image = deepcopy(valid)
        missing_image["m"].append(prediction("b.jpg"))
        cases.append(missing_image)
        metadata = deepcopy(valid)
        metadata["m"][0]["width"] = 64
        cases.append(metadata)
        for cost in [0, -1, math.nan, math.inf, None, True]:
            invalid_cost = deepcopy(valid)
            invalid_cost["l"][0]["gflops"] = cost
            cases.append(invalid_cost)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                export.merge_predictions(case)

    def test_flops_units_actual_shape_and_failure(self):
        utility = SimpleNamespace(get_flops=lambda model, imgsz: 2.5 if imgsz == [224, 224] else 0)
        with patch.dict(sys.modules, {"ultralytics.utils.torch_utils": utility}):
            self.assertEqual(export.profile_gflops(object(), (1, 3, 224, 224)), 2.5)
            with self.assertRaises(RuntimeError):
                export.profile_gflops(object(), (1, 3, 128, 128))
            with self.assertRaises(ValueError):
                export.profile_gflops(object(), (2, 3, 224, 224))

    def test_collect_image_metadata(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Bear" / "example.jpg"
            path.parent.mkdir()
            Image.new("RGB", (32, 16)).save(path)
            metadata = export.collect_images(directory)[0]
            self.assertEqual((metadata["width"], metadata["height"]), (32, 16))
            self.assertEqual(metadata["input_bytes"], path.stat().st_size)
            self.assertEqual(metadata["ground_truth_label"], "Bear")

    def test_benchmark_observes_tensor_shape_and_profiles_once(self):
        class Backend:
            hook = None
            def register_forward_pre_hook(self, hook):
                self.hook = hook
                return SimpleNamespace(remove=lambda: setattr(self, "hook", None))

        backend = Backend()
        class Model:
            task = "classify"
            ckpt_path = "best.pt"
            model = object()
            predictor = SimpleNamespace(model=backend, device=SimpleNamespace(type="cpu"))
            def predict(self, **kwargs):
                if backend.hook:
                    backend.hook(backend, [SimpleNamespace(shape=(1, 3, 224, 224))])
                return [SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=0.9),
                                        names={0: "Bear", 1: "Dog"})]

        images = [prediction("a.jpg"), prediction("b.jpg")]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                 patch.object(export, "profile_gflops", return_value=0.5) as profile:
                records, stats = export.benchmark_inference(Model(), images, "n", directory, 256)
            profile.assert_called_once_with(Model.model, (1, 3, 224, 224))
            self.assertEqual(stats["num_correct"], 2)
            self.assertEqual(records[0]["input_shape"], [1, 3, 224, 224])
            self.assertEqual(records[0]["gflops"], 0.5)
            self.assertIsNone(backend.hook)
            saved = json.loads((Path(directory) / "predictions_n.json").read_text())
            self.assertEqual(len(saved), 2)

    def test_merge_only_needs_no_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in export.MODEL_NAMES:
                export.write_json(Path(directory) / f"predictions_{size}.json", [prediction("a.jpg")])
            args = export.build_parser().parse_args(["--merge-only", "--output-dir", directory])
            export.launch(args)
            merged = json.loads((Path(directory) / "demands.json").read_text())
            self.assertEqual(merged[0]["models"]["nano"]["gflops"], 0.25)


if __name__ == "__main__":
    unittest.main()
