"""Train/profile YOLO classifiers and export STOP-compatible empirical demands."""

import argparse
import csv
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_NAMES = {"n": "nano", "m": "medium", "l": "large"}
IMAGE_FIELDS = ("image", "ground_truth_label", "width", "height", "input_bytes")


def write_json(path, records):
    """Replace only after a complete, finite JSON document has been written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(records, output, indent=2, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def collect_images(val_dir):
    from PIL import Image

    images = []
    for path in sorted(Path(val_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(path) as image:
            width, height = image.size
        input_bytes = path.stat().st_size
        if min(width, height, input_bytes) <= 0:
            raise ValueError(f"Invalid image dimensions or byte count: {path}")
        images.append({"image": str(path), "ground_truth_label": path.parent.name,
                       "width": width, "height": height, "input_bytes": input_bytes})
    if not images:
        raise ValueError(f"No validation images found in {val_dir}")
    return images


def profile_gflops(network, shape):
    from ultralytics.utils.torch_utils import get_flops

    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"Expected one RGB input in BCHW format; received {shape}")
    # Profile a copy outside the timed region so hooks/buffers cannot affect inference.
    value = float(get_flops(deepcopy(network), imgsz=list(shape[-2:])))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            "GFLOPs profiling failed. Install ultralytics-thop and verify that the "
            "checkpoint supports profiling; zero costs cannot be exported."
        )
    return value  # Already GFLOPs, not FLOPs: do not divide by 1e9 again.


def benchmark_inference(model, images, size, output_dir, imgsz, device=None):
    import torch

    if model.task != "classify":
        raise ValueError("Use YOLO classification checkpoints (*-cls.pt)")
    options = {"imgsz": imgsz, "verbose": False, "augment": False}
    if device is not None:
        options["device"] = device
    model.predict(source=images[0]["image"], **options)  # Initialize backend and warm up.
    backend = model.predictor.model
    actual_device = model.predictor.device

    def synchronize():
        if actual_device.type == "cuda":
            torch.cuda.synchronize(actual_device)
        elif actual_device.type == "mps":
            torch.mps.synchronize()

    observed_shapes = []

    def capture_shape(_module, inputs):
        observed_shapes.append(tuple(inputs[0].shape))

    handle = backend.register_forward_pre_hook(capture_shape)
    cache = {}
    records = []
    try:
        for metadata in images:
            observed_shapes.clear()
            synchronize()
            start = time.perf_counter()
            result = model.predict(source=metadata["image"], **options)[0]
            synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            if result.probs is None:
                raise ValueError(f"No classification probabilities for {metadata['image']}")
            if len(observed_shapes) != 1:
                raise RuntimeError("Expected exactly one forward pass per image")
            shape = observed_shapes[0]
            if shape not in cache:
                cache[shape] = profile_gflops(model.model, shape)
            label = result.names[int(result.probs.top1)]
            if metadata["ground_truth_label"] not in set(result.names.values()):
                raise ValueError(f"Class folder {metadata['ground_truth_label']!r} is absent from model labels")
            records.append({**metadata,
                            "predicted_label": label,
                            "confidence": float(result.probs.top1conf),
                            "correct": label == metadata["ground_truth_label"],
                            "inference_time_ms": elapsed_ms,
                            "gflops": cache[shape],
                            "input_shape": list(shape),
                            "checkpoint": str(model.ckpt_path),
                            "device": str(actual_device)})
    finally:
        handle.remove()
    write_json(Path(output_dir) / f"predictions_{size}.json", records)
    timings = [record["inference_time_ms"] for record in records]
    correct = sum(record["correct"] for record in records)
    return records, {"num_images": len(records), "num_correct": correct,
                     "accuracy_from_predictions": correct / len(records),
                     "mean_inference_time_ms": sum(timings) / len(timings),
                     "min_inference_time_ms": min(timings),
                     "max_inference_time_ms": max(timings)}


def merge_predictions(predictions):
    """Join by image path, rejecting duplicate or missing model/image results."""
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError("A complete STOP dataset requires n, m, and l predictions")
    indexes = {}
    for size, records in predictions.items():
        index = {}
        for record in records:
            key = record["image"]
            if key in index:
                raise ValueError(f"Duplicate image in {size}: {key}")
            cost = record.get("gflops")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
                raise ValueError(f"Missing/invalid GFLOPs for {size}: {key}")
            if not isinstance(record.get("correct"), bool):
                raise ValueError(f"Missing boolean correctness for {size}: {key}")
            index[key] = record
        if not index:
            raise ValueError(f"Empty prediction list for {size}")
        indexes[size] = index
    keys = set(indexes["n"])
    if any(set(index) != keys for index in indexes.values()):
        raise ValueError("Model prediction files must contain exactly the same image paths")
    merged = []
    for key in sorted(keys):
        base = {field: indexes["n"][key][field] for field in IMAGE_FIELDS}
        models = {}
        for size, name in MODEL_NAMES.items():
            record = indexes[size][key]
            if any(record[field] != base[field] for field in IMAGE_FIELDS):
                raise ValueError(f"Image metadata differs between model runs: {key}")
            models[name] = {field: value for field, value in record.items() if field not in IMAGE_FIELDS}
        merged.append({**base, "models": models})
    return merged


def save_metrics_csv(metrics, stats, size, checkpoint, epochs, training_seconds, output_dir):
    row = {"model_size": size, "checkpoint": checkpoint, "epochs": epochs,
           "training_time_seconds": training_seconds, **stats}
    for name in ("top1", "top5"):
        if hasattr(metrics, name):
            row[f"{name}_accuracy"] = float(getattr(metrics, name))
    for name, value in getattr(metrics, "speed", {}).items():
        row[f"ultralytics_{name}_ms"] = value
    with (output_dir / f"metrics_{size}.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def launch(args):
    os.environ["WANDB_MODE"] = "disabled"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        predictions = {size: json.loads((output_dir / f"predictions_{size}.json").read_text())
                       for size in MODEL_NAMES}
    else:
        from ultralytics import YOLO

        images = collect_images(Path(args.data) / "val")
        predictions = {}
        sizes = [args.size] if args.size else list(MODEL_NAMES)
        checkpoints = {}
        for item in args.weights:
            size, separator, path = item.partition("=")
            if not separator or size not in sizes or not path or size in checkpoints:
                raise ValueError("--weights must contain unique selected sizes, e.g. n=path/to/best.pt")
            checkpoints[size] = path
        for size in sizes:
            checkpoint = checkpoints.get(size, f"storage/yolo/models/yolo11{size}-cls.pt")
            model = YOLO(checkpoint)
            if model.task != "classify":
                raise ValueError(f"Checkpoint is not a classifier: {checkpoint}")
            train_seconds = 0.0
            if not args.skip_train:
                start = time.perf_counter()
                options = {"seed": args.seed, "data": args.data, "name": f"{size}_all",
                           "project": args.project, "epochs": args.epochs, "imgsz": args.imgsz}
                if args.device is not None:
                    options["device"] = args.device
                model.train(**options)
                train_seconds = time.perf_counter() - start
                checkpoint = str(model.trainer.best)
                model = YOLO(checkpoint)
            options = {"data": args.data, "name": f"val_{size}", "imgsz": args.imgsz}
            if args.device is not None:
                options["device"] = args.device
            metrics = model.val(**options)
            records, stats = benchmark_inference(model, images, size, output_dir, args.imgsz, args.device)
            predictions[size] = records
            save_metrics_csv(metrics, stats, size, checkpoint,
                             0 if args.skip_train else args.epochs, train_seconds, output_dir)
        if args.size:
            print(f"Saved predictions_{args.size}.json; run all n/m/l sizes, then --merge-only.")
            return
    merged = merge_predictions(predictions)
    write_json(output_dir / "demands.json", merged)
    print(f"Saved {len(merged)} demands to {output_dir / 'demands.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=list(MODEL_NAMES), help="Run one size; default: run n, m, l and merge")
    parser.add_argument("--data", default="data/animals/complete")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs_yolo")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--imgsz", type=int, default=224, help="Explicit square training/validation/inference size")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--skip-train", action="store_true", help="Export existing checkpoints without retraining")
    parser.add_argument("--weights", nargs="*", default=[], metavar="SIZE=PATH", help="Checkpoint overrides for n/m/l")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing predictions_n/m/l.json files")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.imgsz <= 0 or args.epochs <= 0:
        parser.error("--imgsz and --epochs must be positive")
    if args.merge_only and (args.size or args.weights):
        parser.error("--merge-only cannot be combined with --size or --weights")
    launch(args)
