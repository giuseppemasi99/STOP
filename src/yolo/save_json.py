import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def benchmark_inference(
    model: YOLO,
    val_dir: Path,
    size: str,
    output_dir: Path,
) -> dict:

    image_paths = sorted(
        p for p in val_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {val_dir}")

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------
    model.predict(
        source=str(image_paths[0]),
        verbose=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    results = []
    inference_times = []

    # ---------------------------------------------------------
    # Image-by-image inference
    # ---------------------------------------------------------
    for image_path in image_paths:

        ground_truth_label = image_path.parent.name

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = model.predict(
            source=str(image_path),
            verbose=False,
        )[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)

        predicted_class_id = int(result.probs.top1)
        predicted_label = result.names[predicted_class_id]
        confidence = float(result.probs.top1conf)

        correct = predicted_label == ground_truth_label

        results.append({
            "image": str(image_path),
            "ground_truth_label": ground_truth_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": correct,
            "inference_time_ms": inference_time_ms,
        })

        print(
            f"{image_path.name} | "
            f"GT: {ground_truth_label} | "
            f"Pred: {predicted_label} | "
            f"Confidence: {confidence:.4f} | "
            f"Time: {inference_time_ms:.2f} ms"
        )

    # ---------------------------------------------------------
    # Save per-image predictions
    # ---------------------------------------------------------
    json_path = output_dir / f"predictions_{size}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nPredictions saved to: {json_path}")

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------
    num_correct = sum(result["correct"] for result in results)

    return {
        "num_images": len(results),
        "num_correct": num_correct,
        "accuracy_from_predictions": num_correct / len(results),
        "mean_inference_time_ms": sum(inference_times) / len(inference_times),
        "min_inference_time_ms": min(inference_times),
        "max_inference_time_ms": max(inference_times),
    }


def save_metrics_csv(
    metrics,
    benchmark_stats: dict,
    size: str,
    pretrained_model: str,
    best_model: str,
    epochs: int,
    training_time_seconds: float,
    output_dir: Path,
) -> None:

    row = {
        "model_size": size,
        "pretrained_model": pretrained_model,
        "best_model": best_model,
        "epochs": epochs,
        "training_time_seconds": training_time_seconds,
    }

    # ---------------------------------------------------------
    # Ultralytics classification metrics
    # ---------------------------------------------------------
    if hasattr(metrics, "top1"):
        row["top1_accuracy"] = float(metrics.top1)

    if hasattr(metrics, "top5"):
        row["top5_accuracy"] = float(metrics.top5)

    # ---------------------------------------------------------
    # Ultralytics internal timing
    # ---------------------------------------------------------
    if hasattr(metrics, "speed"):
        for name, value in metrics.speed.items():
            row[f"ultralytics_{name}_ms"] = float(value)

    # ---------------------------------------------------------
    # Our per-image benchmark
    # ---------------------------------------------------------
    row.update(benchmark_stats)

    csv_path = output_dir / f"metrics_{size}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        writer.writeheader()
        writer.writerow(row)

    print(f"Metrics saved to: {csv_path}")


def launch(args) -> None:

    os.environ["WANDB_MODE"] = "disabled"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    pretrained_model = (
        f"storage/yolo/models/yolo11{args.size}-cls.pt"
    )

    print("=" * 60)
    print(f"MODEL SIZE: {args.size}")
    print(f"PRETRAINED MODEL: {pretrained_model}")
    print("=" * 60)

    model = YOLO(pretrained_model)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print("\nStarting training...\n")

    training_start = time.perf_counter()

    model.train(
        seed=args.seed,
        data=args.data,
        name=f"{args.size}_all",
        project=args.project,
        epochs=args.epochs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    training_end = time.perf_counter()

    training_time_seconds = training_end - training_start

    print(
        f"\nTraining completed in "
        f"{training_time_seconds:.2f} seconds"
    )

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------
    best_model_path = str(model.trainer.best)

    print(f"\nLoading best model: {best_model_path}")

    best_model = YOLO(best_model_path)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    print("\nRunning validation...\n")

    metrics = best_model.val(
        data=args.data,
        name=f"val_{args.size}",
    )

    print(metrics)

    # ---------------------------------------------------------
    # Per-image inference benchmark
    # ---------------------------------------------------------
    print("\nRunning per-image inference benchmark...\n")

    val_dir = Path(args.data) / "val"

    benchmark_stats = benchmark_inference(
        model=best_model,
        val_dir=val_dir,
        size=args.size,
        output_dir=output_dir,
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------
    save_metrics_csv(
        metrics=metrics,
        benchmark_stats=benchmark_stats,
        size=args.size,
        pretrained_model=pretrained_model,
        best_model=best_model_path,
        epochs=args.epochs,
        training_time_seconds=training_time_seconds,
        output_dir=output_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["n", "s", "m", "l"],
        help="YOLO11 model size: n, s, m, or l",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/animals/complete",
        help="Dataset directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs_yolo",
        help="Ultralytics training output directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for JSON and CSV benchmark results",
    )

    args = parser.parse_args()

    launch(args)