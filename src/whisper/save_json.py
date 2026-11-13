"""Benchmark Whisper on LibriSpeech and export STOP-compatible demands."""

import argparse
import hashlib
import io
from importlib.metadata import version
from inspect import signature
import json
import math
import statistics
import time
from pathlib import Path

MODEL_SIZES = ("tiny", "small", "medium", "large")
ENGLISH_MODEL_IDS = {size: f"openai/whisper-{size}.en" for size in ("tiny", "base", "small", "medium")}
ENGLISH_MODEL_IDS.update({"large": "openai/whisper-large-v3", "large-v2": "openai/whisper-large-v2",
                          "large-v3": "openai/whisper-large-v3", "turbo": "openai/whisper-large-v3-turbo"})
MULTILINGUAL_MODEL_IDS = {**ENGLISH_MODEL_IDS,
                          **{size: f"openai/whisper-{size}" for size in ("tiny", "base", "small", "medium")}}
COMMON_FIELDS = ("demand_id", "dataset_index", "audio", "audio_id", "ground_truth_transcript",
                 "duration_seconds", "input_bytes", "audio_sha256", "sample_rate", "num_samples", "benchmark_config")
FLOPS_METHOD = "torch.utils.flop_counter.FlopCounterMode; full pipeline supported tensor ops; multiply-add=2 FLOPs"


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(obj, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def get_model_id(size, multilingual=False):
    return (MULTILINGUAL_MODEL_IDS if multilingual else ENGLISH_MODEL_IDS)[size]


def select_device(requested):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return 0, torch.float16, "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but unavailable")
        return "mps", torch.float32, "mps"
    return -1, torch.float32, "cpu"


def synchronize_device(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def extract_audio(example):
    import numpy as np
    import soundfile as sf

    audio = example["audio"]
    if not isinstance(audio, dict):
        raise TypeError("Use datasets.Audio(decode=False) to read the original encoded audio")
    encoded = audio.get("bytes")
    if encoded is None:
        path = audio.get("path")
        if not path:
            raise ValueError("Audio has neither encoded bytes nor a local file path")
        encoded = Path(path).read_bytes()
    if not encoded:
        raise ValueError("Empty encoded audio")
    waveform, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    waveform = np.asarray(waveform.mean(axis=1), dtype=np.float32)
    if sample_rate <= 0 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Invalid sample rate or empty/non-finite waveform")
    return waveform, int(sample_rate), {
        "duration_seconds": len(waveform) / sample_rate,
        "input_bytes": len(encoded),
        "audio_sha256": hashlib.sha256(encoded).hexdigest(),
        "sample_rate": int(sample_rate), "num_samples": len(waveform),
    }


def get_audio_name(example, index):
    name = example.get("file") or example.get("audio", {}).get("path")
    if name:
        return Path(str(name)).name
    return f"{example.get('id', index)}.flac"


def demand_id(example, index, config, split):
    return f"openslr/librispeech_asr/{config}/{split}/{example.get('id', index)}"


def normalize_text(tokenizer, text):
    for name in ("normalize", "_normalize"):
        normalizer = getattr(tokenizer, name, None)
        if normalizer is not None:
            return normalizer(text)
    raise RuntimeError("Whisper tokenizer must expose its English text normalizer")


def outcome_metrics(reference, prediction, metric, wer_function):
    sample_wer = float(wer_function(reference, prediction)) if reference else (0.0 if not prediction else 1.0)
    if not math.isfinite(sample_wer) or sample_wer < 0:
        raise ValueError("WER must be finite and non-negative")
    exact = reference == prediction
    performance = (100.0 if exact else 0.0) if metric == "exact-match" else 100.0 * max(0.0, 1.0 - sample_wer)
    return {"wer": sample_wer, "correct": exact, "performance": performance,
            "performance_metric": metric}


def transcribe_audio(asr, pipeline_input, duration_s, generate_kwargs=None):
    long_form = duration_s > 30.0
    # Transformers ASR preprocessing pops keys from dictionary inputs.
    # Each timed/profiled call needs its own mapping.
    prediction = asr(dict(pipeline_input), return_timestamps=long_form,
                     generate_kwargs=dict(generate_kwargs or {}))
    return prediction, long_form


def benchmark_query(asr, pipeline_input, duration_s, generate_kwargs, device):
    """Measure latency normally, then repeat greedy inference to count FLOPs."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        prediction, long_form = transcribe_audio(asr, pipeline_input, duration_s, generate_kwargs)
    synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        profiled, _ = transcribe_audio(asr, pipeline_input, duration_s, generate_kwargs)
    synchronize_device(device)
    if prediction["text"] != profiled["text"]:
        raise RuntimeError("Timed and profiled transcriptions differ; cannot attach their costs to one outcome")
    gflops = float(counter.get_total_flops()) / 1e9
    if not math.isfinite(gflops) or gflops <= 0:
        raise RuntimeError("No valid FLOPs counted; use a PyTorch model with eager attention and supported operators")
    return prediction, long_form, elapsed_ms, gflops


def validate_record(record):
    for field in COMMON_FIELDS + ("model", "predicted_transcript", "normalized_reference",
                                  "normalized_prediction", "gflops", "performance", "wer", "correct"):
        if field not in record:
            raise ValueError(f"Missing {field}; old prediction files must be regenerated")
    for field in ("duration_seconds", "input_bytes", "gflops"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {field} in {record['demand_id']}")
    score = record["performance"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("performance must be a finite percentage")
    if not isinstance(record["correct"], bool):
        raise ValueError("correct must be a boolean")
    metric = record["benchmark_config"].get("performance_metric")
    if metric not in ("word-accuracy", "exact-match") or record.get("performance_metric") != metric:
        raise ValueError("Record performance metric does not match the benchmark configuration")
    if record["correct"] != (record["normalized_reference"] == record["normalized_prediction"]):
        raise ValueError("Correctness disagrees with normalized transcripts")
    sample_wer = record["wer"]
    if isinstance(sample_wer, bool) or not isinstance(sample_wer, (int, float)) or not math.isfinite(sample_wer) or sample_wer < 0:
        raise ValueError("WER must be finite and non-negative")
    expected = (100.0 if record["correct"] else 0.0) if metric == "exact-match" else 100.0 * max(0.0, 1.0 - sample_wer)
    if not math.isclose(score, expected, abs_tol=1e-8):
        raise ValueError("Performance disagrees with the selected metric")


def merge_predictions(predictions):
    if set(predictions) != set(MODEL_SIZES):
        raise ValueError("STOP Whisper needs tiny, small, medium, and large; base is a different model")
    indexes = {}
    expected_config = None
    for size in MODEL_SIZES:
        index = {}
        model_ids = set()
        for record in predictions[size]:
            validate_record(record)
            key = record["demand_id"]
            if key in index:
                raise ValueError(f"Duplicate demand in {size}: {key}")
            if expected_config is None:
                expected_config = record["benchmark_config"]
            if record["benchmark_config"] != expected_config:
                raise ValueError("Cannot mix dataset/metric/generation configurations")
            index[key] = record
            model_ids.add(record["model"])
        expected_model = get_model_id(size, expected_config.get("multilingual", False)) if expected_config else None
        if not index or model_ids != {expected_model}:
            raise ValueError(f"Empty or incorrect checkpoint results for {size}")
        indexes[size] = index
    keys = set(indexes["tiny"])
    if any(set(index) != keys for index in indexes.values()):
        raise ValueError("All four models must cover exactly the same audio demand IDs")
    merged = []
    for key in sorted(keys, key=lambda key: indexes["tiny"][key]["dataset_index"]):
        base = {field: indexes["tiny"][key][field] for field in COMMON_FIELDS}
        models = {}
        for size in MODEL_SIZES:
            record = indexes[size][key]
            if any(record[field] != base[field] for field in COMMON_FIELDS):
                raise ValueError(f"Audio metadata differs across models: {key}")
            models[size] = {field: value for field, value in record.items() if field not in COMMON_FIELDS}
        merged.append({**base, "models": models})
    return merged


def load_existing_results(path, config, model_id, dataset, total):
    path = Path(path)
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list) or len(records) > total:
        raise ValueError("Resume file must be a list no longer than the requested sample range")
    for index, record in enumerate(records):
        validate_record(record)
        if record["dataset_index"] != index or record["model"] != model_id or record["benchmark_config"] != config:
            raise ValueError("Resume file is not a matching contiguous prefix; use another output directory or --no-resume")
        example = dataset[index]
        if record["demand_id"] != demand_id(example, index, config["dataset_config"], config["split"]) or record["ground_truth_transcript"] != example["text"].strip():
            raise ValueError("Resume records do not match the current dataset")
    return records


def create_summary(results, wer_function):
    if not results:
        return {"num_samples": 0}
    durations = [record["duration_seconds"] for record in results]
    timings = [record["inference_time_ms"] for record in results]
    return {
        "num_samples": len(results), "model": results[0]["model"],
        "benchmark_config": results[0]["benchmark_config"],
        "corpus_wer": float(wer_function([r["normalized_reference"] for r in results],
                                        [r["normalized_prediction"] for r in results])),
        "mean_sample_wer": statistics.mean(r["wer"] for r in results),
        "mean_performance": statistics.mean(r["performance"] for r in results),
        "exact_match_accuracy": sum(r["correct"] for r in results) / len(results),
        "mean_inference_time_ms": statistics.mean(timings),
        "median_inference_time_ms": statistics.median(timings),
        "p95_inference_time_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
        "mean_gflops": statistics.mean(r["gflops"] for r in results),
        "long_form_samples": sum(r["long_form"] for r in results),
        "total_audio_duration_s": sum(durations), "total_inference_time_s": sum(timings)/1000,
        "overall_real_time_factor": sum(timings)/1000/sum(durations),
        "timing_scope": "ASR call on decoded waveform; loading, decoding and separate FLOPs pass excluded",
        "gflops_method": FLOPS_METHOD,
    }


def create_asr_pipeline(model_id, device, dtype):
    from transformers import pipeline

    # Older Transformers names this argument torch_dtype. Passing dtype there
    # falls through **kwargs into ASR sanitization instead of model loading.
    dtype_argument = "dtype" if "dtype" in signature(pipeline).parameters else "torch_dtype"
    return pipeline("automatic-speech-recognition", model=model_id, device=device,
                    model_kwargs={"attn_implementation": "eager"},
                    **{dtype_argument: dtype})


def result_paths(args, size):
    if args.output:
        output = Path(args.output)
        return output, output.with_suffix(".summary.json")
    root = Path(args.output_dir)
    return root / "predictions" / f"{size}.json", root / "metrics" / f"{size}.summary.json"


def benchmark_model(args, size, dataset, total):
    import torch
    from jiwer import wer
    from tqdm import tqdm

    output, summary_output = result_paths(args, size)
    model_id = get_model_id(size, args.multilingual)
    pipeline_device, dtype, device = select_device(args.device)
    config = {"schema_version": 1, "dataset": "openslr/librispeech_asr",
              "dataset_config": args.dataset_config, "split": args.split,
              "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
              "multilingual": args.multilingual, "performance_metric": args.performance_metric,
              "normalizer": "whisper_english", "decoding": "greedy_temperature_0_beams_1",
              "attention": "eager", "gflops_method": FLOPS_METHOD,
              "device": device, "dtype": str(dtype), "torch_version": torch.__version__,
              "transformers_version": version("transformers"), "jiwer_version": version("jiwer")}
    results = [] if args.no_resume else load_existing_results(output, config, model_id, dataset, total)
    if len(results) == total:
        atomic_write_json(summary_output, create_summary(results, wer))
        return results
    print(f"Loading {model_id}; resuming at {len(results)} / {total}")
    asr = create_asr_pipeline(model_id, pipeline_device, dtype)
    asr.model.eval()
    generate_kwargs = {"do_sample": False, "num_beams": 1, "temperature": 0.0}
    if not model_id.endswith(".en"):
        generate_kwargs.update(language="english", task="transcribe")
    waveform, rate, metadata = extract_audio(dataset[len(results)])
    with torch.inference_mode():
        transcribe_audio(asr, {"array": waveform, "sampling_rate": rate},
                         metadata["duration_seconds"], generate_kwargs)
    synchronize_device(device)
    try:
        for index in tqdm(range(len(results), total), desc=size):
            example = dataset[index]
            waveform, rate, metadata = extract_audio(example)
            prediction, long_form, elapsed_ms, gflops = benchmark_query(
                asr, {"array": waveform, "sampling_rate": rate}, metadata["duration_seconds"], generate_kwargs, device)
            reference = example["text"].strip()
            predicted = prediction["text"].strip()
            normalized_reference = normalize_text(asr.tokenizer, reference)
            normalized_prediction = normalize_text(asr.tokenizer, predicted)
            record = {"dataset_index": index, "demand_id": demand_id(example, index, args.dataset_config, args.split),
                      "audio": get_audio_name(example, index), "audio_id": example.get("id"),
                      "ground_truth_transcript": reference, "predicted_transcript": predicted,
                      "normalized_reference": normalized_reference, "normalized_prediction": normalized_prediction,
                      **metadata, **outcome_metrics(normalized_reference, normalized_prediction, args.performance_metric, wer),
                      "inference_time_ms": elapsed_ms, "gflops": gflops,
                      "real_time_factor": elapsed_ms/1000/metadata["duration_seconds"],
                      "long_form": long_form, "model": model_id, "benchmark_config": config}
            validate_record(record)
            results.append(record)
            if len(results) % args.save_every == 0:
                atomic_write_json(output, results)
    except BaseException:
        atomic_write_json(output, results)
        raise
    atomic_write_json(output, results)
    atomic_write_json(summary_output, create_summary(results, wer))
    del asr
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def launch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        predictions = {size: json.loads((output_dir / "predictions" / f"{size}.json").read_text()) for size in MODEL_SIZES}
    else:
        from datasets import Audio, load_dataset
        dataset = load_dataset("openslr/librispeech_asr", args.dataset_config, split=args.split)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        total = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
        if not total:
            raise ValueError("Dataset is empty")
        sizes = [args.model_size] if args.model_size else MODEL_SIZES
        predictions = {size: benchmark_model(args, size, dataset, total) for size in sizes}
        if args.model_size:
            print("Single-model predictions saved. Run all four canonical sizes, then --merge-only.")
            return
    merged = merge_predictions(predictions)
    atomic_write_json(output_dir / "demands.json", merged)
    print(f"Saved {len(merged)} demands to {output_dir / 'demands.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=list(ENGLISH_MODEL_IDS), help="One model; default: tiny/small/medium/large and merge")
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--dataset-config", choices=["clean", "other"], default="clean")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", default="json_demands/whisper")
    parser.add_argument("--output", help="Single-model prediction file (requires --model-size)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--performance-metric", choices=["word-accuracy", "exact-match"], default="word-accuracy",
                        help="word-accuracy: 100*max(0,1-WER); exact-match: normalized transcript equality")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if (args.max_samples is not None and args.max_samples <= 0) or args.save_every <= 0:
        parser.error("--max-samples and --save-every must be positive")
    if args.output and not args.model_size:
        parser.error("--output requires --model-size")
    if args.merge_only and (args.model_size or args.output):
        parser.error("--merge-only cannot be combined with --model-size or --output")
    launch(args)
