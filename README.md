# Sustainable Geo-Distributed AI Inference


### Installing Highs solver
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
export PATH="$PATH:$(pwd)"
./vcpkg install highs
```

### Running STOP
Clone the repository and then run the following commands:
```
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .
cd ..
./build/STOP --method STOP --model_name Yolo --n_machines 3 --max_demands_per_round 1000 --seed 0 --C_avg 0.001 --R_avg 2 --V 500
```

### Sampling demands from JSON

Pass `--demands_json /path/to/demands.json` together with `--model_name Yolo`,
`--model_name Whisper`, or `--model_name Whisper3` to sample the empirical
workload instead of generating synthetic input sizes.
The file is loaded once. Each timestep draws `n_demands` entries uniformly with
replacement using the simulation seed; repeated entries are allowed within and
across rounds. Without this argument, the existing synthetic workloads are used.

For YOLO, the JSON must be a non-empty array with one entry per image:

```json
[
  {
    "image": "data/animals/complete/val/Bear/example.jpg",
    "width": 1920,
    "height": 1080,
    "input_bytes": 345678,
    "models": {
      "nano": {"gflops": 12.3, "correct": true},
      "medium": {"gflops": 28.7, "correct": true},
      "large": {"gflops": 51.2, "correct": false}
    }
  }
]
```

These costs are illustrative. Width and height must be positive integers;
`input_bytes` and all three `gflops` values must be finite and positive.
`get_sizes` returns width times height, `get_sigma` returns transfer bytes, and
`get_omega` returns GFLOPs in nano/medium/large order for the same sampled records.
Specify bytes for the representation actually transmitted, and GFLOPs for the
input after the model's preprocessing. No image files are opened at runtime.
Each model entry must also provide an observed outcome. In priority order, the
loader accepts a numeric `performance` on a 0–100 scale, compares its
`predicted_label` to the record's `ground_truth_label`, or reads its boolean
`correct`. Label comparison uses exact string equality. Confidence and timing
fields are optional and unused. Outcomes are retained with the sampled demand.
For Whisper, use one entry per audio clip, with `duration_seconds`, `input_bytes`,
and four model costs in **tiny/small/medium/large** order, matching the existing
four-model implementation:

```json
[
  {
    "audio": "data/audio/example.wav",
    "duration_seconds": 12.25,
    "input_bytes": 392000,
    "models": {
      "tiny": {"gflops": 10.0, "performance": 93.0},
      "small": {"gflops": 20.0, "performance": 94.0},
      "medium": {"gflops": 30.0, "performance": 95.0},
      "large": {"gflops": 40.0, "performance": 96.0}
    }
  }
]
```

The audio costs above are illustrative, and must describe the full clip rather
than a single processing segment. Duration must be finite and positive.
`get_sizes` rounds duration up to integer seconds to preserve its existing
interface; `get_sigma` and `get_omega` return the exact bytes and GFLOPs supplied
in the JSON, without recomputing them from rounded duration. No audio is opened.
`Whisper3` uses tiny/small/medium and does not require a large-model entry.

The STOP normalization uses the dataset's mean medium-model GFLOPs for YOLO,
and the mean across all selected model variants for Whisper/Whisper3.
The dataset path is saved in `config.json`; result directory naming is unchanged,
so use a separate working directory to preserve runs with identical parameters.

The JSON loader and accessor tests can be run independently of the simulator:

```bash
cmake --build build --target test_json_demands
ctest --test-dir build --output-on-failure
```


### Ground-truth performance feedback

For JSON workloads, model-wide mean performance is calculated once over the full
dataset. The raw target is the unweighted mean of those model means, fixed for
all timesteps. The dispatcher uses those model-wide estimates when selecting
assignments; individual sampled outcomes enter feedback only after scheduling.

STOP scales expected scores, the target, and achieved per-model batch averages
using the same piecewise linear function, with linear extrapolation outside the
anchors (no clamping):

| Family | Raw percentage → scaled score |
| --- | --- |
| YOLO | 85 → 1, 90 → 50, 95 → 100 |
| Whisper | 93 → 1, 94 → 33, 95 → 66, 96 → 100 |
| Whisper3 | 93 → 1, 94 → 50, 95 → 100 |

After scheduling, each model's accuracy is averaged over its assigned demands.
The feedback deficit is `sum_j count_j * (scaled_target - scale(achieved_j))`.
Models with zero assignments contribute zero. `performance` logs overall raw
achieved performance; `performance_j` logs raw per-model means, using JSON null
when no demands were assigned. `x_at_delta` logs the deficit.
The STOP objective and accuracy queue use a fixed positive normalization of 50;
an extrapolated target can be zero or negative, so dividing by that target would
be invalid or reverse the feedback sign. The original accuracy weight remains.
This per-model scaled constraint differs from a constraint on overall raw
accuracy, because slopes vary between interpolation segments.

Whisper may provide a task-specific numeric `performance` (higher is better)
instead of exact transcript-match correctness. The example percentages are
illustrative. A transcription metric and its normalization must be chosen before
creating those values; the loader does not calculate WER or normalize transcripts.

`--min_demands_per_round` defaults to **10**. Arrival counts are sampled uniformly
from `[min_demands_per_round, max_demands_per_round - 1]`, so the maximum must
exceed the minimum. Pass `--min_demands_per_round 0` for the earlier arrival range.
At least ten arrivals does not guarantee ten assignments per model: singleton
model groups and extrapolated scores remain possible.


### Exporting YOLO demands

`src/yolo/save_json.py` trains and benchmarks nano, medium, and large classifiers
by default, then writes `benchmark_results/demands.json` in the simulator schema.
It also retains `predictions_n.json`, `predictions_m.json`, `predictions_l.json`
and per-model metrics CSVs. Install `requirements.txt`, including Pillow and
`ultralytics-thop`, in the environment used for training/inference.

```bash
python src/yolo/save_json.py --data data/animals/complete --epochs 10 --imgsz 224
```

To export existing trained checkpoints without training again:

```bash
python src/yolo/save_json.py --skip-train --data data/animals/complete --imgsz 224 \
  --weights n=runs_yolo/n_all/weights/best.pt m=runs_yolo/m_all/weights/best.pt l=runs_yolo/l_all/weights/best.pt
```

An optional `--device cpu` or `--device 0` applies to training, validation and
inference. `--size n` (or `m`, `l`) processes just one variant. After running each
variant separately with the same validation data and output directory, merge:

```bash
python src/yolo/save_json.py --merge-only --output-dir benchmark_results
```

Merging joins exact image paths and requires all three variants with matching
image metadata and no duplicates. Old prediction files without the new metadata
and GFLOPs must be regenerated. `s` is not included because the simulator currently
uses nano/medium/large.

`width` and `height` describe the original image; `input_bytes` is the compressed
file's byte count, assuming the dispatcher transmits that file. Each model result
contains the observed inference `input_shape`, GFLOPs, labels, correctness,
confidence, checkpoint path, device and measured `inference_time_ms`. The latter
includes the prediction call's preprocessing and postprocessing. GFLOPs are
profiled on a model copy using the actual tensor height/width, cached per shape,
and never inferred from wall-clock time. Profiling runs outside the measured
prediction interval and must return a finite, positive cost.

`--imgsz` defaults to 224 and is passed explicitly to training, validation, and
prediction. If preprocessing produces the same tensor size for every image,
GFLOPs are constant per model, even when original image resolutions differ.
Folder names under `val/` must exactly match the checkpoint's class names.

Exporter tests (no Ultralytics installation or model downloads required):

```bash
python -B -m unittest discover -s tests -p 'test_yolo_export.py' -v
```


### Exporting Whisper demands

`src/whisper/save_json.py` benchmarks LibriSpeech with **tiny, small, medium,
and large** and writes `benchmark_results_whisper/demands.json` for
`--model_name Whisper`. Model predictions and summaries are also saved separately.
It preserves manual FLAC decoding, English text normalization, long-form
transcription, and periodic resumable checkpoints from the original benchmark.
Dependencies include Transformers, Datasets, soundfile, jiwer, and a PyTorch
version exposing `torch.utils.flop_counter.FlopCounterMode`.

```bash
python src/whisper/save_json.py --dataset-config clean --split validation --device cuda --max-samples 10
```

Omit `--max-samples` for the full split. The first run loads/downloads LibriSpeech
and the pretrained checkpoints. English-only checkpoints are used where available;
`large` is `openai/whisper-large-v3`. Pass `--multilingual` to select multilingual
checkpoints for the smaller models.

To benchmark separately and merge later, use the same dataset, metric, device,
and configuration for each model:

```bash
python src/whisper/save_json.py --model-size tiny --max-samples 10
python src/whisper/save_json.py --model-size small --max-samples 10
python src/whisper/save_json.py --model-size medium --max-samples 10
python src/whisper/save_json.py --model-size large --max-samples 10
python src/whisper/save_json.py --merge-only
```

Merging joins stable dataset demand IDs and checks that all four models cover
exactly the same samples, transcripts, encoded-audio hashes, metadata and benchmark
configuration. Existing exports without the new fields must be regenerated.
`base`, `large-v2`, `large-v3`, and `turbo` remain available for standalone
benchmarks, but cannot replace the four canonical model entries during merging.
`--output` overrides a single-model prediction filename; merge-only expects the
standard `predictions_tiny/small/medium/large.json` names in `--output-dir`.

Each demand includes exact `duration_seconds`, compressed FLAC `input_bytes`,
sample rate and sample count. The bytes represent sending the original encoded
file, while timing starts after decoding and mono conversion. Each model result
contains its transcripts, normalized transcripts, raw WER, normalized exact-match
`correct`, inference time, GFLOPs, model ID and a numeric simulator `performance`.
The default score is **`100 * max(0, 1 - utterance_WER)`**. WER can exceed one;
only the performance score is clipped at zero, while raw WER remains unchanged.
Use `--performance-metric exact-match` for scores of 0 or 100 instead.
Because the loader prioritizes `performance`, it uses the chosen metric rather
than always treating an imperfect transcript as completely incorrect. This is
an average of per-utterance scores, not corpus-level word accuracy. Summaries
retain corpus WER and mean sample WER separately.

GFLOPs are counted during a **second, untimed, full transcription** with PyTorch's
`FlopCounterMode`. This includes the encoder and the autoregressive decoder,
including long-form processing for clips longer than 30 seconds. The measured
and profiled runs use greedy decoding with eager attention and must produce
identical transcripts. Cost varies with the actual generation path, rather than
being inferred from duration or wall-clock time. Multiplication plus addition
counts as two FLOPs. The counter reports theoretical costs for registered
operators; unsupported operations contribute zero, so this is a supported-operator
compute estimate, not a complete hardware instruction count. See the
[PyTorch FLOP counter documentation](https://docs.pytorch.org/docs/2.14/generated/torch.utils.flop_counter.FlopCounterMode.html).
Profiling adds another inference pass per clip; reported latency excludes it.

Resume validates a contiguous dataset prefix and compatible configuration/model.
Use a different output directory or `--no-resume` when changing the metric or
benchmark configuration. The script saves successful records on interruption
or error. `--merge-only` works without loading models or the dataset.

Whisper exporter tests, including a real PyTorch FLOP-counting check on a small
synthetic transcription function:

```bash
python -B -m unittest discover -s tests -p 'test_whisper_export.py' -v
```


### Running both exporters

```bash
./run_benchmarks.sh --device cuda --epochs 100
```

This runs the full Whisper validation split for all four models, then trains
YOLO nano/medium/large for up to 100 epochs each and exports the best checkpoints.
The defaults are 100 YOLO epochs, input resolution 224, and automatic device
selection. The runner saves the final datasets as `json_demands/whisper.json`
and `json_demands/yolo.json`. Per-model predictions and metrics remain in
`benchmark_results_whisper/` and `benchmark_results/`; YOLO checkpoints go under
`runs_yolo/`. Each final dataset is copied after its exporter succeeds, with an
atomic replacement so a failed copy cannot leave a partially written final file.
Whisper resumes compatible saved results; YOLO starts new training runs.

For a smaller run or to inspect commands without running models:

```bash
./run_benchmarks.sh --whisper-max-samples 10 --epochs 10
./run_benchmarks.sh --dry-run
```

Use `PYTHON=/path/to/python` for a shared environment, or set
`WHISPER_PYTHON` and `YOLO_PYTHON` separately. Both environments must already
contain their exporter dependencies. The runner checks these before starting
and stops if either exporter fails. `./run_benchmarks.sh --help` lists path,
device, and training options. Relative paths are resolved from the repository.
