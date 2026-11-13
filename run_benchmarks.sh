#!/usr/bin/env bash
# Run the empirical workload exporters sequentially to avoid GPU contention.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_dir"

whisper_python="${WHISPER_PYTHON:-${PYTHON:-python}}"
yolo_python="${YOLO_PYTHON:-${PYTHON:-python}}"
epochs=100
imgsz=224
data="data/animals/complete"
device=auto
max_samples=""
whisper_output="benchmark_results_whisper"
yolo_output="benchmark_results"
project="runs_yolo"
whisper_json="json_demands/whisper.json"
yolo_json="json_demands/yolo.json"
dry_run=false

usage() {
    cat <<'HELP'
Usage: ./run_benchmarks.sh [options]

Runs Whisper tiny/small/medium/large, then trains and exports YOLO nano/medium/large.
Whisper resumes existing compatible results. YOLO starts a new training run and
uses the best validation checkpoint for each model.
Final datasets: json_demands/whisper.json and json_demands/yolo.json.

Options:
  --epochs N                  YOLO training epochs (default: 100)
  --imgsz N                   YOLO input resolution (default: 224)
  --data PATH                 YOLO dataset (default: data/animals/complete)
  --device auto|cuda|cpu|mps   Device for both exporters (default: auto; cuda uses GPU 0)
  --whisper-max-samples N      Limit Whisper clips (default: full validation split)
  --whisper-output-dir PATH    Whisper export folder (default: benchmark_results_whisper)
  --yolo-output-dir PATH       YOLO export folder (default: benchmark_results)
  --project PATH              YOLO training folder (default: runs_yolo)
  --dry-run                   Print commands without loading data or running models
  -h, --help                  Show this help

Relative paths are resolved from this repository, regardless of your current directory.
Set PYTHON to choose one interpreter, or WHISPER_PYTHON and YOLO_PYTHON separately.
Dependencies must already be installed in the selected environments.
HELP
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --epochs|--imgsz|--data|--device|--whisper-max-samples|--whisper-output-dir|--yolo-output-dir|--project)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --epochs) epochs="$2" ;;
                --imgsz) imgsz="$2" ;;
                --data) data="$2" ;;
                --device) device="$2" ;;
                --whisper-max-samples) max_samples="$2" ;;
                --whisper-output-dir) whisper_output="$2" ;;
                --yolo-output-dir) yolo_output="$2" ;;
                --project) project="$2" ;;
            esac
            shift 2
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for count in "$epochs" "$imgsz" "${max_samples:-1}"; do
    if [[ ! "$count" =~ ^[0-9]+$ || ! "$count" =~ [1-9] ]]; then
        printf 'Epochs, image size, and sample limit must be positive integers.\n' >&2
        exit 2
    fi
done

whisper_cmd=("$whisper_python" src/whisper/save_json.py --output-dir "$whisper_output")
yolo_cmd=("$yolo_python" src/yolo/save_json.py --data "$data" --epochs "$epochs"
          --imgsz "$imgsz" --output-dir "$yolo_output" --project "$project")
case "$device" in
    auto) whisper_cmd+=(--device auto) ;;
    cuda) whisper_cmd+=(--device cuda); yolo_cmd+=(--device 0) ;;
    cpu|mps) whisper_cmd+=(--device "$device"); yolo_cmd+=(--device "$device") ;;
    *) printf 'Device must be auto, cuda, cpu, or mps.\n' >&2; exit 2 ;;
esac
if [[ -n "$max_samples" ]]; then
    whisper_cmd+=(--max-samples "$max_samples")
fi

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

printf 'Whisper command:\n'
print_command "${whisper_cmd[@]}"
printf 'YOLO command:\n'
print_command "${yolo_cmd[@]}"
printf 'Final datasets: %s and %s\n' "$whisper_json" "$yolo_json"
if "$dry_run"; then
    exit 0
fi

# Fail early, before starting either expensive benchmark, if an environment is incomplete.
"$whisper_python" -c 'import datasets, transformers, soundfile, jiwer, numpy, tqdm; from torch.utils.flop_counter import FlopCounterMode'
"$yolo_python" -c 'import torch, ultralytics, PIL, thop'
if [[ ! -d "$data/train" || ! -d "$data/val" ]]; then
    printf 'YOLO dataset must contain train/ and val/: %s\n' "$data" >&2
    exit 2
fi
if [[ "$whisper_output" == "$yolo_output" ]]; then
    printf 'Whisper and YOLO need different output folders to preserve both demands.json files.\n' >&2
    exit 2
fi

mkdir -p -- json_demands
printf '\nRunning Whisper...\n'
"${whisper_cmd[@]}"
cp -- "$whisper_output/demands.json" "$whisper_json.tmp"
mv -- "$whisper_json.tmp" "$whisper_json"
printf '\nRunning YOLO...\n'
"${yolo_cmd[@]}"
cp -- "$yolo_output/demands.json" "$yolo_json.tmp"
mv -- "$yolo_json.tmp" "$yolo_json"
printf '\nExports complete:\n  Whisper: %s\n  YOLO: %s\n' \
       "$whisper_json" "$yolo_json"
