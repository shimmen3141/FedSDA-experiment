#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  FDE_RUN_DIR=<directory> [FDE_WORKERS=4] bash tools/run_server_sweep.sh <variant> <run_pareto_sweep options...>

Environment variables:
  FDE_RUN_DIR       Result root shared by related variants. If omitted, create a timestamped root.
  FDE_WORKERS       Number of independent run processes (default: 4).
  FDE_VENV_DIR      Virtual environment directory (default: <repository>/.venv).
  FDE_PYTHON        Python executable override (default: <venv>/bin/python).
  FDE_TIME_BIN      GNU time executable (default: /usr/bin/time).
  FDE_NO_RECOVERY   1 to skip automatic recovery plots (default: 1), 0 to enable them.
  FDE_TAG            Output filename tag override (default: variant).
  FDE_DRY_RUN       1 to print the resolved command without running it.

The wrapper owns --workers, --out-dir, --raw-dir, --tag, and --no-recovery.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

variant=$1
shift

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workers=${FDE_WORKERS:-4}
venv_dir=${FDE_VENV_DIR:-"$repo_root/.venv"}
python_bin=${FDE_PYTHON:-"$venv_dir/bin/python"}
time_bin=${FDE_TIME_BIN:-/usr/bin/time}
no_recovery=${FDE_NO_RECOVERY:-1}
tag=${FDE_TAG:-$variant}
run_dir=${FDE_RUN_DIR:-"$repo_root/results/results_$(date +%Y%m%d_%H%M%S)"}

if [[ ! $workers =~ ^[1-9][0-9]*$ ]]; then
    echo "FDE_WORKERS must be a positive integer: $workers" >&2
    exit 2
fi
if [[ $no_recovery != 0 && $no_recovery != 1 ]]; then
    echo "FDE_NO_RECOVERY must be 0 or 1: $no_recovery" >&2
    exit 2
fi
for argument in "$@"; do
    case "$argument" in
        --workers|--out-dir|--raw-dir|--tag|--no-recovery)
            echo "$argument is managed by run_server_sweep.sh" >&2
            exit 2
            ;;
    esac
done

if [[ $run_dir != /* ]]; then
    run_dir="$repo_root/$run_dir"
fi
variant_dir="$run_dir/$variant"
pareto_dir="$variant_dir/pareto"
raw_dir="$variant_dir/raw"
log_dir="$run_dir/logs"
mkdir -p "$pareto_dir" "$raw_dir" "$log_dir"

if [[ ${FDE_DRY_RUN:-0} != 1 ]]; then
    if [[ ! -x $python_bin ]]; then
        echo "Python executable was not found: $python_bin" >&2
        exit 2
    fi
    if [[ ! -x $time_bin ]]; then
        echo "GNU time executable was not found: $time_bin" >&2
        exit 2
    fi
fi

command=(
    "$python_bin" -u "$repo_root/run_pareto_sweep.py"
    "$@"
    --workers "$workers"
    --out-dir "$pareto_dir"
    --raw-dir "$raw_dir"
    --tag "$tag"
)
if [[ $no_recovery == 1 ]]; then
    command+=(--no-recovery)
fi

echo "Run directory: $run_dir"
echo "Variant: $variant"
echo "Workers: $workers"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ ${FDE_DRY_RUN:-0} == 1 ]]; then
    exit 0
fi

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
"$time_bin" -v -o "$log_dir/$variant.time.txt" \
    "${command[@]}" 2>&1 | tee "$log_dir/$variant.log"
