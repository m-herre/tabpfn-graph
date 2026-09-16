#!/usr/bin/env bash
#SBATCH --job-name=molhiv-tabpfn35
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=benchmark-results/slurm/%x-%j.out
#SBATCH --error=benchmark-results/slurm/%x-%j.err

set -euo pipefail

repo=${SLURM_SUBMIT_DIR:-$PWD}
python_bin=${PYTHON:-python}

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$repo/benchmark-results/matplotlib-cache"
export TABPFN_NO_BROWSER=1
# OGB 1.3.6 stores trusted PyG dataset objects, not a weights-only checkpoint.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

nvidia-smi
"$python_bin" -u -c 'import tabpfn, torch; print({"tabpfn": tabpfn.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)})'
"$python_bin" -u benchmarks/benchmark_ogb.py \
  --dataset ogbg-molhiv \
  --root data/ogb \
  --presets balanced \
  --models tabpfn \
  --random-state 0 \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --output benchmark-results/ogbg-molhiv-tabpfn35-balanced-seed0.json
