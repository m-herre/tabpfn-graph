#!/usr/bin/env bash
#SBATCH --job-name=graph-tpfn35
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
repo=${SLURM_SUBMIT_DIR:-$PWD}
python_bin=${PYTHON:-python}
cd "$repo"

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

nvidia-smi
"$python_bin" -u benchmarks/quick_molhiv_tabpfn35.py \
  --preset balanced \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --n-estimators 8 \
  --seed 0 \
  --output "benchmark-results/molhiv-tabpfn35-${SLURM_JOB_ID}.json"
