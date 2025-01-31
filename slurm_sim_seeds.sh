#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output="slurm_train_hybrid_models_%j.out"
source ~/.bashrc

set -euo pipefail

git config --global --replace-all user.email slurm
git config --global --replace-all user.name slurm@invalid.com

mamba activate mctm

python --version
mamba --version

nvidia-smi
nvidia-smi -pm 1

python -c 'import tensorflow as tf; assert len(tf.config.list_physical_devices("GPU")) > 0'

cd experiments

env | sort

export MLFLOW_EXPERIMENT_NAME="sim_seeds_$(date -I)_${SEED}"
export MLFLOW_TRACKING_URI=http://login1:5000

dvc exp run --force --temp -S "seed=${SEED}" "eval-sim@dataset${DATASET_NUM}-${MODEL_NAME}"
