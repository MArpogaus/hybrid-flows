#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=marpogau@htwg-konstanz.de
#SBATCH --output="slurm_train_hybrid_models_%j.out"
source ~/.bashrc

set -euo pipefail

mamba activate mctm

python --version
mamba --version

nvidia-smi

python -c 'import tensorflow as tf; assert len(tf.config.list_physical_devices("GPU")) > 0'

cd experiments

env | sort

dvc repro -f --glob "eval-benchmark@dataset${DATASET_NUM}*hybrid*"
