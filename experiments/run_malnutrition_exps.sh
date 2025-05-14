#!/usr/bin/env bash
set -euxo pipefail

# Cleanup previous runs
# dvc exp rm -A
# dvc queue remove --success --failed
# dvc gc --all-experiments -a -f
# git gc --aggressive

MODELS=$(dvc status eval-malnutrition | grep -oP '@\K.*(?=:)' | sort | uniq)
PYTHON="srun --partition=gpu1 --gres=gpu:1 --mem=256GB --time=48:00:00 --export=ALL,MLFLOW_TRACKING_URI=http://$(hostname):5000 python"
EXP_NAME="malnutrition-seeds-$(date -I)"

for model in $MODELS; do
	for seed in {1..20}; do
		dvc exp run --force --temp \
			-C datasets/malnutrition/india.raw \
			-S "python=$PYTHON" \
			-S "seed=$seed" \
			-S "train-malnutrition-experiment-name=$EXP_NAME" \
			-S "eval-malnutrition-experiment-name=$EXP_NAME" \
			eval-malnutrition@${model} &
		sleep 15
	done
done

wait
