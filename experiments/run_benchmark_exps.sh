#!/usr/bin/env bash
set -euxo pipefail

# Cleanup previous runs
# dvc exp rm -A
# dvc queue remove --success --failed
# dvc gc --all-experiments -a -f
# git gc --aggressive

MODELS=$(dvc status eval-benchmark | grep -oP '@dataset\d-\K.*(?=:)' | sort | uniq)
PYTHON="srun --partition=gpu1 --gres=gpu:1 --mem=256GB --time=48:00:00 --export=ALL,MLFLOW_TRACKING_URI=http://$(hostname):5000 python"
EXP_NAME="benchmark-seeds-$(date -I)"

for dataset in {0,4}; do
	for model in $MODELS; do
		for seed in {1..20}; do
			dvc exp run --force --temp \
				-S "python=$PYTHON" \
				-S "seed=$seed" \
				-S "train-benchmark-experiment-name=$EXP_NAME" \
				-S "eval-benchmark-experiment-name=$EXP_NAME" \
				eval-benchmark@dataset${dataset}-${model} &
                        sleep 15
		done
	done
done

wait
