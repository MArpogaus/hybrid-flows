#!/usr/bin/env bash
set -euxo pipefail

dvc exp rm -A
dvc queue remove --success --failed
dvc gc --all-experiments -a -f
git gc --aggressive

MODELS=$(dvc status eval-malnutrition | grep -oP '@\K.*(?=:)' | sort | uniq)
EXP_NAME="malnutrition-seeds-$(date -I)"

for model in $MODELS; do
	for seed in {1..20}; do
		dvc exp run --force --temp \
			-C datasets/malnutrition/india.raw \
			-S "seed=$seed" \
			-S "train-malnutrition-experiment-name=$EXP_NAME" \
			-S "eval-malnutrition-experiment-name=$EXP_NAME" \
			eval-malnutrition@${model} &
		sleep 15
	done
done

wait
