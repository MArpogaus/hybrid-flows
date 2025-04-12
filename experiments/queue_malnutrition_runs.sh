#!/usr/bin/env bash
set -euxo pipefail

dvc exp rm -A
dvc queue remove --success --failed
dvc gc --all-experiments -a -f
git gc --aggressive

MODELS=$(dvc status eval-malnutrition | grep -oP '@\K.*(?=:)' | sort | uniq)

for model in $MODELS; do
	exp_name="malnutrition-seeds-$(date -I)"
	dvc exp run --force --queue -S "seed=range(1,21)" \
		-S "train-malnutrition-experiment-name=$exp_name" \
		-S "eval-malnutrition-experiment-name=$exp_name" \
		eval-malnutrition@${model}
done
