#!/usr/bin/env bash
dvc exp rm -A
dvc queue remove --all
dvc gc --all-experiments -a -f
git gc --aggressive

MODELS=$(dvc status eval-sim | grep -oP '@dataset\d-\K.*(?=:)' | sort | uniq)

for dataset in {0,1}; do
	for model in $MODELS; do
		exp_name="sim-seeds-$model-$(date -I)"
		dvc exp run --force --queue -S "seed=range(1,21)" \
			-S "train-sim-experiment-name=$exp_name" \
			-S "eval-sim-experiment-name=$exp_name" \
			eval-sim@dataset${dataset}-${model}
	done
done
