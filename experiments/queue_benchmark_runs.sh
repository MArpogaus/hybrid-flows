#!/usr/bin/env bash
set -euxo pipefail

dvc exp rm -A
dvc queue remove --success --failed
dvc gc --all-experiments -a -f
git gc --aggressive

MODELS=$(dvc status eval-benchmark | grep -oP '@dataset\d-\K.*(?=:)' | sort | uniq)

for dataset in {0,4}; do
	for model in $MODELS; do
		exp_name="benchmark-seeds-$(date -I)"
		dvc exp run --force --queue -S "seed=range(1,21)" \
			-S "train-benchmark-experiment-name=$exp_name" \
			-S "eval-benchmark-experiment-name=$exp_name" \
			eval-benchmark@dataset${dataset}-${model}
	done
done
