#!/usr/bin/env bash
dvc exp rm -A
dvc queue remove --all
dvc gc --all-experiments -a -f
git gc --aggressive

dvc status | grep eval.*@ | tr -d : | parallel --delay 5 dvc exp run --force --copy datasets/malnutrition/india.raw --queue {}
