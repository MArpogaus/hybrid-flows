#!/usr/bin/env bash
dvc status | grep eval.*@ | tr -d : | parallel --delay 5 dvc exp run --force --copy datasets/malnutrition/india.raw --queue {}
