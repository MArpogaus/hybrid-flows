#!/usr/bin/env bash
cd experiments

dvc status | grep eval-sim@ | tr -d : | parallel --delay 25 echo sbatch ./slurm_dvc_exp.sh {}
