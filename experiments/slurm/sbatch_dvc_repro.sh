#!/usr/bin/env bash
cd experiments

dvc status | grep @ | tr -d : | parallel --delay 25 sbatch ../slurm_dvc_exp.sh {}
