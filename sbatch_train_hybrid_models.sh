#!/usr/bin/env bash
parallel --delay 25 sbatch --export=ALL,DATASET_NUM={} ./slurm_train_hybrid_models.sh ::: {0..4}
