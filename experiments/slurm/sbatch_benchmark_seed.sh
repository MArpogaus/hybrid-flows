#!/bin bash

parallel --delay 10 sbatch --export=ALL,DATASET_NUM={1},MODEL_NAME={2},SEED={3} slurm_benchmark_seeds.sh ::: {0..4} ::: unconditional_masked_autoregressive_flow_quadratic_spline unconditional_hybrid_masked_autoregressive_flow_quadratic_spline ::: {1..20}
