#!/usr/bin/env bash
#!/bin bash
MODELS="
conditional_hybrid_masked_autoregressive_flow_bernstein_poly
conditional_hybrid_masked_autoregressive_flow_quadratic_spline
conditional_multivariate_transformation_model
"

parallel --delay 10 sbatch --export="ALL,MLFLOW_EXPERIMENT_NAME=malnutrition_seeds_$(date -I)" ./slurm_dvc_exp.sh --copy datasets/malnutrition/india.raw --force --temp -S seed={2} eval-malnutrition@{1} ::: $MODELS ::: {1..20}
