#!/bin/bash
set -eu
set -o pipefail

# Function to display the script usage
usage() {
	echo "Usage: $0 [-h] DATASET_TYPE DATASET_NAME MODEL TRIALS JOBS"
	echo "Example: ./scripts/run_optuna.sh -t benchmark power masked_autoregressive_flow 10 2"
	echo "NOTE: There needs to be a file in optuna directory for the respective arguments."
	echo "    -h               Display this help message"
	echo "    -t               Test mode"
	echo "    -p               Use pruning"
	echo "    -d               Prefix for study name with current date"
	echo "    -v               Log level debug (info otherwise)"
	echo "    DATASET_TYPE     Type of the dataset"
	echo "    DATASET_NAME     Dataset name"
	echo "    MODEL            Model name"
	echo "    TRAILS           Number trials per jobs"
	echo "    JOBS             Number of jobs"
	exit 1
}

# Parse command line arguments
while getopts "htvdp" option; do
	case "$option" in
	h)
		usage
		;;
	t)
		EXTRA_ARGS+="--test-mode=true "
		STUDY_NAME+=test_
		;;
	v)
		EXTRA_ARGS+="--log-level debug "
		;;
	p)
		EXTRA_ARGS+="--use-pruning=true "
		;;
	d)
		STUDY_NAME+=$(date -I)_
		;;
	*)
		echo "Error: Invalid option -$OPTARG"
		usage
		;;
	esac
done
shift $((OPTIND - 1)) # remove flags from args

# Set variables based on provided arguments
if [ "$#" -ne 5 ]; then
	echo "Error: 5 arguments required."
	usage
fi

DATASET_TYPE=$1
DATASET_NAME=$2
MODEL=$3
TRIALS=$4
JOBS=$5

PARAMETER_FILE_PATH=params/${DATASET_TYPE}/${DATASET_NAME}/${MODEL}.yaml
STUDY_BASE_NAME=${MODEL}_${DATASET_TYPE}_${DATASET_NAME}
STUDY_NAME+=${STUDY_BASE_NAME}
EXPERIMENT_NAME=${STUDY_NAME}_optuna
DB_NAME=optuna/optuna_study_${STUDY_NAME}.db
RESULTS_PATH=results/optuna/${STUDY_NAME}
PARAMETER_SPACE_DEFINITION_FILE=optuna/parameter_space_definition_${STUDY_BASE_NAME}.yaml

# Create study
if [ ! -e $DB_NAME ]; then
	optuna create-study --study-name "${STUDY_NAME}" --storage sqlite:///$DB_NAME --directions minimize
fi

run_optuna_script() {
	python scripts/optuna_new.py \
		--experiment-name ${EXPERIMENT_NAME} \
		--parameter_file_path ${PARAMETER_FILE_PATH} \
		--model_name ${MODEL} \
		--dataset_type ${DATASET_TYPE} \
		--dataset_name ${DATASET_NAME} \
		--results_path ${RESULTS_PATH} \
		--parameter_space_definition_file ${PARAMETER_SPACE_DEFINITION_FILE} \
		--study-name ${STUDY_NAME} \
		--load-study=sqlite:///${DB_NAME} \
		--n-trials=${TRIALS} \
		--seed=$1 \
		${EXTRA_ARGS:-}
}

for ((seed = 1; seed <= $JOBS; seed++)); do
	run_optuna_script $seed &
	sleep 5
done

wait # Wait for all the background jobs to finish
