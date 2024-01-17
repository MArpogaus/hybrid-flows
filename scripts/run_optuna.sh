#!/bin/bash
set -eu
set -o pipefail

# Function to display the script usage
usage() {
	echo "Usage: $0 [-h] STAGE_NAME DISTRIBUTION DATASET STUDY_NAME JOBS"
	echo "    -h               Display this help message"
	echo "    STAGE_NAME       Name of the stage"
	echo "    DISTRIBUTION     Distribution type"
	echo "    DATASET          Dataset name"
	echo "    STUDY_NAME       Study name"
	echo "    TRAILS           Number trials per jobs"
	echo "    JOBS             Number of jobs"
	exit 1
}

# Parse command line arguments
while getopts ":h" option; do
	case "$option" in
	h)
		usage
		;;
	*)
		echo "Error: Invalid option -$OPTARG"
		usage
		;;
	esac
done

# Set variables based on provided arguments
if [ "$#" -ne 5 ]; then
	echo "Error: 5 arguments required."
	usage
fi

STAGE_NAME=$1
DISTRIBUTION=$2
DATASET=$3
TRIALS=$4
JOBS=$5

STUDY_NAME=${STAGE_NAME}_${DISTRIBUTION}_${DATASET}
DB_NAME=optuna/optuna_study_${STUDY_NAME}.db
RESULTS_PATH=results/optuna/${STUDY_NAME}
PARAMETER_SPACE_DEFINITION_FILE=optuna/parameter_space_definition_${STUDY_NAME}.yaml

# Create study
if [ ! -e $DB_NAME ]; then
	optuna create-study --study-name "${STUDY_NAME}" --storage sqlite:///$DB_NAME --directions minimize
fi

run_optuna_script() {
	python scripts/optuna_new.py \
		--experiment-name optuna \
		--stage-name ${STAGE_NAME}@${DISTRIBUTION}-${DATASET} \
		--distribution ${DISTRIBUTION} \
		--dataset ${DATASET} \
		--results_path ${RESULTS_PATH} \
		--parameter_space_definition_file ${PARAMETER_SPACE_DEFINITION_FILE} \
		--study-name=${STUDY_NAME} \
		--load-study=sqlite:///${DB_NAME} \
		--n-trials=${TRIALS} \
        --use-pruning=true \
		--seed=$1
}

for ((seed = 1; seed <= $JOBS; seed++)); do
	run_optuna_script $seed &
    sleep 1
done

wait # Wait for all the background jobs to finish
