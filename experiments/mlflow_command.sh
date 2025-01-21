#!/bin/bash
set -euxo pipefail

cd "$(dirname $0)" || exit 1

DVC_LOCAL_PATH=/data/mctm/
DELAY=30
NPROC=${NPROC:-8}

# Configure remote, pull and check out cache
[ -e $DVC_LOCAL_PATH ] && {
    dvc remote add --force --local local /data/mctm/
    dvc pull -r local --force --allow-missing
}

# Helper functions
nvidia_smi_while_proc_is_running(){
    echo "PROCESS IS RUNNING"
    while kill -0 "$1" >/dev/null 2>&1; do
        nvidia-smi
        sleep 1
    done
    echo "PROCESS TERMINATED"
}
dvc_repro_parallel(){
    dvc status $1 | grep @$3 | tr -d : | parallel --ungroup --delay $DELAY $2 dvc repro > dvc_repro_parallel_$1.log 2> dvc_repro_parallel_$1.err.log &
    nvidia_smi_while_proc_is_running $!
}

# reproduce experiments for simulation data
# dvc_repro_parallel eval-sim -j$NPROC

# reproduce experiments for benchmark data
# dvc_repro_parallel train-malnutrition -j4

# reproduce experiments for benchmark data
dvc_repro_parallel train-benchmark -j6 '.*hybrid'

# Ensure pipeline has been fully reproduced
# dvc repro eval-sim
# dvc repro train-benchmark

# Push all changes to the remote cache
[ -e $DVC_LOCAL_PATH ] && {
    dvc push -r local
}
