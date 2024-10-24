#!/bin/bash
set -u

cd $(dirname $0)

# Configure remote, pull and check out cache
dvc remote add --force --local local /data/mctm/
dvc pull -r local --force
dvc checkout

# Helper functions
dvc_queue_exp_for_stage(){
    for s in $(dvc status | grep "$1" | tr -d :);
    do
        echo "Queuing experiment for stage $s"
        dvc exp run --queue "$s" -n "$s"
    done
}
dvc_apply_all_exps(){
    for s in $(dvc exp ls --names-only);
    do
        echo "Applying experiment '$s' to workspace"
        dvc exp apply "$s"
        dvc push -r local "$s"
    done
}
wait_for_queue() {
    while : ; do
        # Check if there are any queued or running experiments
        queue_status=$(dvc queue status | tail -n +2 | grep -P '(Queued|Running)')

        if [[ -z "$queue_status" ]]; then
            echo "All experiments have been processed."
            break
        fi

        echo "Still having $(wc -l <<< "$queue_status") experiments in the queue..."
        sleep 10  # Wait for 10 seconds before checking again
    done
}
dvc_repro_parallel(){
    # Queue all matching stages as experiments for parallel executions
    dvc_queue_exp_for_stage "$1"

    # Start parallel execution of experiments
    dvc queue start "$2"

    # wait until dvc queue has been processed
    wait_for_queue

    # Show results
    dvc exp show --only-changed

    # Apply all experiments to the workspace
    dvc_apply_all_exps

    # Clear queue and remove experiments
    dvc queue remove --all
    dvc exp rm --rev HEAD
}

# reproduce experiments for simulation data
dvc_repro_parallel train-sim -j16

# reproduce experiments for benchmark data
dvc_repro_parallel train-benchmark -j4
