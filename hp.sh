#!/usr/sbin/bash
set -euxo pipefail


# TODO: how to read this from params? I have only found it provided in the python api as params_show
declare -A models=(
    [unconditional_distributions]="
        bernstein_flow
        masked_autoregressive_bernstein_flow
        multivariate_bernstein_flow
        multivariate_normal"
    [unconditional_benchmark_distributions]="
        bernstein_flow
        masked_autoregressive_bernstein_flow
    "
)

get_params(){
    echo "-S $1.$2.fit_kwds.epochs=1"
    # echo "-S $1.$2.fit_kwds.batch_size=512,1024
    #       -S $1.$2.fit_kwds.learning_rate=0.05,0.01,0.005
    #       -S $1.$2.fit_kwds.epochs=5000
    #       -S $1.$2.fit_kwds.lr_patience=100,1000
    #       -S $1.$2.distribution_kwds.order=50,100
    #       -S $1.$2.parameter_kwds.hidden_units=[16,16],[512,512]"
}

# clean dvc queue
dvc queue stop --kill
dvc queue remove --all

for stage in "${!models[@]}"; do
    distributions=${models[$stage]}
    for distribution in $distributions; do
        # sometimes failes when queued and takes long. why cache not used?
        echo "queueing experiment ${stage}-${distribution}"
        dvc exp run --queue \
            $(get_params $stage $distribution)
    done
done

[ ! $(pgrep mlflow) ] && mlflow ui &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc queue start -j 10 # NOTE: Possible race condition on lock > -j 1 leading to failed tasks, also worker not picking up new tasks
watch dvc queue status
