#!/usr/sbin/bash
#set -euxo pipefail
#set -e

declare -A models=(
    [unconditional_benchmark_distributions]="
        bernstein_flow
        masked_autoregressive_bernstein_flow"
)

datasets="
    power
    hepmass
    miniboone
"

get_params(){

    if [ "$2" == "masked_autoregressive_bernstein_flow" ]; then
        echo "-S $1.$2.$3.fit_kwds.batch_size=32,128,512
          -S $1.$2.$3.fit_kwds.learning_rate=0.01,0.001
          -S $1.$2.$3.fit_kwds.lr_patience=5,10
          -S $1.$2.$3.parameter_kwds.hidden_units=[64,64,64],[512,512,512]"
    else
        echo "-S $1.$2.$3.fit_kwds.batch_size=512,1024,2048
          -S $1.$2.$3.fit_kwds.learning_rate=0.01,0.001,0.0001
          -S $1.$2.$3.fit_kwds.lr_patience=10,25,40,1000"
    fi

}

# clean dvc queue
dvc queue stop --kill
dvc queue remove --all

rm -r results/ || echo "results not present" # to remove old artifacts
dvc checkout
# reproduce pipeline to cache results
dvc repro

#rm my_joblog.log || echo "logfile not present" # used by parallel
counter=0
for stage in "${!models[@]}"; do
    distributions=${models[$stage]}
    for distribution in $distributions; do
        for dataset in $datasets; do
            echo "queueing experiment ${stage}-${distribution}"
            dvc exp run --queue \
                $(get_params $stage $distribution $dataset)
            counter=$((counter+1))
        done
    done
done

# [ ! $(pgrep mlflow) ] && mlflow ui &
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

echo "executing experiments"
# while true; do
#   status="$(dvc queue status | head -n-2 | tail -n+2)"
#   status=$(echo "$status" | grep "Queued")

#   if [ -z "${status}" ]; then
#     break  # Exit the loop when ids array is empty
#   fi

#   ids=($(echo "$status" | sed -n 's/^\([^[:space:]]\+\).*/\1/p'))



#   #dvc queue start

  

#   dvc queue start -j 5
#   sleep 900 # every 15 minutes
#   #parallel --joblog my_joblog.log -j 10 --eta run_exp_with_id {} ::: "${ids[@]}"
# done


dvc queue start -j 10
