#!/usr/sbin/bash
#set -euxo pipefail
#set -e

declare -A models=(
    [unconditional_distributions]="
        bernstein_flow
        masked_autoregressive_bernstein_flow
        multivariate_bernstein_flow
        multivariate_normal"

    [conditional_distributions]="
        bernstein_flow
        multivariate_bernstein_flow
        multivariate_normal
        masked_autoregressive_bernstein_flow
        coupling_bernstein_flow"

    [unconditional_hybrid_distributions]="
        coupling_bernstein_flow"

    [unconditional_hybrid_pre_trained_distributions]="
        coupling_bernstein_flow"
)

datasets="
    moons_dataset
    circles_dataset
"

get_params(){
    # echo "-S $1.$2.$3.fit_kwds.epochs=1"

    echo "-S $1.$2.$3.fit_kwds.batch_size=512,1024
          -S $1.$2.$3.fit_kwds.learning_rate=0.05,0.01,0.005
          -S $1.$2.$3.fit_kwds.lr_patience=100,1000"

    if [ "$2" != "multivariate_normal" ]; then
        echo "-S $1.$2.$3.distribution_kwds.order=50,80"
    fi

    if [ "$2" != "bernstein_flow" ] && [ "$2" != "multivariate_bernstein_flow" ] && [ "$2" != "multivariate_normal" ]; then
        echo "-S $1.$2.$3.parameter_kwds.hidden_units=[16,16],[16,16,16],[512,512]"
    fi

}

# clean dvc queue
dvc queue stop --kill
dvc queue remove --all
rm -r results/ || echo "results not present" # to remove old artifacts
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

#[ ! $(pgrep mlflow) ] && mlflow ui &
#export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

echo "executing experiments"
while true; do
  status="$(dvc queue status | head -n-2 | tail -n+2)"
  status=$(echo "$status" | grep "Queued")

  if [ -z "${status}" ]; then
    break  # Exit the loop when ids array is empty
  fi

  ids=($(echo "$status" | sed -n 's/^\([^[:space:]]\+\).*/\1/p'))



  #dvc queue start

  

  dvc queue start -j 10
  sleep 900 # every 15 minutes
  #parallel --joblog my_joblog.log -j 10 --eta run_exp_with_id {} ::: "${ids[@]}"
done


#dvc queue start -j 10 # NOTE: Possible race condition on lock > -j 1 leading to failed tasks, also worker not picking up new tasks
# show progress every 30 minutes because it is time consuming
#watch -n 1800 echo "Progress" $(dvc queue status | grep "Success\|Failed" | wc -l) "/" $(dvc queue status | wc -l) $(dvc queue status | grep Failed | wc -l) Failed
