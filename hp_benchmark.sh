#!/usr/sbin/bash
#set -euxo pipefail
#set -e

declare -A models=(
    [unconditional_benchmark_distributions]="
        bernstein_flow
        masked_autoregressive_bernstein_flow"
)

datasets="
    POWER_dataset
    HEPMASS_dataset
    BSDS300_dataset
    MINIBOONE_dataset
"

get_params(){

    echo "-S $1.$2.$3.fit_kwds.batch_size=512,1024
          -S $1.$2.$3.fit_kwds.learning_rate=0.5,0.1,0.05,0.01,0.005,0.001
          -S $1.$2.$3.fit_kwds.lr_patience=10,50,100,500,1000,5000,10000"

    if [ "$2" == "masked_autoregressive_bernstein_flow" ]; then
        echo "-S $1.$2.$3.parameter_kwds.hidden_units=[16,16],[16,16,16],[512,512],[512,1028,512],[1028,512,1028],[16,16,16,16,16]"
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

  

  dvc queue start -j 20
  sleep 900 # every 15 minutes
  #parallel --joblog my_joblog.log -j 10 --eta run_exp_with_id {} ::: "${ids[@]}"
done


#dvc queue start -j 10 # NOTE: Possible race condition on lock > -j 1 leading to failed tasks, also worker not picking up new tasks
# show progress every 30 minutes because it is time consuming
#watch -n 1800 echo "Progress" $(dvc queue status | grep "Success\|Failed" | wc -l) "/" $(dvc queue status | wc -l) $(dvc queue status | grep Failed | wc -l) Failed
