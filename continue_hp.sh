#!/usr/sbin/bash
#set -euxo pipefail
#set -e

# echo "executing experiments"
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


#dvc queue start -j 10 # NOTE: Possible race condition on lock > -j 1 leading to failed tasks, also worker not picking up new tasks
# show progress every 30 minutes because it is time consuming
#watch -n 1800 echo "Progress" $(dvc queue status | grep "Success\|Failed" | wc -l) "/" $(dvc queue status | wc -l) $(dvc queue status | grep Failed | wc -l) Failed
