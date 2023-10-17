status="$(dvc queue status | head -n-2 | tail -n+2)"
ids=$(echo "$status" | sed -n 's/^\([^[:space:]]\+\).*/\1/p')
ids=( "$ids" )

num_finished=$(echo "$status" | grep "Success\|Failed" | wc -l)
num_total=$(echo "$status" | wc -l )
num_failed=$(echo "$status" | grep Failed | wc -l)
num_running=$(echo "$status" | grep Running | wc -l)
num_queued=$(echo "$status" | grep Queued | wc -l)
num_successful=$(echo "$status" | grep Success | wc -l)
echo "Progress" $num_finished "/" $num_total "($num_running Running, $num_queued Queued, $num_successful Successful, $num_failed Failed)"
