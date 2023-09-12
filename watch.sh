status="$(dvc queue status | head -n-2 | tail -n+2)"
ids=$(echo "$status" | sed -n 's/^\([^[:space:]]\+\).*/\1/p')
ids=( "$ids" )

echo "$ids"

num_finished=$(echo "$status" | grep "Success\|Failed" | wc -l)
 num_total=$(echo "$status" | wc -l )
 num_failed=$(echo "$status" | grep Failed | wc -l)
# echo "Progress" $num_finished "/" $num_total $num_failed Failed
