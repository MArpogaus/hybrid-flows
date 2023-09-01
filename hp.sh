#!/usr/sbin/bash
set -euxo pipefail


distributions=(
	bernstein_flow
	coupling_bernstein_flow
	masked_autoregressive_bernstein_flow
	multivariate_bernstein_flow
	multivariate_normal
)

get_params(){
    echo "-S $1.$2.fit_kwds.batch_size=512,1024
          -S $1.$2.fit_kwds.learning_rate=0.05,0.01,0.005
          -S $1.$2.fit_kwds.epochs=5000
          -S $1.$2.fit_kwds.lr_patience=100,1000
          -S $1.$2.distribution_kwds.order=50,100
          -S $1.$2.parameter_kwds.hidden_units=[16,16],[512,512]"
}

for stage in unconditional_hybrid_pre_trained_distributions; do
    for distribution in coupling_bernstein_flow; do
        dvc exp run --queue \
            $(get_params $stage $distribution)
    done
done

[ ! $(pgrep mlflow) ] && mlflow ui &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc queue start -j 10
watch dvc queue status
