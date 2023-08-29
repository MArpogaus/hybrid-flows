#!/usr/sbin/bash
set -euxo pipefail


distributions=(
	bernstein_flow
	coupling_bernstein_flow
	masked_autoregressive_bernstein_flow
	multivariate_bernstein_flow
	multivariate_normal
)

get_fit_kwds(){
    echo "-S $1.$2.fit_kwds.batch_size=32,128,512
          -S $1.$2.fit_kwds.learning_rate=0.001,0.005
          -S $1.$2.fit_kwds.lr_patience=25,100"
}

for stage in unconditional_hybrid_distributions; do
    for distribution in coupling_bernstein_flow masked_autoregressive_bernstein_flow; do
        dvc exp run --queue \
             -S $stage.$distribution.parameter_kwds.hidden_units=[16,16],[512,512],[16,16,16] \
             $(get_fit_kwds $stage $distribution)
    done
done
