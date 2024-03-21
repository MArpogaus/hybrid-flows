# %% imports
from pprint import pprint

import dvc.api
import tensorflow as tf
from mctm.models import HybridDenistyRegressionModel

# %% params
stage_name = "unconditional_hybrid_pre_trained@coupling_flow-moons"
stage, distribution_dataset = stage_name.split("@")

dims = 2

distribution, dataset = distribution_dataset.split("-")

params = dvc.api.params_show(stages=stage_name)

dataset_kwargs = params["datasets"][dataset]
model_kwargs = params[stage + "_distributions"][distribution][dataset]
fit_kwargs = model_kwargs.pop("fit_kwargs")

pprint(model_kwargs)

# %% params
model_kwargs = {
    # "base_checkpoint_path": "unconditional_elementwise_flow_moons/mcp/weights",
    "base_checkpoint_path": None,
    "base_distribution": "elementwise_flow",
    "base_distribution_kwargs": {
        "allow_flexible_bounds": False,
        "bijector_name": "bernstein_poly",
        "high": 4,
        "low": -4,
        "order": 50,
        "scale": False,
        "shift": False,
        "smooth_bounds": True,
    },
    "base_parameter_kwargs": {"conditional": False, "dtype": "float32"},
    "distribution_kwargs": {
        "bijector_name": "bernstein_poly",
        "coupling_layers": 4,
        "high": 1,
        "low": 0,
        "order": 20,
    },
    "freeze_base_model": False,
    "parameter_kwargs": {
        "activation": "tanh",
        "batch_norm": False,
        "dropout": False,
        "hidden_units": [32, 32, 32],
    },
}

# %% model
model = HybridDenistyRegressionModel(
    dims=dims, distribution=distribution, **model_kwargs
)
model.build([None, dims])
model.summary()
# %% log_prob
dist = model(None)

dist.log_prob(tf.ones((1, dims)))


# %%
@tf.function
def call_log_prob():
    dist = model(None)

    return dist.log_prob(tf.ones((1, dims)))


call_log_prob()
