# %%
import logging

import numpy as np
import tensorflow as tf
from mctm.models import DensityRegressionModel
from tensorflow_probability import bijectors as tfb

logging.basicConfig(level=logging.DEBUG)
# %%
dims = 3
model = DensityRegressionModel(
    dims=3,
    distribution="normalizing_flow",
    transformations=[
        {
            "bijector_name": "bernstein_poly",
            "parameters_shape": [8],
            "parameter_fn": "parameter_vector",
            "parameter_fn_kwargs": {"dtype": "float32"},
            "parameter_constraints": {
                "name": "mctm.activations.get_thetas_constrain_fn",
                "low": -3,
            },
            "extrapolation": True,
        },
        # {
        #     "bijector_name": "shift",
        #     # "parameters_shape": 1,
        #     "parameter_fn": "parameter_vector",
        #     "parameter_fn_kwargs": {"dtype": "float32"},
        # },
        {
            "bijector_name": "Scale_Matvec_Linear_Operator",
            "parameters_shape": [np.sum(np.arange(dims + 1))],
            "parameter_fn": "parameter_vector",
            "parameter_fn_kwargs": {"dtype": "float32"},
            # "parameter_fn": "parameter_vector_or_simple_network",
            # "parameter_fn_kwargs": {
            #     # "input_shape": 3,
            #     # "hidden_units": [2] * 4,
            #     # "activation": "relu",
            #     # "batch_norm": False,
            #     # "dropout": False,
            #     "conditional": False,
            #     # "conditional_event_shape": (2),
            # },
            "parameter_constraints": lambda x: tf.linalg.LinearOperatorLowerTriangular(
                tfb.FillScaleTriL(diag_shift=1e-5)(x)
            ),
        },
    ],
)

# %%
dist = model(None)  # tf.ones((1, 2)))
# tfd.Independent(dist, 2)
dist

# %%
model.trainable_parameters

# %%
dist.bijector  # .bijector.bijectors

# %%
dist.bijector(tf.ones(3))
