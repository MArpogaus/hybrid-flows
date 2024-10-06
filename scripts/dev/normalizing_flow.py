# %% import
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import plot_2d_data, plot_samples, setup_latex
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

# %%
print(f"{tf.__version__=}\n{tfp.__version__=}")


# %% functions
def nll(y, dist):
    return -dist.log_prob(y)


def get_marginal_dist(dist):
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(0, 1),
        bijector=tfb.Invert(tfb.Chain(dist.bijector.bijector.bijectors[1:])),
    )


def nll_adj(y, dist):
    marginal_dist = tfd.Independent(
        get_marginal_dist(dist),
        1,
    )

    return -dist.log_prob(y) - marginal_dist.log_prob(y)


def preprocess_dataset(data, model) -> dict:
    return {
        "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
        "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
    }


# %% setup latex for plotting
setup_latex(fontsize=10)

# %% globals
dims = 3
distribution = "normalizing_flow"

# %% Simple Model
model = DensityRegressionModel(
    dims=dims,
    distribution=distribution,
    bijectors=[
        {
            "bijector": "BernsteinPolynomial",
            "bijector_kwargs": {
                "domain": [-5, 5],
                "extrapolation": False,
                "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
                "parameters_constraint_fn_kwargs": {
                    "allow_flexible_bounds": False,
                    # "bounds": "identity",
                    "high": 5,
                    "low": -5,
                    # "min_slope": 0.001
                },
                "parameters_fn": "parameter_vector",
                "parameters_fn_kwargs": {
                    "dtype": "float32",
                    "parameter_shape": [1, 100],
                },
            },
        },
        {
            "bijector": "RealNVP",
            "bijector_kwargs": {
                "nested_bijector": "BernsteinPolynomial",
                "nested_bijector_kwargs": {
                    "domain": [-5, 5],
                    "extrapolation": False,
                    "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
                    "parameters_constraint_fn_kwargs": {
                        "allow_flexible_bounds": False,
                        # "bounds": "identity",
                        "high": 5,
                        "low": -5,
                        # "min_slope": 0.001
                    },
                },
                "num_masked": 1,
                "parameters_fn": "fully_connected_network",
                "parameters_fn_kwargs": {
                    "activation": "relu",
                    "batch_norm": False,
                    "dropout": False,
                    # "hidden_units": [128] * 3,
                    "hidden_units": [16] * 2,
                    "dtype": "float32",
                    "input_shape": [1],
                    "parameter_shape": [1, 100],
                },
            },
        },
    ],
)
dist = model(None)
# %% test
from pprint import pprint
from copy import deepcopy

key1 = "key1"
key2 = "key2"
key3 = "key3"
key4 = "key4"
key5 = "key5"
l = [
    {key1: "a", key2: {"b": 1}},
    {key1: "b", key2: {key5: "h", key3: "take me"}},
    {key1: "c", key2: {key4: "keep me"}},
    {
        key1: "d",
        key2: {
            key5: [
                {key1: "e", key2: {"b": 1}},
                {key1: "f", key2: {key3: "take me"}},
                {key1: "g", key2: {key4: "aaaa"}},
            ]
        },
    },
]


def init_parameters_fn_for_bijectors(l):
    ll = []
    for e in l:
        entry = deepcopy(e)
        if key3 in entry[key2].keys():
            entry[key2][key4] = entry[key2].pop(key3)
        elif key4 not in entry[key2].keys():
            entry[key2][key4] = None
        if isinstance(entry[key2].get(key5, None), list):
            entry[key2][key5] = init_parameters_fn_for_bijectors(entry[key2][key5])
        ll.append(entry)
    return ll


pprint(l)
pprint(init_parameters_fn_for_bijectors(l))
pprint(l)

# %%
nvp = [
    {
        "bijector": "RealNVP",
        "bijector_kwargs": {"num_masked": 1},
        "nested_bijectors": {
            "bijector": "BernsteinPolynomial",
            "bijector_kwargs": {
                "domain": [-5, 5],
                "extrapolation": False,
                "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
                "parameters_constraint_fn_kwargs": {
                    "allow_flexible_bounds": False,
                    # "bounds": "identity",
                    "high": 5,
                    "low": -5,
                    # "min_slope": 0.001
                },
            },
        },
        "parameters_fn": "fully_connected_network",
        "parameters_fn_kwargs": {
            "activation": "relu",
            "batch_norm": False,
            "dropout": False,
            # "hidden_units": [128] * 3,
            "hidden_units": [16] * 2,
            "dtype": "float32",
            "input_shape": [1],
            "parameter_shape": [1, 100],
        },
    },
    tfb.Permute(permutation=[1, 0]),
]

# %% get params
from pprint import pprint
from copy import deepcopy

__BIJECTOR_NAME_KEY__ = "bijector"
__BIJECTOR_KWARGS_KEY__ = "bijector_kwargs"
__NESTED_BIJECTOR_KEY__ = "nested_bijector"
__TRAINABLE_KEY__ = "trainable"
__PARAMETERS_KEY__ = "parameters"
__PARAMETERS_FN_KEY__ = "parameters_fn"
__PARAMETERS_FN_KWARGS_KEY__ = "parameters_fn_kwargs"
__PARAMETERS_CONSTRAINT_FN_KEY__ = "parameters_constraint_fn"
__PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__ = "parameters_constraint_fn_kwargs"


l = [
    {
        __BIJECTOR_NAME_KEY__: "a",
        __BIJECTOR_KWARGS_KEY__: {"b": 1},
        __NESTED_BIJECTOR_KEY__: [
            {
                __BIJECTOR_NAME_KEY__: "b",
                __NESTED_BIJECTOR_KEY__: {
                    __BIJECTOR_NAME_KEY__: "h",
                    __BIJECTOR_KWARGS_KEY__: {"this": "is", "random": "!"},
                    __PARAMETERS_FN_KEY__: 4,
                },
            },
            {
                __BIJECTOR_NAME_KEY__: "c",
                __BIJECTOR_KWARGS_KEY__: {},
                __PARAMETERS_KEY__: "keep me",
            },
            {
                __BIJECTOR_NAME_KEY__: "d",
                __BIJECTOR_KWARGS_KEY__: {},
                __NESTED_BIJECTOR_KEY__: [
                    {__BIJECTOR_NAME_KEY__: "e", __BIJECTOR_KWARGS_KEY__: {"b": 1}},
                    {
                        __BIJECTOR_NAME_KEY__: "f",
                        __BIJECTOR_KWARGS_KEY__: {},
                        __PARAMETERS_FN_KEY__: 42,
                    },
                    {
                        __BIJECTOR_NAME_KEY__: "g",
                        __BIJECTOR_KWARGS_KEY__: {},
                        __PARAMETERS_KEY__: "aaaa",
                    },
                ],
            },
        ],
    }
]

trainable_variables = []
non_trainable_variables = []


def set_parameter_fn(entry):
    parameters_fn = entry.get(__PARAMETERS_FN_KEY__, lambda x: x**2)
    variables = None
    if entry.get(__TRAINABLE_KEY__, True):
        trainable_variables.append(variables)
    else:
        non_trainable_variables.append(variables)

    result = {
        __BIJECTOR_NAME_KEY__: entry[__BIJECTOR_NAME_KEY__],
        __BIJECTOR_KWARGS_KEY__: entry.get(__BIJECTOR_KWARGS_KEY__, {}),
        __PARAMETERS_KEY__: parameters_fn,
    }
    if __NESTED_BIJECTOR_KEY__ in entry.keys():
        result[__NESTED_BIJECTOR_KEY__] = entry[__NESTED_BIJECTOR_KEY__]
    return result


def eval_parameter_fn(entry):
    parameters_fn = entry[__PARAMETERS_KEY__]
    if callable(parameters_fn):
        val = parameters_fn(2)
    else:
        val = parameters_fn
    result = {
        __BIJECTOR_NAME_KEY__: entry[__BIJECTOR_NAME_KEY__],
        __BIJECTOR_KWARGS_KEY__: entry.get(__BIJECTOR_KWARGS_KEY__, {}),
        __PARAMETERS_KEY__: val,
    }
    if __NESTED_BIJECTOR_KEY__ in entry.keys():
        result[__NESTED_BIJECTOR_KEY__] = entry[__NESTED_BIJECTOR_KEY__]
    return result


def init_bijector_fn(entry):
    nested_bijector = entry.get(__NESTED_BIJECTOR_KEY__, None)
    if nested_bijector is not None:
        return f"{entry[__BIJECTOR_NAME_KEY__]}({entry[__PARAMETERS_KEY__]}, [{', '.join(nested_bijector)}])"
    else:
        return f"{entry[__BIJECTOR_NAME_KEY__]}({entry[__PARAMETERS_KEY__]},  **{str(entry[__BIJECTOR_KWARGS_KEY__])})"


def recursive_process(
    list_of_dicts,
    fn,
    nested_key=__NESTED_BIJECTOR_KEY__,
):
    processed_dict = []
    for e in list_of_dicts:
        node = deepcopy(e)
        nested_node = e.get(nested_key, None)
        if nested_node is not None:
            if isinstance(nested_node, dict):
                nested_node = [nested_node]
            node[nested_key] = recursive_process(nested_node, fn)

        node = fn(node)
        processed_dict.append(node)
    return processed_dict


pprint(l)
print("-" * 25)
pl = recursive_process(l, set_parameter_fn)
pprint(pl)
print("-" * 25)
ppl = recursive_process(pl, eval_parameter_fn)
pprint(ppl)
print("-" * 25)
pppl = recursive_process(ppl, init_bijector_fn)
pprint(pppl)
print("-" * 25)
pprint(l)

x = a(
    4,
    [
        b(4, [h(4, **{"this": "is", "random": "!"})]),
        c(4, **{}),
        d(
            4,
            [
                e(4, **{"b": 1}),
                f(42, **{}),
                g(4, **{}),
            ],
        ),
    ],
)
