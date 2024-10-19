"""Unit test for density regression models."""

import os

import pytest
import tensorflow as tf

from mctm.distributions import (
    __BIJECTOR_KWARGS_KEY__,
    __BIJECTOR_NAME_KEY__,
    __INVERT_BIJECTOR_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
    __PARAMETERS_FN_KWARGS_KEY__,
)
from mctm.models import DensityRegressionModel

# Set random seed for reproducibility
tf.random.set_seed(1)

# Define toy data parameters
NUM_SAMPLES = 1000
DATA_DIMS = 3


@pytest.fixture(params=[1, 16, 32])
def batch_size(request):
    """Fixture yielding different batch sizes."""
    return request.param


@pytest.fixture(
    params=[
        (
            "multivariate_normal",
            {
                "dims": DATA_DIMS,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "conditional": False,
                },
            },
        ),
        (
            "multivariate_normal",
            {
                "dims": DATA_DIMS,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "activation": "relu",
                    "hidden_units": [16] * 2,
                    "batch_norm": False,
                    "dropout": 0,
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
            },
        ),
        (
            "coupling_flow",
            {
                __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                __BIJECTOR_KWARGS_KEY__: {
                    "domain": [0, 1],
                    "extrapolation": False,
                },
                __INVERT_BIJECTOR_KEY__: True,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "activation": "relu",
                    "hidden_units": [16] * 2,
                    "batch_norm": False,
                    "dropout": 0,
                    "conditional": False,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "allow_flexible_bounds": False,
                    "bounds": "linear",
                    "high": 1,
                    "low": 0,
                },
                "dims": DATA_DIMS,
                "num_layers": 3,
                "num_parameters": 16,
            },
        ),
        (
            "coupling_flow",
            {
                __BIJECTOR_NAME_KEY__: "RationalQuadraticSpline",
                __BIJECTOR_KWARGS_KEY__: {"range_min": -5},
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "activation": "relu",
                    "hidden_units": [32] * 2,
                    "batch_norm": False,
                    "dropout": 0,
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "interval_width": 10,
                    "min_slope": 0.001,
                    "min_bin_width": 0.001,
                    "nbins": 32,
                },
                "dims": DATA_DIMS,
                "num_layers": 2,
                "num_parameters": 32 * 3 - 1,
                "num_masked": 2,
            },
        ),
        (
            "masked_autoregressive_flow",
            {
                "dims": DATA_DIMS,
                "num_layers": 2,
                "num_parameters": 8,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "hidden_units": [16] * 4,
                    "activation": "relu",
                    "conditional": False,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "allow_flexible_bounds": False,
                    "bounds": "linear",
                    "high": -4,
                    "low": 4,
                },
                __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
                "invert": True,
            },
        ),
        (
            "masked_autoregressive_flow",
            {
                "dims": DATA_DIMS,
                "num_layers": 2,
                "num_parameters": 8,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "hidden_units": [16] * 4,
                    "activation": "relu",
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "allow_flexible_bounds": False,
                    "bounds": "linear",
                    "high": -4,
                    "low": 4,
                },
                __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
                "invert": True,
            },
        ),
        (
            "masked_autoregressive_flow",
            {
                "dims": DATA_DIMS,
                "num_layers": 3,
                "num_parameters": 32 * 3 - 1,
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "hidden_units": [32] * 2,
                    "activation": "relu",
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "interval_width": 10,
                    "min_slope": 0.001,
                    "min_bin_width": 0.001,
                    "nbins": 32,
                },
                __BIJECTOR_NAME_KEY__: "RationalQuadraticSpline",
                __BIJECTOR_KWARGS_KEY__: {"range_min": -5},
            },
        ),
        (
            "masked_autoregressive_flow_first_dim_masked",
            {
                "dims": DATA_DIMS,
                "num_layers": 4,
                "num_parameters": 8,
                "x0_parameters_fn_kwargs": {
                    "activation": "relu",
                    "hidden_units": [16] * 2,
                    "batch_norm": False,
                    "dropout": 0,
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
                "maf_parameters_fn_kwargs": {
                    "hidden_units": [16] * 4,
                    "activation": "relu",
                    "conditional": False,
                },
                __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                    "allow_flexible_bounds": False,
                    "bounds": "linear",
                    "high": -4,
                    "low": 4,
                },
                __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
                "invert": True,
            },
        ),
    ]
)
def model_config(request):
    """Fixture yielding different density regression model configurations."""
    return request.param


@pytest.fixture
def data():
    """Generate toy data."""
    tf.random.set_seed(1)
    eps = tf.random.normal((NUM_SAMPLES, DATA_DIMS))
    x = tf.random.uniform((NUM_SAMPLES, DATA_DIMS), -1, 1)
    y = x**2 + 2 * x + eps * 0.1 * x
    return x, y


def test_density_regression_model(data, model_config, batch_size, tmpdir):
    """Test Density Regression Model."""
    compile_kwargs = dict(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

    # Unpack data
    x, y = data

    # Unpack model configuration
    (
        distribution_name,
        distribution_kwargs,
    ) = model_config

    # Get Parameter kwargs
    if distribution_name == "masked_autoregressive_flow_first_dim_masked":
        parameter_kwargs = distribution_kwargs["x0_" + __PARAMETERS_FN_KWARGS_KEY__]
    else:
        parameter_kwargs = distribution_kwargs[__PARAMETERS_FN_KWARGS_KEY__]

    # Initialize model
    model = DensityRegressionModel(
        distribution=distribution_name,
        **distribution_kwargs,
    )

    # Compile and train model
    model.compile(**compile_kwargs)
    model.fit(x=x, y=y, epochs=1, verbose=False)

    # Save model weights
    weights_file = os.path.join(tmpdir, "weights.h5")
    model.save_weights(weights_file)

    # Create a new model with the same configuration
    new_model = DensityRegressionModel(
        distribution=distribution_name,
        **distribution_kwargs,
    )
    new_model.compile(**compile_kwargs)
    new_model.build([None, DATA_DIMS])

    # Load saved weights
    new_model.load_weights(weights_file)

    # Check equality of trainable variables
    for var1, var2 in zip(model.trainable_variables, new_model.trainable_variables):
        assert tf.reduce_all(var1 == var2)

    # Check equality of non-trainable variables
    for var1, var2 in zip(
        model.non_trainable_variables, new_model.non_trainable_variables
    ):
        assert tf.reduce_all(var1 == var2)

    with tf.GradientTape(persistent=True) as tape:
        # Test sampling
        if parameter_kwargs.get("conditional", False):
            conditional_input = tf.ones(
                (batch_size, parameter_kwargs["conditional_event_shape"])
            )
            dist = model(conditional_input)
            assert dist.batch_shape == [batch_size]
            samples = dist.sample(10)
            assert samples.shape == (10, batch_size, DATA_DIMS)
        else:
            dist = model(None)
            assert dist.batch_shape == []
            samples = dist.sample(10)
            assert samples.shape == (10, DATA_DIMS)

        assert dist.event_shape == [DATA_DIMS]

        test_input = tf.ones((batch_size, DATA_DIMS))
        loss = compile_kwargs["loss"](test_input, dist)
        grads = tape.gradient(loss, model.trainable_weights)
        for g, v in zip(grads, model.trainable_weights):
            assert g.shape == v.shape
