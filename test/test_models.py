"""Unit test for density regression models."""

import os

import pytest
import tensorflow as tf

from mctm.distributions import (
    __BIJECTOR_KWARGS_KEY__,
    __BIJECTOR_NAME_KEY__,
    __INVERT_BIJECTOR_KEY__,
    __PARAMETER_SLICE_SIZE_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
    __PARAMETERS_FN_KWARGS_KEY__,
)
from mctm.models import DensityRegressionModel, HybridDensityRegressionModel

# Set random seed for reproducibility
tf.random.set_seed(1)

# Define toy data parameters
NUM_SAMPLES = 1024
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
                    "conditional_event_shape": DATA_DIMS,
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
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
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
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "activation": "relu",
                    "hidden_units": [16] * 2,
                    "batch_norm": False,
                    "dropout": 0,
                    "conditional": False,
                },
                "dims": DATA_DIMS,
                "num_layers": 3,
                "num_parameters": 16,
                "nested_bijectors": [
                    {
                        __BIJECTOR_NAME_KEY__: "Scale",
                        __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                        __PARAMETER_SLICE_SIZE_KEY__: 1,
                    },
                    {
                        __BIJECTOR_NAME_KEY__: "Shift",
                        __PARAMETER_SLICE_SIZE_KEY__: 1,
                    },
                    {
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                            "allow_flexible_bounds": False,
                            "bounds": "linear",
                            "high": 1,
                            "low": 0,
                        },
                        __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                        __BIJECTOR_KWARGS_KEY__: {
                            "domain": [0, 1],
                            "extrapolation": False,
                        },
                        __INVERT_BIJECTOR_KEY__: True,
                        __PARAMETER_SLICE_SIZE_KEY__: 14,
                    },
                ],
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
                __INVERT_BIJECTOR_KEY__: True,
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
                __INVERT_BIJECTOR_KEY__: True,
            },
        ),
        (
            "masked_autoregressive_flow",
            {
                "dims": DATA_DIMS,
                "num_layers": 2,
                "num_parameters": 8,
                "nested_bijectors": [
                    {
                        __BIJECTOR_NAME_KEY__: "Scale",
                        __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                        __PARAMETER_SLICE_SIZE_KEY__: 1,
                    },
                    {__BIJECTOR_NAME_KEY__: "Shift", __PARAMETER_SLICE_SIZE_KEY__: 1},
                    {
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
                            "allow_flexible_bounds": False,
                            "bounds": "linear",
                            "high": -4,
                            "low": 4,
                        },
                        __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
                        __BIJECTOR_KWARGS_KEY__: {
                            "domain": [0, 1],
                            "extrapolation": False,
                        },
                        __INVERT_BIJECTOR_KEY__: True,
                        __PARAMETER_SLICE_SIZE_KEY__: 6,
                    },
                ],
                __PARAMETERS_FN_KWARGS_KEY__: {
                    "hidden_units": [16] * 4,
                    "activation": "relu",
                    "conditional": True,
                    "conditional_event_shape": DATA_DIMS,
                },
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
                __INVERT_BIJECTOR_KEY__: True,
            },
        ),
    ]
)
def density_regression_model_config(request):
    """Fixture yielding different density regression model configurations."""
    return request.param


@pytest.fixture(
    params=[
        {
            "marginal_bijectors": [
                {
                    "bijector": "BernsteinPolynomial",
                    "bijector_kwargs": {
                        "domain": (-4, 4),
                        "extrapolation": True,
                    },
                    "parameters_fn": "parameter_vector",
                    "parameters_fn_kwargs": {
                        "parameter_shape": [DATA_DIMS, 10],
                        "dtype": "float32",
                    },
                    "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",  # noqa: E501
                    "parameters_constraint_fn_kwargs": {
                        "low": -4,
                        "high": 4,
                        "bounds": "smooth",
                        "allow_flexible_bounds": True,
                    },
                },
                {
                    "bijector": "Shift",
                    "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
                    "parameters_fn_kwargs": {
                        "parameter_shape": [DATA_DIMS],
                        "dtype": "float",
                        "polynomial_order": 6,
                        "conditional_event_shape": DATA_DIMS,
                        "extrapolation": True,
                    },
                },
            ],
            "joint_bijectors": [
                {
                    "bijector": "ScaleMatvecLinearOperator",
                    "parameters_fn": "bernstein_polynomial",
                    "parameters_fn_kwargs": {
                        "parameter_shape": [sum(range(DATA_DIMS))],
                        "dtype": "float",
                        "polynomial_order": 6,
                        "conditional_event_shape": DATA_DIMS,
                        "domain": (-1, 1),
                        "extrapolation": True,
                        "initializer": tf.ones,
                    },
                    "parameters_constraint_fn": "mctm.activations.lambda_parameters_constraint_fn",  # noqa: E501
                }
            ],
        }
    ]
)
def hybrid_density_regression_model_config(request):
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


def test_density_regression_model(
    data, density_regression_model_config, batch_size, tmpdir
):
    """Test Density Regression Model."""
    compile_kwargs = dict(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

    # Unpack data
    x, y = data

    # Unpack model configuration
    (
        distribution_name,
        distribution_kwargs,
    ) = density_regression_model_config

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


def test_hybrid_density_regression_model(
    data, hybrid_density_regression_model_config, batch_size, tmpdir
):
    """Test Density Regression Model."""
    compile_kwargs = dict(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

    # Unpack data
    x, y = data

    num_marginal_bijectors = len(
        hybrid_density_regression_model_config["marginal_bijectors"]
    )
    num_joint_bijectors = len(hybrid_density_regression_model_config["joint_bijectors"])

    # Check parameter handling
    for marginals_trainable, joint_trainable in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        # Initialize model
        model = HybridDensityRegressionModel(
            marginals_trainable=marginals_trainable,
            joint_trainable=joint_trainable,
            dims=DATA_DIMS,
            **hybrid_density_regression_model_config,
        )

        assert len(model.trainable_variables) == sum(
            (
                num_marginal_bijectors if marginals_trainable else 0,
                num_joint_bijectors if joint_trainable else 0,
            )
        )
        assert len(model.non_trainable_variables) == sum(
            (
                0 if marginals_trainable else num_marginal_bijectors,
                0 if joint_trainable else num_joint_bijectors,
            )
        )

        # Compile and train model
        model.compile(**compile_kwargs)
        model.fit(x=x, y=y, epochs=1, verbose=False)

        # Save model weights
        weights_file = os.path.join(tmpdir, "weights.h5")
        model.save_weights(weights_file)

        # Create a new model with the same configuration
        new_model = HybridDensityRegressionModel(
            marginals_trainable=True,
            joint_trainable=True,
            dims=DATA_DIMS,
            **hybrid_density_regression_model_config,
        )
        # Use property assignment
        new_model.marginals_trainable = marginals_trainable
        new_model.joint_trainable = joint_trainable
        new_model.compile(**compile_kwargs)

        # Call model once to get weights ready
        new_model(x)

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
            conditional_input = tf.ones((batch_size, DATA_DIMS))
            dist = model(conditional_input)
            assert dist.batch_shape == [batch_size]
            samples = dist.sample(10)
            assert samples.shape == (10, batch_size, DATA_DIMS)
            assert dist.event_shape == [DATA_DIMS]

            test_input = tf.ones((batch_size, DATA_DIMS))
            loss = compile_kwargs["loss"](test_input, dist)
            grads = tape.gradient(loss, model.trainable_weights)
            if not (model.marginals_trainable or model.joint_trainable):
                assert len(grads) == 0
            for g, v in zip(grads, model.trainable_weights):
                assert g.shape == v.shape
