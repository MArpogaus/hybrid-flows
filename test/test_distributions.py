# %% Description ###############################################################
"""Unit test for normalizing flow module."""

# %% imports ###################################################################
import logging
from typing import Callable

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinPolynomial
from hybrid_flows.distributions import (
    __ALL_KEYS__,
    __BIJECTOR_KWARGS_KEY__,
    __BIJECTOR_NAME_KEY__,
    __INVERT_BIJECTOR_KEY__,
    __NESTED_BIJECTOR_KEY__,
    __PARAMETER_SLICE_SIZE_KEY__,
    __PARAMETERIZED_BY_PARENT_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
    __PARAMETERS_FN_KEY__,
    __PARAMETERS_FN_KWARGS_KEY__,
    __PARAMETERS_KEY__,
    __TRAINABLE_KEY__,
    _get_eval_parameter_fn,
    _get_layer_overwrites,
    _get_num_masked,
    _init_bijector_from_dict,
    _init_parameters_fn,
    get_coupling_flow,
    get_masked_autoregressive_flow,
    get_masked_autoregressive_flow_first_dim_masked,
    get_normalizing_flow,
)
from hybrid_flows.parameters import (
    get_test_parameters_fn,
    get_test_parameters_nested_fn,
)
from hybrid_flows.utils.decorators import recurse_on_key
from tensorflow_probability import bijectors as tfb

logging.basicConfig(level=logging.DEBUG)


# %% fixtures ##################################################################
@pytest.fixture(params=[3, 48])
def dims(request):
    """Fixture returning different data dimensions."""
    return request.param


@pytest.fixture
def mock_bijector_data(dims):
    """Mock data for bijector configurations."""
    return [
        {
            __BIJECTOR_NAME_KEY__: "Scale",
            __BIJECTOR_KWARGS_KEY__: {"scale": 2.0},
        },
        {
            __BIJECTOR_NAME_KEY__: "Shift",
            __PARAMETERS_KEY__: 1.0,
        },
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: 3.0,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1],
                "param_shape": [dims - 1, 2],
            },
        },
        tfb.Permute(permutation=list(reversed(range(dims)))),
    ]


@pytest.fixture
def mock_deeply_nested_bijectors(dims):
    """Mock data for deeply nested bijector configurations."""
    return [
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __BIJECTOR_KWARGS_KEY__: {"scale": 2.0},
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERS_KEY__: 1.0,
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: lambda x: x[..., 0],
                },
                {
                    __BIJECTOR_NAME_KEY__: "RealNVP",
                    __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
                    __NESTED_BIJECTOR_KEY__: [
                        {
                            __BIJECTOR_NAME_KEY__: "Scale",
                            __PARAMETERIZED_BY_PARENT_KEY__: True,
                            __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                            __PARAMETER_SLICE_SIZE_KEY__: 1,
                        },
                        {
                            __BIJECTOR_NAME_KEY__: "Shift",
                            __PARAMETERIZED_BY_PARENT_KEY__: True,
                            __PARAMETER_SLICE_SIZE_KEY__: 1,
                            __PARAMETERS_KEY__: 3.0,
                        },
                    ],
                    __PARAMETERS_FN_KEY__: "test_parameters_nested",
                    __PARAMETERS_FN_KWARGS_KEY__: {
                        "input_shape": [1],
                        "param_shape": [dims - 2, 2] if dims > 3 else [2],
                    },
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1, 1],  # [B, nm]
                "param_shape": [1],
            },
        },
    ]


@pytest.fixture
def mock_maf_bijectors(dims):
    """Mock data for MAF bijector configurations."""
    return [
        {
            __BIJECTOR_NAME_KEY__: "MaskedAutoregressiveFlow",
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: tf.abs,
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: -3.0,
                    __TRAINABLE_KEY__: False,
                },
                tfb.Sigmoid(),
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [dims],
                "param_shape": [dims, 2],
            },
        },
    ]


@pytest.fixture
def mock_realnvp_bijectors(dims):
    """Mock data for RealNVP bijector configurations."""
    permutation = list(reversed(range(dims)))
    return [
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {},
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: 0.0,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1],
                "param_shape": [dims - 1, 2],
            },
            __PARAMETERS_CONSTRAINT_FN_KEY__: lambda x: x,
        },
        tfb.Permute(permutation=permutation),
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {},
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: 1.0,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1],
                "param_shape": [dims - 1, 2],
            },
            __PARAMETERS_CONSTRAINT_FN_KEY__: lambda x: x / 2,
        },
        tfb.Permute(permutation=permutation),
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {},
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: 2.0,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1],
                "param_shape": [dims - 1, 2],
            },
            __PARAMETERS_CONSTRAINT_FN_KEY__: lambda x: x / 3,
        },
        tfb.Permute(permutation=permutation),
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_CONSTRAINT_FN_KEY__: "tf.math.softplus",
                    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {},
                },
                {
                    __BIJECTOR_NAME_KEY__: "Shift",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
                    __PARAMETER_SLICE_SIZE_KEY__: 1,
                    __PARAMETERS_KEY__: 3.0,
                },
            ],
            __PARAMETERS_FN_KEY__: "test_parameters_nested",
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": [1],
                "param_shape": [dims - 1, 2],
            },
            __PARAMETERS_CONSTRAINT_FN_KEY__: lambda x: x / 4,
        },
        tfb.Permute(permutation=permutation),
    ]


@pytest.fixture(
    params=[
        {
            "activation": "relu",
            "batch_norm": False,
            "dropout": 0,
            "hidden_units": [16] * 4,
        },
        {
            "activation": "relu",
            "batch_norm": True,
            "dropout": 0,
            "hidden_units": [32] * 2,
            "conditional": True,
            "conditional_event_shape": 10,
        },
    ]
)
def feed_forward_kwargs(request):
    """Yield different kwargs for feed forward networks."""
    return request.param


@pytest.fixture(params=[3, 4])
def coupling_bernstein_flow_kwargs(request, feed_forward_kwargs):
    """Mock kwargs for coupling Bernstein flow."""
    return {
        __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
        __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
        __INVERT_BIJECTOR_KEY__: True,
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
            "allow_flexible_bounds": False,
            "bounds": "linear",
            "high": 1,
            "low": 0,
        },
        __PARAMETERS_FN_KWARGS_KEY__: feed_forward_kwargs,
        "dims": 5,
        "layer_overwrites": {
            -2: {"parameters_constraint_fn_kwargs": {"high": 5, "low": -5}},
            -1: {"parameters_constraint_fn_kwargs": {"high": 5, "low": -5}},
        },
        "num_layers": request.param,
        "num_parameters": 16,
    }


@pytest.fixture(
    params=[
        {
            "activation": "relu",
            "hidden_units": [16] * 2,
            "conditional": True,
            "conditional_event_shape": 10,
        },
        {
            "activation": "relu",
            "hidden_units": [16] * 4,
            "conditional": False,
        },
    ]
)
def made_kwargs(request):
    """Yield different kwargs for made networks."""
    return request.param


@pytest.fixture(params=[5, 8])
def coupling_spline_flow_kwargs(request, feed_forward_kwargs):
    """Mock kwargs for coupling Bernstein flow."""
    return {
        __BIJECTOR_NAME_KEY__: "RationalQuadraticSpline",
        __BIJECTOR_KWARGS_KEY__: {"range_min": -5},
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
            "interval_width": 10,
            "min_slope": 0.001,
            "min_bin_width": 0.001,
            "nbins": 32,
        },
        __PARAMETERS_FN_KWARGS_KEY__: feed_forward_kwargs,
        "dims": 2,
        "num_layers": request.param,
        "num_parameters": 32 * 3 - 1,
        "num_masked": 1,
    }


@pytest.fixture(params=[5, 7])
def masked_autoregressive_bernstein_flow_kwargs(request, made_kwargs):
    """Mock kwargs for coupling Bernstein flow."""
    return {
        __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
        __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
        __INVERT_BIJECTOR_KEY__: True,
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
            "allow_flexible_bounds": False,
            "bounds": "linear",
            "high": -4,
            "low": 4,
        },
        __PARAMETERS_FN_KWARGS_KEY__: made_kwargs,
        "dims": 7,
        "num_layers": 3,
        "num_parameters": request.param,
    }


@pytest.fixture(params=[2, 3])
def masked_autoregressive_spline_flow_kwargs(request, made_kwargs):
    """Mock kwargs for coupling Bernstein flow."""
    return {
        __BIJECTOR_NAME_KEY__: "RationalQuadraticSpline",
        __BIJECTOR_KWARGS_KEY__: {"range_min": -5},
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
            "interval_width": 10,
            "min_slope": 0.001,
            "min_bin_width": 0.001,
            "nbins": 32,
        },
        __PARAMETERS_FN_KWARGS_KEY__: made_kwargs,
        "dims": request.param,
        "num_layers": 3,
        "num_parameters": 32 * 3 - 1,
    }


@pytest.fixture(params=[1, 3, 4])
def masked_autoregressive_flow_first_dim_masked_kwargs(
    request, feed_forward_kwargs, made_kwargs
):
    """Mock kwargs for coupling Bernstein flow."""
    return {
        __BIJECTOR_NAME_KEY__: "BernsteinPolynomial",
        __BIJECTOR_KWARGS_KEY__: {"domain": [0, 1], "extrapolation": False},
        __INVERT_BIJECTOR_KEY__: True,
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__: {
            "allow_flexible_bounds": False,
            "bounds": "linear",
            "high": -3,
            "low": 3,
        },
        "dims": 9,
        "num_layers": request.param,
        "num_parameters": 8,
        "x0_parameters_fn_kwargs": feed_forward_kwargs,
        "maf_parameters_fn_kwargs": made_kwargs,
    }


@pytest.fixture(params=[8, 16, 32])
def batch_size(request):
    """Fixture yielding different batch sizes."""
    return request.param


@pytest.fixture(params=[1, 7, 32])
def num_samples(request):
    """Fixture yielding different batch sizes."""
    return request.param


@pytest.fixture
def mock_parameters_fn():
    """Mock parameter function."""
    return get_test_parameters_fn


@pytest.fixture
def mock_parameters_nested_fn():
    """Mock parameter function."""
    return get_test_parameters_nested_fn


@pytest.fixture(
    params=[
        {
            "get_base_distribution": (
                lambda dims: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(dims)
                )
            )
        },
        {"distribution_name": "logistic"},
        {},
    ]
)
def mock_base_distribution_data(request):
    """Mock data for base distribution."""
    return request.param


# %% functions #################################################################
def get_num_made_variables(masked_autoregressive_flow_kwargs):
    """Get number of trainable variables of all made networks of MAF."""
    return (
        (
            len(
                masked_autoregressive_flow_kwargs["parameters_fn_kwargs"][
                    "hidden_units"
                ]
            )
            + 1
        )
        * masked_autoregressive_flow_kwargs["num_layers"]
        * 2
    )


def assert_all_parameter_have_unique_id(all_parameters):
    """Check that all parameters or parameter functions are unique."""

    @recurse_on_key(__NESTED_BIJECTOR_KEY__)
    def process(d):
        if isinstance(d, dict):
            if d[__PARAMETERS_KEY__] is not None:
                return d[__PARAMETERS_KEY__], id(d[__PARAMETERS_KEY__])
            else:
                return None
        else:
            return d, id(d)

    ids = list(i for i in map(process, all_parameters) if i is not None)
    assert len(ids) == len(set(i[1] for i in ids))


def assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims):
    """Check if samples have the expected shape."""
    transformed_sample = dist.sample(num_samples)
    base_sample = dist.bijector.inverse(transformed_sample)
    reconstructed_sample = dist.bijector.forward(base_sample)

    if is_conditional:
        expected_shape = (num_samples, batch_size, dims)
    else:
        expected_shape = (num_samples, dims)
    assert transformed_sample.shape == expected_shape
    assert base_sample.shape == expected_shape
    assert reconstructed_sample.shape == expected_shape
    assert tf.reduce_all(tf.abs(reconstructed_sample - transformed_sample) < 1e-6)


def check_maf_params_fn(
    params_dict,
    layer,
    batch_size,
    dims,
    masked_autoregressive_bernstein_flow_kwargs,
    is_conditional,
):
    """Check if the maf parameters fn looks as expected."""
    assert callable(params_dict[__PARAMETERS_KEY__])

    test_input2 = layer * tf.ones((batch_size, dims))
    output_shape = [
        batch_size,
        dims,
        masked_autoregressive_bernstein_flow_kwargs["num_parameters"],
    ]

    parameters_fn = params_dict[__PARAMETERS_KEY__]
    assert parameters_fn.__name__ == "dec_fn"
    made_fn = parameters_fn.__closure__[0].cell_contents
    expected_name = f"maf_layer_{layer}"
    if is_conditional:
        assert made_fn.__closure__[2].cell_contents.name == expected_name
    else:
        assert made_fn.__closure__[1].cell_contents.name == expected_name
    assert parameters_fn(test_input2).shape == output_shape
    for k in params_dict.keys():
        assert k in __ALL_KEYS__


def assert_flow_shapes(batch_size, dims, flow):
    """Check shapes of transformed samples."""
    y1 = tf.ones((batch_size, dims))
    y2 = tf.ones((10, batch_size, dims))
    z1 = flow.forward(y1)
    z2 = flow.forward(y2)
    y11 = flow.inverse(tf.identity(z1))
    y22 = flow.inverse(tf.identity(z2))
    assert tf.reduce_all(y1.shape == z1.shape)
    assert tf.reduce_all(y2.shape == z2.shape)
    assert tf.reduce_all(y11.shape == z1.shape)
    assert tf.reduce_all(y22.shape == z2.shape)


def assert_bijectors_shapes(flow, test_input1, dims):
    """Check bijectors type and parameters."""
    scale_bijector = flow[0]
    assert isinstance(scale_bijector, tfb.Scale)
    assert scale_bijector.scale == 2.0

    shift_bijector = flow[1]
    assert isinstance(shift_bijector, tfb.Shift)
    assert shift_bijector.shift == 1.0

    realnvp_bijector = flow[2]
    assert isinstance(realnvp_bijector, tfb.RealNVP)
    assert realnvp_bijector._num_masked == 1
    assert isinstance(realnvp_bijector._bijector_fn, Callable)

    test_input2 = tf.ones([5, 1])
    test_result = test_input1 * test_input2
    netsed_flow = realnvp_bijector._bijector_fn(test_input2)
    assert isinstance(netsed_flow, tfb.Chain)
    assert len(netsed_flow.bijectors) == 2

    assert isinstance(netsed_flow.bijectors[0], tfb.Scale)
    assert tf.reduce_all(
        netsed_flow.bijectors[0].scale == tf.math.softplus(test_result)
    )
    assert isinstance(netsed_flow.bijectors[1], tfb.Shift)
    assert tf.reduce_all(netsed_flow.bijectors[1].shift == 3 + test_result)

    permute_bijector = flow[3]
    assert isinstance(permute_bijector, tfb.Permute)
    assert tf.reduce_all(permute_bijector.permutation == list(reversed(range(dims))))


# %% tests #####################################################################
def test_init_parameters_fn_constant(mock_bijector_data):
    """Test _init_parameters_fn with constant parameters."""
    bijector_config = {
        __PARAMETERS_KEY__: 42,
        __BIJECTOR_NAME_KEY__: "dummy",
    }
    parameters_fn, trainable_variables, non_trainable_variables = _init_parameters_fn(
        [bijector_config], variables_name="test"
    )
    assert parameters_fn is not None
    assert parameters_fn[0][__PARAMETERS_KEY__]() == 42


def test_init_parameters_fn_callable(mock_parameters_fn):
    """Test _init_parameters_fn with callable parameters."""
    bijector_config = {
        __PARAMETERS_FN_KEY__: mock_parameters_fn,
        __PARAMETERS_FN_KWARGS_KEY__: {"input_shape": [1], "param_shape": [10]},
        __BIJECTOR_NAME_KEY__: "dummy",
    }
    all_parameters_fn, trainable_variables, non_trainable_variables = (
        _init_parameters_fn([bijector_config], variables_name=None)
    )
    parameters_fn = all_parameters_fn[0][__PARAMETERS_KEY__]
    assert parameters_fn is not None
    test_input = tf.ones([2, 1])
    assert tf.reduce_all(parameters_fn(test_input) == tf.ones([2, 10]))


def test_init_parameters_fn_string(mock_bijector_data):
    """Test _init_parameters_fn with string parameters."""
    bijector_config = {
        __PARAMETERS_FN_KEY__: "test_parameters",
        __PARAMETERS_FN_KWARGS_KEY__: {"input_shape": [1], "param_shape": [5]},
        __BIJECTOR_NAME_KEY__: "dummy",
    }
    all_parameters_fn, trainable_variables, non_trainable_variables = (
        _init_parameters_fn([bijector_config], variables_name="test")
    )
    parameters_fn = all_parameters_fn[0][__PARAMETERS_KEY__]
    assert parameters_fn is not None
    test_input = tf.ones([3, 1])
    assert tf.reduce_all(parameters_fn(test_input) == tf.ones([3, 5]))


def test_init_bijector_from_dict(mock_bijector_data, dims):
    """Test _init_bijector_from_dict with simple bijectors and nested bijectors."""
    bijectors_parameters_fns, trainable_variables, non_trainable_variables = (
        _init_parameters_fn(mock_bijector_data, variables_name=None)
    )
    test_input1 = 10.0
    eval_parameter_fn = _get_eval_parameter_fn(test_input1)
    all_parameters = list(map(eval_parameter_fn, bijectors_parameters_fns))
    assert_all_parameter_have_unique_id(all_parameters)
    flow = list(map(_init_bijector_from_dict, all_parameters))
    assert_bijectors_shapes(flow, test_input1, dims)


def test_get_normalizing_flow(
    num_samples,
    mock_bijector_data,
    mock_parameters_fn,
    mock_base_distribution_data,
    dims,
):
    """Test get_normalizing_flow."""
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_normalizing_flow(
        bijectors=mock_bijector_data,
        dims=dims,
        inverse_flow=True,
        reverse_flow=True,
        **mock_base_distribution_data,
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)
    assert len(trainable_variables) == 2
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v, e in zip(trainable_variables, (1.0, 3.0)):
        assert isinstance(v, tf.Variable)
        assert v.trainable
        assert v == e
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0
    assert len(set(map(id, non_trainable_variables))) == len(non_trainable_variables)

    test_input1 = 4
    test_input2 = 8 * tf.range(20.0)[..., None]
    output_shape = sum(mock_bijector_data[2][__PARAMETERS_FN_KWARGS_KEY__].values(), [])
    test_result = test_input1 * (tf.ones(output_shape) * test_input2[..., None])
    input_shape, all_parameters = parameter_fn(test_input1)
    assert input_shape is None
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert len(all_parameters) == len(mock_bijector_data)
    assert all_parameters[0][__PARAMETERS_KEY__] is None
    assert all_parameters[1][__PARAMETERS_KEY__] == 1.0
    assert callable(all_parameters[2][__PARAMETERS_KEY__])
    assert tf.reduce_all(
        all_parameters[2][__PARAMETERS_KEY__](test_input2) == test_result
    )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.invert._Invert)
    assert isinstance(flow.bijector, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(32, dims, flow)

    bijectors = dist.bijector.bijector.bijectors
    assert_bijectors_shapes(list(reversed(bijectors)), test_input1, dims)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, False, None, dims)


def test_get_normalizing_flow_deeply_nested(
    num_samples,
    mock_deeply_nested_bijectors,
    mock_base_distribution_data,
    dims,
    batch_size,
):
    """Test get_normalizing_flow with deeply nested bijectors."""
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_normalizing_flow(
        bijectors=mock_deeply_nested_bijectors,
        inverse_flow=True,
        reverse_flow=False,
        dims=dims,
        **mock_base_distribution_data,
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)
    assert len(trainable_variables) == 2
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v, e in zip(trainable_variables, (1.0, 3.0)):
        assert isinstance(v, tf.Variable)
        assert v.trainable
        assert v == e
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0
    assert len(set(map(id, non_trainable_variables))) == len(non_trainable_variables)

    test_input1 = 2
    test_input = tf.reshape(
        tf.range(batch_size * dims, dtype=tf.float32), [batch_size, dims]
    )
    test_input2 = test_input[:, :1]
    test_input3 = test_input[:, 2:]
    output_shape = sum(
        mock_deeply_nested_bijectors[0][__PARAMETERS_FN_KWARGS_KEY__].values(), []
    )
    test_result = test_input1 * (tf.ones(output_shape) * test_input2[..., None])
    input_shape, all_parameters = parameter_fn(test_input1)
    assert input_shape is None
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert len(all_parameters) == len(mock_deeply_nested_bijectors)
    assert callable(all_parameters[0][__PARAMETERS_KEY__])
    assert tf.reduce_all(
        all_parameters[0][__PARAMETERS_KEY__](test_input2) == test_result
    )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.invert._Invert)

    assert_flow_shapes(batch_size, dims, flow)

    realnvp_bijector = flow.bijector
    assert isinstance(realnvp_bijector, tfb.RealNVP)
    assert realnvp_bijector._num_masked == 1
    assert isinstance(realnvp_bijector._bijector_fn, Callable)

    nested_flow_1 = realnvp_bijector._bijector_fn(test_input2)
    assert isinstance(nested_flow_1, tfp.python.bijectors.chain._Chain)
    assert len(nested_flow_1.bijectors) == len(
        mock_deeply_nested_bijectors[0][__NESTED_BIJECTOR_KEY__]
    )

    scale_bijector = nested_flow_1.bijectors[0]
    assert isinstance(scale_bijector, tfb.Scale)
    assert scale_bijector.scale == 2.0

    shift_bijector = nested_flow_1.bijectors[1]
    assert isinstance(shift_bijector, tfb.Shift)
    assert tf.reduce_all(shift_bijector.shift == test_result[..., 0] + 1)

    realnvp_bijector = nested_flow_1.bijectors[2]
    assert isinstance(realnvp_bijector, tfb.RealNVP)
    assert realnvp_bijector._num_masked == 1
    assert isinstance(realnvp_bijector._bijector_fn, Callable)

    output_shape = sum(
        mock_deeply_nested_bijectors[0][__NESTED_BIJECTOR_KEY__][-1][
            __PARAMETERS_FN_KWARGS_KEY__
        ].values(),
        [],
    )
    test_result2 = (
        test_input1 * (tf.ones(output_shape) * test_input3[..., None]) + test_result
    )

    nested_flow_2 = realnvp_bijector._bijector_fn(test_input3)
    assert isinstance(nested_flow_2, tfb.Chain)
    assert len(nested_flow_2.bijectors) == 2

    assert isinstance(nested_flow_2.bijectors[0], tfb.Scale)
    assert tf.reduce_all(
        nested_flow_2.bijectors[0].scale == tf.math.softplus(test_result2[..., 0])
    )
    assert isinstance(nested_flow_2.bijectors[1], tfb.Shift)
    assert tf.reduce_all(nested_flow_2.bijectors[1].shift == 3 + test_result2[..., 1])

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, False, batch_size, dims)


def test_get_normalizing_flow_maf(mock_maf_bijectors, dims):
    """Test get_normalizing_flow with MAF bijectors."""
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_normalizing_flow(
        bijectors=mock_maf_bijectors, inverse_flow=False, dims=dims
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)
    assert len(trainable_variables) == 0
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 1
    assert len(set(map(id, non_trainable_variables))) == len(non_trainable_variables)
    assert isinstance(non_trainable_variables[0], tf.Variable)
    assert non_trainable_variables[0].trainable
    assert (
        non_trainable_variables[0]
        == mock_maf_bijectors[0][__NESTED_BIJECTOR_KEY__][1][__PARAMETERS_KEY__]
    )

    test_input1 = -5
    test_input2 = tf.ones([dims, 1]) * 4
    output_shape = sum(mock_maf_bijectors[0][__PARAMETERS_FN_KWARGS_KEY__].values(), [])
    test_result = test_input1 * (tf.ones(output_shape) * test_input2[..., None])
    input_shape, all_parameters = parameter_fn(test_input1)
    assert input_shape is None
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert len(all_parameters) == len(mock_maf_bijectors)
    assert tf.reduce_all(
        all_parameters[0][__PARAMETERS_KEY__](test_input2) == test_result
    )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == []
    assert dist.event_shape == [dims]

    flow = dist.bijector
    assert isinstance(flow, tfb.MaskedAutoregressiveFlow)
    assert isinstance(flow._bijector_fn, Callable)

    # parameters fn invalid -> Not working here
    # assert_flow_shapes(1, dims - 1, flow)

    nested_flow = flow._bijector_fn(test_input2)
    assert isinstance(nested_flow, tfp.python.bijectors.chain._Chain)
    assert len(nested_flow.bijectors) == len(
        mock_maf_bijectors[0][__NESTED_BIJECTOR_KEY__]
    )

    scale_bijector = nested_flow.bijectors[0]
    assert isinstance(scale_bijector, tfb.Scale)
    assert tf.reduce_all(scale_bijector.scale == tf.abs(test_result[..., 0]))

    shift_bijector = nested_flow.bijectors[1]
    assert isinstance(shift_bijector, tfb.Shift)
    assert tf.reduce_all(
        shift_bijector.shift
        == test_result[..., 1]
        + mock_maf_bijectors[0][__NESTED_BIJECTOR_KEY__][1][__PARAMETERS_KEY__]
    )

    exp_bijector = nested_flow.bijectors[2]
    assert isinstance(exp_bijector, tfb.Sigmoid)

    # sampling is not working since we dont use a valid parameter fn here


def test_get_normalizing_flow_realnvp(num_samples, mock_realnvp_bijectors, dims):
    """Test get_normalizing_flow with RealNVP bijectors."""
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_normalizing_flow(
        bijectors=mock_realnvp_bijectors,
        inverse_flow=False,
        reverse_flow=False,
        dims=dims,
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)
    assert len(trainable_variables) == len(mock_realnvp_bijectors) // 2
    for i, v in enumerate(trainable_variables):
        assert isinstance(v, tf.Variable)
        assert v.trainable
        assert v == i
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0

    test_input1 = 5
    input_shape, all_parameters = parameter_fn(test_input1)
    assert input_shape is None
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert len(all_parameters) == len(mock_realnvp_bijectors)
    for i, p in enumerate(all_parameters):
        if isinstance(p, dict):
            test_input2 = i * tf.range(20.0)[..., None]
            output_shape = sum(p[__PARAMETERS_FN_KWARGS_KEY__].values(), [])
            test_result = test_input1 * (tf.ones(output_shape) * test_input2[..., None])

            assert callable(p[__PARAMETERS_KEY__])
            assert tf.reduce_all(p[__PARAMETERS_KEY__](test_input2) == test_result)
            for k in p.keys():
                assert k in __ALL_KEYS__
    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == []
    assert dist.event_shape == [dims]

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(32, dims, flow)

    for i, (p, b) in enumerate(zip(mock_realnvp_bijectors, dist.bijector.bijectors)):
        if isinstance(p, dict):
            assert isinstance(b, tfb.RealNVP)
            assert b._num_masked == 1
            assert isinstance(b._bijector_fn, Callable)

            test_input2 = -tf.range(20.0)[..., None]
            output_shape = sum(p[__PARAMETERS_FN_KWARGS_KEY__].values(), [])
            test_result = (
                test_input1
                * (tf.ones(output_shape) * test_input2[..., None])
                / (i // 2 + 1)
            )
            nested_flow = b._bijector_fn(test_input2)
            assert isinstance(nested_flow, tfb.Chain)
            assert len(nested_flow.bijectors) == 2

            assert isinstance(nested_flow.bijectors[0], tfb.Scale)
            assert tf.reduce_all(
                tf.abs(
                    nested_flow.bijectors[0].scale
                    - tf.math.softplus(test_result[..., 0])
                )
                < 1e-6
            )
            assert isinstance(nested_flow.bijectors[1], tfb.Shift)
            assert tf.reduce_all(
                nested_flow.bijectors[1].shift == i // 2 + test_result[..., 1]
            )
        else:
            assert p == b

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, False, None, dims)


def test_get_coupling_bernstein_flow(
    batch_size, num_samples, coupling_bernstein_flow_kwargs
):
    """Testing a coupling flow with Bernstein polynomial transformation."""
    dims = coupling_bernstein_flow_kwargs["dims"]
    is_conditional = coupling_bernstein_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__].get(
        "conditional", False
    )
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_coupling_flow(**coupling_bernstein_flow_kwargs)

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)

    expected_lenght = (
        (
            len(
                coupling_bernstein_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                    "hidden_units"
                ]
            )
            + 1
        )
        * coupling_bernstein_flow_kwargs["num_layers"]
        * (4 if is_conditional else 2)
        * (
            2
            if coupling_bernstein_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__].get(
                "batch_norm", False
            )
            else 1
        )
    )
    assert len(trainable_variables) == expected_lenght
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v in trainable_variables:
        assert isinstance(v, tf.Variable)
        assert v.trainable
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0

    if is_conditional:
        test_input1 = tf.ones(
            (
                batch_size,
                coupling_bernstein_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                    "conditional_event_shape"
                ],
            )
        )
    else:
        test_input1 = None
    input_shape, all_parameters = parameter_fn(test_input1)
    assert tf.reduce_all(input_shape == (test_input1.shape if is_conditional else None))
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    expected_lenght = (
        2 * coupling_bernstein_flow_kwargs["num_layers"]
        - coupling_bernstein_flow_kwargs["num_layers"] % 2
    )
    assert len(all_parameters) == expected_lenght
    for i, p in enumerate(all_parameters):
        if isinstance(p, dict):
            assert callable(p[__PARAMETERS_KEY__])
            num_masked = coupling_bernstein_flow_kwargs.get(
                "num_maksed",
                _get_num_masked(coupling_bernstein_flow_kwargs["dims"], i // 2),
            )

            test_input2 = i * tf.ones((batch_size, num_masked))
            output_shape = [
                batch_size,
                dims - num_masked,
                coupling_bernstein_flow_kwargs["num_parameters"],
            ]

            assert p[__PARAMETERS_KEY__](test_input2).shape == output_shape
            for k in p.keys():
                assert k in __ALL_KEYS__

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == ([batch_size] if is_conditional else [])
    assert dist.event_shape == [coupling_bernstein_flow_kwargs["dims"]]

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(batch_size, dims, flow)

    for i, (b) in enumerate(dist.bijector.bijectors):
        if i % 2 == 0:
            num_masked = coupling_bernstein_flow_kwargs.get(
                "num_maksed",
                _get_num_masked(coupling_bernstein_flow_kwargs["dims"], i // 2),
            )

            test_input2 = i * tf.ones((batch_size, num_masked))
            output_shape = [
                batch_size,
                dims - num_masked,
                coupling_bernstein_flow_kwargs["num_parameters"],
            ]
            assert isinstance(b, tfb.RealNVP)
            assert b._num_masked == num_masked
            assert isinstance(b._bijector_fn, Callable)

            nested_flow = b._bijector_fn(test_input2)

            assert isinstance(nested_flow, tfp.python.bijectors.invert._Invert)
            assert isinstance(nested_flow.bijector, BernsteinPolynomial)
            bpoly_bijector = nested_flow.bijector
            layer_overwrites = _get_layer_overwrites(
                coupling_bernstein_flow_kwargs["layer_overwrites"],
                i // 2,
                coupling_bernstein_flow_kwargs["num_layers"],
            )
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., 0]
                    - layer_overwrites.get(
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
                        coupling_bernstein_flow_kwargs[
                            __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                        ],
                    )["low"]
                )
                < 1e-5
            )
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., -1]
                    - layer_overwrites.get(
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
                        coupling_bernstein_flow_kwargs[
                            __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                        ],
                    )["high"]
                )
                < 1e-5
            )
        else:
            assert isinstance(b, tfb.Permute)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims)


def test_get_coupling_spline_flow(batch_size, num_samples, coupling_spline_flow_kwargs):
    """Testing a coupling flow with spline transformation."""
    dims = coupling_spline_flow_kwargs["dims"]
    is_conditional = coupling_spline_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__].get(
        "conditional", False
    )

    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_coupling_flow(**coupling_spline_flow_kwargs)

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)

    expected_lenght = (
        (
            len(
                coupling_spline_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                    "hidden_units"
                ]
            )
            + 1
        )
        * coupling_spline_flow_kwargs["num_layers"]
        * (4 if is_conditional else 2)
        * (
            2
            if coupling_spline_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__].get(
                "batch_norm", False
            )
            else 1
        )
    )
    assert len(trainable_variables) == expected_lenght
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v in trainable_variables:
        assert isinstance(v, tf.Variable)
        assert v.trainable
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0

    if is_conditional:
        test_input1 = tf.ones(
            (
                batch_size,
                coupling_spline_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                    "conditional_event_shape"
                ],
            )
        )
    else:
        test_input1 = None
    input_shape, all_parameters = parameter_fn(test_input1)
    assert tf.reduce_all(input_shape == (test_input1.shape if is_conditional else None))
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert (
        len(all_parameters)
        == 2 * coupling_spline_flow_kwargs["num_layers"]
        - coupling_spline_flow_kwargs["num_layers"] % 2
    )
    for i, p in enumerate(all_parameters):
        if isinstance(p, dict):
            assert callable(p[__PARAMETERS_KEY__])
            num_masked = coupling_spline_flow_kwargs.get(
                "num_maksed",
                _get_num_masked(coupling_spline_flow_kwargs["dims"], i // 2),
            )

            test_input2 = i * tf.ones((batch_size, num_masked))
            output_shape = [
                batch_size,
                dims - num_masked,
                coupling_spline_flow_kwargs["num_parameters"],
            ]

            assert p[__PARAMETERS_KEY__](test_input2).shape == output_shape
            for k in p.keys():
                assert k in __ALL_KEYS__

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == ([batch_size] if is_conditional else [])
    assert dist.event_shape == [coupling_spline_flow_kwargs["dims"]]

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(batch_size, dims, flow)

    for i, (b) in enumerate(dist.bijector.bijectors):
        if i % 2 == 0:
            num_masked = coupling_spline_flow_kwargs.get(
                "num_maksed",
                _get_num_masked(coupling_spline_flow_kwargs["dims"], i // 2),
            )

            test_input2 = i * tf.ones((batch_size, num_masked))
            output_shape = [
                batch_size,
                dims - num_masked,
                coupling_spline_flow_kwargs["num_parameters"],
            ]
            assert isinstance(b, tfb.RealNVP)
            assert b._num_masked == num_masked
            assert isinstance(b._bijector_fn, Callable)

            nested_flow = b._bijector_fn(test_input2)

            assert isinstance(nested_flow, tfb.RationalQuadraticSpline)
        else:
            assert isinstance(b, tfb.Permute)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims)


def test_get_masked_autoregressive_bernstein_flow(
    batch_size, num_samples, masked_autoregressive_bernstein_flow_kwargs
):
    """Testing a masked_autoregressive flow with Bernstein polynomial transformation."""
    dims = masked_autoregressive_bernstein_flow_kwargs["dims"]
    is_conditional = masked_autoregressive_bernstein_flow_kwargs[
        __PARAMETERS_FN_KWARGS_KEY__
    ].get("conditional", False)
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_masked_autoregressive_flow(**masked_autoregressive_bernstein_flow_kwargs)

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)

    made_variables = get_num_made_variables(masked_autoregressive_bernstein_flow_kwargs)
    lu_variables = masked_autoregressive_bernstein_flow_kwargs["num_layers"]
    assert len(trainable_variables) == made_variables + lu_variables
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v in trainable_variables:
        assert isinstance(v, tf.Variable)
        assert v.trainable
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == lu_variables

    if is_conditional:
        test_input1 = tf.ones(
            batch_size,
            masked_autoregressive_bernstein_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                "conditional_event_shape"
            ],
        )
    else:
        test_input1 = None
    input_shape, all_parameters = parameter_fn(test_input1)
    assert tf.reduce_all(input_shape == (test_input1.shape if is_conditional else None))
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert (
        len(all_parameters)
        == masked_autoregressive_bernstein_flow_kwargs["num_layers"] * 2
    )
    for i, p in enumerate(all_parameters):
        if isinstance(p, dict) and i % 2 == 0:
            check_maf_params_fn(
                p,
                i // 2,
                batch_size,
                dims,
                masked_autoregressive_bernstein_flow_kwargs,
                is_conditional,
            )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == ([batch_size] if is_conditional else [])
    assert dist.event_shape == [masked_autoregressive_bernstein_flow_kwargs["dims"]]

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(batch_size, dims, flow)

    for i, (b) in enumerate(dist.bijector.bijectors):
        if i % 2 == 0:
            test_input2 = i * tf.ones((batch_size, dims))
            assert isinstance(b, tfb.MaskedAutoregressiveFlow)
            assert isinstance(b._bijector_fn, Callable)

            nested_flow = b._bijector_fn(test_input2)

            assert isinstance(nested_flow, tfp.python.bijectors.invert._Invert)
            assert isinstance(nested_flow.bijector, BernsteinPolynomial)
            bpoly_bijector = nested_flow.bijector
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., 0]
                    - masked_autoregressive_bernstein_flow_kwargs[
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                    ]["low"]
                )
                < 1e-5
            )
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., -1]
                    - masked_autoregressive_bernstein_flow_kwargs[
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                    ]["high"]
                )
                < 1e-5
            )
        else:
            assert isinstance(b, tfb.ScaleMatvecLU)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims)


def test_get_masked_autoregressive_spline_flow(
    batch_size, num_samples, masked_autoregressive_spline_flow_kwargs
):
    """Testing a masked_autoregressive flow with spline transformation."""
    batch_size = 32
    dims = masked_autoregressive_spline_flow_kwargs["dims"]
    is_conditional = masked_autoregressive_spline_flow_kwargs[
        __PARAMETERS_FN_KWARGS_KEY__
    ].get("conditional", False)
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_masked_autoregressive_flow(**masked_autoregressive_spline_flow_kwargs)

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)

    made_variables = get_num_made_variables(masked_autoregressive_spline_flow_kwargs)
    lu_variables = masked_autoregressive_spline_flow_kwargs["num_layers"]
    assert len(trainable_variables) == made_variables + lu_variables
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v in trainable_variables:
        assert isinstance(v, tf.Variable)
        assert v.trainable
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == lu_variables

    if is_conditional:
        test_input1 = tf.ones(
            batch_size,
            masked_autoregressive_spline_flow_kwargs[__PARAMETERS_FN_KWARGS_KEY__][
                "conditional_event_shape"
            ],
        )
    else:
        test_input1 = None
    input_shape, all_parameters = parameter_fn(test_input1)
    assert tf.reduce_all(input_shape == (test_input1.shape if is_conditional else None))
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert (
        len(all_parameters)
        == masked_autoregressive_spline_flow_kwargs["num_layers"] * 2
    )
    for i, p in enumerate(all_parameters):
        if isinstance(p, dict) and i % 2 == 0:
            check_maf_params_fn(
                p,
                i // 2,
                batch_size,
                dims,
                masked_autoregressive_spline_flow_kwargs,
                is_conditional,
            )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == ([batch_size] if is_conditional else [])
    assert dist.event_shape == [masked_autoregressive_spline_flow_kwargs["dims"]]

    flow = dist.bijector
    assert isinstance(flow, tfp.python.bijectors.chain._Chain)

    assert_flow_shapes(batch_size, dims, flow)

    for i, (b) in enumerate(dist.bijector.bijectors):
        if i % 2 == 0:
            test_input2 = i * tf.ones((batch_size, dims))
            assert isinstance(b, tfb.MaskedAutoregressiveFlow)
            assert isinstance(b._bijector_fn, Callable)

            nested_flow = b._bijector_fn(test_input2)

            assert isinstance(nested_flow, tfb.RationalQuadraticSpline)
        else:
            assert isinstance(b, tfb.ScaleMatvecLU)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims)


def test_get_masked_autoregressive_flow_first_dim_masked(
    batch_size, num_samples, masked_autoregressive_flow_first_dim_masked_kwargs
):
    """Testing a masked_autoregressive flow with the first dimension masked."""
    dims = masked_autoregressive_flow_first_dim_masked_kwargs["dims"]
    x0_is_conditional = masked_autoregressive_flow_first_dim_masked_kwargs[
        "x0_parameters_fn_kwargs"
    ].get("conditional", False)
    maf_is_conditional = masked_autoregressive_flow_first_dim_masked_kwargs[
        "maf_parameters_fn_kwargs"
    ].get("conditional", False)
    is_conditional = x0_is_conditional or maf_is_conditional
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_masked_autoregressive_flow_first_dim_masked(
        **masked_autoregressive_flow_first_dim_masked_kwargs
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)

    expected_x0_lenght = (
        (
            len(
                masked_autoregressive_flow_first_dim_masked_kwargs[
                    "x0_parameters_fn_kwargs"
                ]["hidden_units"]
            )
            + 1
        )
        * (4 if x0_is_conditional else 2)
        * (
            2
            if masked_autoregressive_flow_first_dim_masked_kwargs[
                "x0_parameters_fn_kwargs"
            ].get("batch_norm", False)
            else 1
        )
    )
    expected_made_variables = (
        (
            len(
                masked_autoregressive_flow_first_dim_masked_kwargs[
                    "maf_parameters_fn_kwargs"
                ]["hidden_units"]
            )
            + 1
        )
        * masked_autoregressive_flow_first_dim_masked_kwargs["num_layers"]
        * 2
    )
    expected_lu_variables = masked_autoregressive_flow_first_dim_masked_kwargs[
        "num_layers"
    ]
    if expected_lu_variables == 1:
        # we skip permutations if there is just one MAF layer
        expected_lu_variables = 0
    assert len(trainable_variables) == (
        expected_x0_lenght + expected_made_variables + expected_lu_variables
    )
    assert len(set(map(id, trainable_variables))) == len(trainable_variables)
    for v in trainable_variables:
        assert isinstance(v, tf.Variable)
        assert v.trainable
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == expected_lu_variables

    if is_conditional:
        test_input1 = tf.ones(
            (
                batch_size,
                masked_autoregressive_flow_first_dim_masked_kwargs[
                    "x0_parameters_fn_kwargs"
                ].get(
                    "conditional_event_shape",
                    masked_autoregressive_flow_first_dim_masked_kwargs[
                        "maf_parameters_fn_kwargs"
                    ].get("conditional_event_shape"),
                ),
            )
        )
    else:
        test_input1 = None
    input_shape, all_parameters = parameter_fn(test_input1)
    assert tf.reduce_all(input_shape == (test_input1.shape if is_conditional else None))
    assert isinstance(all_parameters, list)

    assert_all_parameter_have_unique_id(all_parameters)
    assert (
        len(all_parameters[0][__NESTED_BIJECTOR_KEY__])
        == masked_autoregressive_flow_first_dim_masked_kwargs["num_layers"]
        + expected_lu_variables
    )

    realnvp_test_input2 = tf.ones((batch_size, 1))
    output_shape = [
        batch_size,
        dims - 1,
        masked_autoregressive_flow_first_dim_masked_kwargs["num_parameters"],
    ]
    realnvp_parameter_fn = all_parameters[0][__PARAMETERS_KEY__]
    assert realnvp_parameter_fn(realnvp_test_input2).shape == output_shape

    all_nested_parameters = all_parameters[0][__NESTED_BIJECTOR_KEY__]
    for i, p in enumerate(all_nested_parameters):
        if isinstance(p, dict) and i % 2 == 0:
            check_maf_params_fn(
                p,
                i // 2,
                batch_size,
                dims - 1,
                masked_autoregressive_flow_first_dim_masked_kwargs,
                maf_is_conditional,
            )

    dist = distribution_fn((input_shape, all_parameters))
    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.batch_shape == ([batch_size] if is_conditional else [])
    assert dist.event_shape == [
        masked_autoregressive_flow_first_dim_masked_kwargs["dims"]
    ]

    flow = dist.bijector
    assert isinstance(flow, tfb.RealNVP)

    assert_flow_shapes(batch_size, dims, flow)

    assert isinstance(flow._bijector_fn, Callable)
    nested_flow = flow._bijector_fn(realnvp_test_input2)

    if masked_autoregressive_flow_first_dim_masked_kwargs["num_layers"] > 1:
        assert isinstance(nested_flow, tfp.python.bijectors.chain._Chain)
        nested_bijectors = nested_flow.bijectors
    else:
        assert isinstance(nested_flow, tfb.MaskedAutoregressiveFlow)
        nested_bijectors = [nested_flow]

    for i, (b) in enumerate(nested_bijectors):
        if i % 2 == 0:
            test_input2 = i * tf.ones((batch_size, dims - 1))
            output_shape = [
                batch_size,
                dims - 1,
                masked_autoregressive_flow_first_dim_masked_kwargs["num_parameters"],
            ]
            assert isinstance(b, tfb.MaskedAutoregressiveFlow)
            assert isinstance(b._bijector_fn, Callable)

            nested_flow = b._bijector_fn(test_input2)

            assert isinstance(nested_flow, tfp.python.bijectors.invert._Invert)
            assert isinstance(nested_flow.bijector, BernsteinPolynomial)
            bpoly_bijector = nested_flow.bijector
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., 0]
                    - masked_autoregressive_flow_first_dim_masked_kwargs[
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                    ]["low"]
                )
                < 1e-5
            )
            assert tf.reduce_all(
                tf.abs(
                    bpoly_bijector.thetas[..., -1]
                    - masked_autoregressive_flow_first_dim_masked_kwargs[
                        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__
                    ]["high"]
                )
                < 1e-5
            )
        else:
            assert isinstance(b, tfb.ScaleMatvecLU)

    # Test flow forward and inverse transformations
    assert_sample_shapes(dist, num_samples, is_conditional, batch_size, dims)
