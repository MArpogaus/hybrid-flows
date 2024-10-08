"""Unit test for normalizing flow module."""

from typing import Callable

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from mctm.distributions import (
    __BIJECTOR_KWARGS_KEY__,
    __BIJECTOR_NAME_KEY__,
    __NESTED_BIJECTOR_KEY__,
    __PARAMETER_SLICE_SIZE_KEY__,
    __PARAMETERIZED_BY_PARENT_KEY__,
    __PARAMETERS_FN_KEY__,
    __PARAMETERS_FN_KWARGS_KEY__,
    __PARAMETERS_KEY__,
    _get_eval_parameter_fn,
    _init_bijector_from_dict,
    _init_parameters_fn,
    get_normalizing_flow,
)
from mctm.parameters import get_test_parameters_fn
from tensorflow_probability import bijectors as tfb


@pytest.fixture
def mock_bijector_data():
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
            __BIJECTOR_NAME_KEY__: "Log",
        },
        {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {"num_masked": 1},
            __NESTED_BIJECTOR_KEY__: [
                {
                    __BIJECTOR_NAME_KEY__: "Scale",
                    __PARAMETERIZED_BY_PARENT_KEY__: True,
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
            __PARAMETERS_FN_KWARGS_KEY__: {"input_shape": [1], "param_shape": [2]},
        },
        tfb.Permute(permutation=[1, 0]),
    ]


@pytest.fixture
def mock_parameters_fn():
    """Mock parameter function."""
    return get_test_parameters_fn


@pytest.fixture
def mock_base_distribution_data():
    """Mock data for base distribution."""
    return {
        "get_base_distribution": lambda: tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(2)
        ),
    }


def test_init_parameters_fn_constant(mock_bijector_data):
    """Test _init_parameters_fn with constant parameters."""
    bijector_config = {
        __PARAMETERS_KEY__: 42,
        __BIJECTOR_NAME_KEY__: "dummy",
    }
    parameters_fn, trainable_variables, non_trainable_variables = _init_parameters_fn(
        [bijector_config]
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
        _init_parameters_fn([bijector_config])
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
        _init_parameters_fn([bijector_config])
    )
    parameters_fn = all_parameters_fn[0][__PARAMETERS_KEY__]
    assert parameters_fn is not None
    test_input = tf.ones([3, 1])
    assert tf.reduce_all(parameters_fn(test_input) == tf.ones([3, 5]))


def test_init_bijector_from_dict(mock_bijector_data):
    """Test _init_bijector_from_dict with simple bijectors and nested bijectors."""
    bijectors_parameters_fns, trainable_variables, non_trainable_variables = (
        _init_parameters_fn(mock_bijector_data)
    )
    test_input1 = 10.0
    eval_parameter_fn = _get_eval_parameter_fn(test_input1)
    all_parameters = list(map(eval_parameter_fn, bijectors_parameters_fns))
    flow = list(map(_init_bijector_from_dict, all_parameters))
    scale_bijector = flow[0]
    assert isinstance(scale_bijector, tfb.Scale)
    assert scale_bijector.scale == 2.0

    shift_bijector = flow[1]
    assert isinstance(shift_bijector, tfb.Shift)
    assert shift_bijector.shift == 1.0

    exp_bijector = flow[2]
    assert isinstance(exp_bijector, tfb.Log)

    realnvp_bijector = flow[3]
    assert isinstance(realnvp_bijector, tfb.RealNVP)
    assert realnvp_bijector._num_masked == 1
    assert isinstance(realnvp_bijector._bijector_fn, Callable)

    test_input2 = tf.ones([5, 1])
    test_result = test_input1 * test_input2
    netsed_flow = realnvp_bijector._bijector_fn(test_input2)
    assert isinstance(netsed_flow, tfb.Chain)
    assert len(netsed_flow.bijectors) == 2

    assert isinstance(netsed_flow.bijectors[0], tfb.Scale)
    assert tf.reduce_all(netsed_flow.bijectors[0].scale == test_result)
    assert isinstance(netsed_flow.bijectors[1], tfb.Shift)
    assert tf.reduce_all(netsed_flow.bijectors[1].shift == 3 + test_result)

    permute_bijector = flow[4]
    assert isinstance(permute_bijector, tfb.Permute)
    assert tf.reduce_all(permute_bijector.permutation == [1, 0])


def test_get_normalizing_flow(
    mock_bijector_data, mock_parameters_fn, mock_base_distribution_data
):
    """Test get_normalizing_flow."""
    (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    ) = get_normalizing_flow(
        bijectors=mock_bijector_data, **mock_base_distribution_data
    )

    assert callable(distribution_fn)
    assert callable(parameter_fn)
    assert isinstance(trainable_variables, list)
    assert len(trainable_variables) == 2
    for v, e in zip(trainable_variables, (1.0, 3.0)):
        assert isinstance(v, tf.Variable)
        assert v.trainable
        assert v == e
    assert isinstance(non_trainable_variables, list)
    assert len(non_trainable_variables) == 0

    test_input1 = 4
    test_input2 = 8
    output_shape = sum(mock_bijector_data[3][__PARAMETERS_FN_KWARGS_KEY__].values(), [])
    test_result = test_input1 * test_input2 * tf.ones(output_shape)
    all_parameters = parameter_fn(test_input1)
    assert isinstance(all_parameters, list)
    assert len(all_parameters) == len(mock_bijector_data)
    assert all_parameters[0][__PARAMETERS_KEY__] is None
    assert all_parameters[1][__PARAMETERS_KEY__] == 1.0
    assert all_parameters[2][__PARAMETERS_KEY__] is None
    assert callable(all_parameters[3][__PARAMETERS_KEY__])
    assert tf.reduce_all(
        all_parameters[3][__PARAMETERS_KEY__](test_input2) == test_result
    )

    flow = distribution_fn(all_parameters)
    assert isinstance(flow, tfp.distributions.Distribution)

    # Test flow forward and inverse transformations
    transformed_sample = flow.sample(7)
    base_sample = flow.bijector.inverse(transformed_sample)
    reconstructed_sample = flow.bijector.forward(base_sample)

    assert transformed_sample.shape == (7, 2)
    assert base_sample.shape == (7, 2)
    assert reconstructed_sample.shape == (7, 2)
    assert tf.reduce_all(tf.abs(reconstructed_sample - transformed_sample) < 1e-6)
