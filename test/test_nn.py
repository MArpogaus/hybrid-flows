# %% Description ###############################################################
"""Test for neural network building functions."""

# %% imports ###################################################################
from typing import List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from hybrid_flows.nn import (
    ConstantLayer,
    build_conditional_net,
    build_fully_connected_autoregressive_net,
    build_fully_connected_net,
    build_fully_connected_res_net,
    build_masked_autoregressive_net,
)


# %% fixtures ##################################################################
@pytest.fixture(params=[(2, 3, 3), (23, 3), (2, 33), (32, 16)])
def random_tensor_fixture(request) -> tf.Tensor:
    """Fixture that creates a random tensor."""
    return tf.random.normal(request.param)


# %% tests #####################################################################
def test_constant_layer(random_tensor_fixture):
    """Test the ConstantLayer class."""
    parameter_shape = [3, 3]
    layer = ConstantLayer(parameter_shape)
    input_tensor = random_tensor_fixture
    output = layer(input_tensor)

    # Basic shape test
    assert output.shape == tf.broadcast_static_shape(
        tf.TensorShape([1] + parameter_shape), input_tensor[..., :1, None].shape
    )

    # Check if output is constant along the batch dimension
    assert np.allclose(output.numpy()[0], output.numpy()[1])

    # Checking trainable weights
    assert len(layer.trainable_variables) == 1
    assert layer.trainable_variables[0].shape == parameter_shape


@pytest.mark.parametrize(
    "input_shape, hidden_units, activation, batch_norm, output_shape, dropout",
    [
        ((8,), [16, 32], "relu", True, (10,), 0.5),
        ((16,), [32, 64, 128], "sigmoid", False, (5,), 0.2),
    ],
)
def test_build_fully_connected_net(
    input_shape: Tuple[int, ...],
    hidden_units: List[int],
    activation: str,
    batch_norm: bool,
    output_shape: Tuple[int, ...],
    dropout: float,
):
    """Test build_fully_connected_net function."""
    model = build_fully_connected_net(
        input_shape, hidden_units, activation, batch_norm, output_shape, dropout
    )

    # Checking input and output shapes
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1:] == output_shape

    # Validate dense layers and activation functions
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)
    ]
    assert len(dense_layers) == len(hidden_units) + 1  # Including output layer

    activation_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Activation)
    ]
    assert len(activation_layers) == len(
        hidden_units
    )  # Should be equal to number of hidden units

    # Check dropout layers
    dropout_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    ]
    if dropout > 0:
        assert len(dropout_layers) == len(
            hidden_units
        )  # Should have dropout layer for each hidden unit

    # Check batch normalization
    batch_norm_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    if batch_norm:
        assert (
            len(batch_norm_layers) == len(hidden_units) + 1
        )  # Should apply batch norm after each dense layer

    total_params = sum(layer.count_params() for layer in dense_layers)
    assert total_params > 0, "The model should have non-zero parameters"


@pytest.mark.parametrize(
    "input_shape, conditional_event_shape, output_shape, hidden_units, activation,"
    "batch_norm, dropout",
    [
        ((8,), (3,), (5,), [16, 32], "relu", True, 0.5),
        ((16,), (5,), (10,), [16, 32], "tanh", False, 0.0),
    ],
)
def test_build_conditional_net(
    input_shape: Tuple[int, ...],
    conditional_event_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    hidden_units: List[int],
    activation: str,
    batch_norm: bool,
    dropout: float,
):
    """Test build_conditional_net function."""
    parameter_net = build_fully_connected_net(
        input_shape, [32, 64], "relu", True, output_shape, 0.1
    )
    conditional_net = build_conditional_net(
        input_shape,
        conditional_event_shape,
        output_shape,
        parameter_net,
        conditioning_net_build_fn=build_fully_connected_net,
        hidden_units=hidden_units,
        activation=activation,
        batch_norm=batch_norm,
        dropout=dropout,
    )

    # Checking input shapes
    assert conditional_net.input_shape[0][1:] == input_shape
    assert conditional_net.input_shape[1][1:] == conditional_event_shape

    # Check output shape
    assert conditional_net.output_shape[1:] == output_shape

    # Ensure models have expected components
    add_layer = [
        layer
        for layer in conditional_net.layers
        if isinstance(layer, tf.keras.layers.Add)
    ]
    assert len(add_layer) == 1, "Model should use Add layer to combine conditionals"


@pytest.mark.parametrize(
    "input_shape, output_shape, conditional, conditional_event_shape",
    [
        ((8,), (8, 3), False, None),
        ((16,), (16, 5), True, (4,)),
    ],
)
def test_build_masked_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    conditional: bool,
    conditional_event_shape: Tuple[int],
):
    """Test build_masked_autoregressive_net function."""
    model = build_masked_autoregressive_net(
        input_shape, output_shape, conditional, conditional_event_shape
    )

    # Validate input and output shapes
    if conditional:
        assert model.input_shape[0][1:] == input_shape
        assert model.input_shape[1][1:] == conditional_event_shape
    else:
        assert model.input_shape[1:] == input_shape
    assert model.output_shape[1:] == output_shape

    # Validate presence of AutoregressiveNetwork layer
    autoregressive_layers = [
        layer for layer in model.layers if layer.name.endswith("network")
    ]
    assert len(autoregressive_layers) == 1, (
        "Model should include an AutoregressiveNetwork layer"
    )

    assert sum(layer.count_params() for layer in model.layers) > 0, (
        "Model must have parameters"
    )


@pytest.mark.parametrize(
    "input_shape, output_shape, hidden_units, activation, batch_norm, dropout",
    [
        ((8,), (8, 3), [16, 32], "relu", True, 0.5),
        ((16,), (16, 5), [16, 32, 64], "sigmoid", False, False),
    ],
)
def test_build_fully_connected_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    hidden_units: List[int],
    activation: str,
    batch_norm: bool,
    dropout: float,
):
    """Test build_fully_connected_autoregressive_net function."""
    model = build_fully_connected_autoregressive_net(
        input_shape,
        output_shape,
        hidden_units=hidden_units,
        activation=activation,
        batch_norm=batch_norm,
        dropout=dropout,
    )

    # Validate input and output consistency
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1:] == output_shape

    # Validate layers created by submodel_build_fn
    assert any(isinstance(layer, tf.keras.layers.Layer) for layer in model.layers), (
        "Model should have multiple layers"
    )

    assert sum(layer.count_params() for layer in model.layers) > 0, (
        "Model must have parameters"
    )


@pytest.mark.parametrize(
    "input_shape, output_shape, res_blocks, hidden_features, activation, batch_norm,"
    "dropout",
    [
        ((8,), (10,), 2, 16, "relu", True, 0.2),
        ((16,), (5,), 3, 32, "relu", False, 0.0),
    ],
)
def test_build_fully_connected_res_net(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    res_blocks: int,
    hidden_features: int,
    activation: str,
    batch_norm: bool,
    dropout: float,
):
    """Test build_fully_connected_res_net function."""
    model = build_fully_connected_res_net(
        input_shape,
        output_shape,
        res_blocks,
        hidden_features,
        activation,
        batch_norm,
        dropout,
    )

    # Validate input/output shapes and consistency
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1:] == output_shape

    # Check presence of required components
    add_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Add)
    ]
    assert len(add_layers) == res_blocks, (
        "Model should have Add layers equal to the number of res_blocks"
    )

    # Check if dense layers exist and are connected properly
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)
    ]
    assert len(dense_layers) == res_blocks * 2 + 2, (
        "Model should have two Dense layers per res block plus two for input and output"
    )

    # Validate dropout usage
    if dropout > 0:
        dropout_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dropout)
        ]
        assert len(dropout_layers) == res_blocks, (
            "Should have dropout layer for each res block with dropout configured"
        )

    # Check if batch normalization is present where applicable
    if batch_norm:
        batch_norm_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        assert len(batch_norm_layers) == res_blocks * 2, (
            "Each res block should have two batch normalization layers"
        )

    total_params = sum(layer.count_params() for layer in dense_layers)
    assert total_params > 0, "The model should have non-zero parameters"
