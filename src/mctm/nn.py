"""Contains helper functions to build neural networks."""
from typing import Callable, List, Tuple

import tensorflow as tf
from tensorflow import keras as K


# class definitions ############################################################
class ConstantLayer(K.layers.Layer):
    """A layer to generate constant values of predefined shape trainable in the model.

    Parameters
    ----------
    parameter_shape
        Shape of the constant parameter.

    """

    def __init__(self, parameter_shape: Tuple[int, ...]):
        """Initialize a constant layer.

        Parameters
        ----------
        parameter_shape
            Shape of the constant parameter.

        """
        super().__init__()
        self.theta_0 = self.add_weight(
            shape=parameter_shape, initializer="random_normal", trainable=True
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Operates on inputs to produce a tensor of constants scaled by inputs.

        Parameters
        ----------
        inputs
            Input tensor.

        Returns
        -------
            Output tensor of constant values scaled by inputs.

        """
        return self.theta_0[None, ...] * tf.ones_like(inputs)[:, :1, None]


# function definitions #########################################################
def _get_fully_connected_net(
    input_shape: Tuple[int, ...],
    hidden_units: List[int],
    activation: str,
    batch_norm: bool,
    output_shape: Tuple[int, ...],
    dropout: float,
    conditional: bool = False,
    conditional_event_shape: Tuple[int, ...] = None,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "fc",
    **kwargs,
) -> K.Model:
    """Create a simple fully connected neural network.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    hidden_units
        Number of units in each hidden layer.
    activation
        Activation function for the hidden units.
    batch_norm
        If True, apply batch normalization.
    output_shape
        Shape of the output layer.
    dropout
        Dropout rate, between 0 and 1.
    conditional
        If True, network is conditional, by default False.
    conditional_event_shape
        Shape of additional conditional input, by default None.
    dtype
        The dtype of the operation, by default tf.float32
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to Dense layers.

    Returns
    -------
        A Keras model representing the fully connected network.

    """
    x = K.Input(shape=input_shape, name="input", dtype=dtype)
    inputs = [x]

    if conditional:
        assert (
            conditional_event_shape is not None
        ), "Conditional event shape must be provided if network is conditional."
        c = K.Input(
            shape=conditional_event_shape, name="conditional_input", dtype=dtype
        )
        inputs.append(c)

    if batch_norm:
        x = K.layers.BatchNormalization(name="batch_norm", dtype=dtype)(x)
        if conditional:
            c = K.layers.BatchNormalization(name="conditional_batch_norm", dtype=dtype)(
                c
            )

    for i, h in enumerate(hidden_units):
        x = K.layers.Dense(
            h, activation=None, name=f"hidden_layer_{i}", dtype=dtype, **kwargs
        )(x)
        if dropout > 0:
            x = K.layers.Dropout(dropout, name=f"dropout_{i}", dtype=dtype)(x)
        if conditional:
            c_out = K.layers.Dense(
                h,
                activation=None,
                name=f"conditional_hidden_layer_{i}",
                dtype=dtype,
                **kwargs,
            )(c)
            if dropout > 0:
                c_out = K.layers.Dropout(
                    dropout, name=f"conditional_dropout_{i}", dtype=dtype
                )(c_out)
            x = K.layers.Add(name=f"add_c_out_{i}", dtype=dtype)([x, c_out])
        x = K.layers.Activation(activation, name=f"{activation}_{i}", dtype=dtype)(x)

    pv = K.layers.Dense(
        tf.reduce_prod(output_shape),
        activation="linear",
        name="parameter_vector",
        dtype=dtype,
        **kwargs,
    )(x)
    pv_reshaped = K.layers.Reshape(output_shape, name="reshape_pv", dtype=dtype)(pv)
    return K.Model(inputs=inputs, outputs=pv_reshaped, name=name)


def _get_res_net(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    res_blocks: int,
    res_block_units: int,
    activation: str = "relu",
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "fc_res_net",
    **kwargs,
) -> K.Model:
    """Generate an autoregressive ResNet model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    res_blocks
        Number of residual blocks, by default 0.
    res_block_units
        Hidden units of residual blocks in and output layers.
    activation
        Activation function for the hidden units, by default `"relu"`.
    dtype
        The dtype of the operation, by default `tf.float32`.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to `_get_fully_connected_net`.


    Returns
    -------
        The parameter network model.

    """
    inputs = K.Input(shape=input_shape, name="res_net_input", dtype=dtype)

    x = inputs
    if res_blocks:
        input_shape = [res_block_units]
        x = K.layers.Dense(
            tf.reduce_prod(res_block_units),
            activation=activation,
            name="hidden_layer_res_in",
            dtype=dtype,
        )(x)
        for r in range(res_blocks):
            fc_out = _get_fully_connected_net(
                input_shape=input_shape,
                output_shape=input_shape,
                activation=activation,
                name=f"fc_res{r}",
                dtype=dtype,
                **kwargs,
            )(x)
            res_out = K.layers.Add(name=f"add_res{r}")([fc_out, x])
            x = K.layers.Activation(activation, name=f"{activation}_res{r}")(res_out)
    out = _get_fully_connected_net(
        input_shape=input_shape,
        output_shape=output_shape,
        activation=activation,
        name="fc_out",
        dtype=dtype,
        **kwargs,
    )(x)

    return K.models.Model(inputs=inputs, outputs=out, name=name)


def _get_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    submodel_build_fn: Callable[..., K.Model],
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "ar_net",
    **kwargs,
) -> K.Model:
    """Generate an autoregressive model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    submodel_build_fn
        Function to build the sub models for each autoregressive component.
    dtype
        The dtype of the model, by default `tf.float32`.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to `submodel_build_fn`.


    Returns
    -------
        The autoregressive parameter network model.

    """
    inputs = K.Input(shape=input_shape, name="ar_input", dtype=dtype)

    fc = [ConstantLayer(output_shape)(inputs)]
    for i in range(1, input_shape[0]):
        x = inputs[..., :i]
        fc_res_net = submodel_build_fn(
            input_shape=[i],
            output_shape=output_shape,
            name=f"ar_sub_net_{i}",
            dtype=dtype,
            **kwargs,
        )(x)

        fc.append(fc_res_net)

    out = K.layers.Concatenate(axis=-2)(fc)

    return K.models.Model(inputs=inputs, outputs=out, name=name)


def _get_autoregressive_res_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    name: str = "ar_res_net",
    **kwargs,
) -> K.Model:
    """Generate an autoregressive fully connected model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to Dense `_get_res_net`.

    Returns
    -------
        The parameter network model as a callable and a list of its trainable variables.

    """
    return _get_autoregressive_net(
        input_shape=input_shape,
        output_shape=output_shape,
        submodel_build_fn=_get_fully_connected_net,
        **kwargs,
    )


def _get_autoregressive_res_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    name: str = "ar_res_net",
    **kwargs,
) -> K.Model:
    """Generate an autoregressive ResNet model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to Dense `_get_res_net`.

    Returns
    -------
        The parameter network model as a callable and a list of its trainable variables.

    """
    return _get_autoregressive_net(
        input_shape=input_shape,
        output_shape=output_shape,
        submodel_build_fn=_get_res_net,
        **kwargs,
    )


def _get_conditional_net(
    input_shape: Tuple[int],
    conditional_event_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    parameter_net: K.Model,
    conditioning_net_build_fn: Callable[..., K.Model] = _get_fully_connected_net,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "conditional_net",
    **kwargs,
) -> K.Model:
    """Combine a given model with an additional conditional input.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    conditional_event_shape
        Shape of additional conditional input.
    output_shape
        Shape of the output layer.
    parameter_net
        Keras Model to add add conditional input to.
    conditioning_net_build_fn:
        Function to build the conditioning model for each autoregressive component.
    dtype
        The dtype of the operation, by default tf.float32.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to Dense layers.

    Returns
    -------
        The conditional Model.

    """
    inputs = []
    normal_input = K.Input(shape=input_shape, name="input", dtype=dtype)
    conditional_input = K.Input(
        shape=conditional_event_shape, name="conditional_input", dtype=dtype
    )
    inputs.append(conditional_input)
    inputs = [normal_input, conditional_input]

    out = parameter_net(normal_input)

    conditional_fc_out = conditioning_net_build_fn(
        input_shape=conditional_event_shape,
        output_shape=output_shape,
        name="conditioning_net",
        **kwargs,
    )(conditional_input)

    out = K.layers.Add(name="add_conditional_out")([out, conditional_fc_out])

    return K.models.Model(inputs=inputs, outputs=out, name=name)
