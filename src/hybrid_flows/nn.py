"""Contains helper functions to build neural networks."""

from functools import reduce
from typing import Callable, List, Tuple

import tensorflow as tf
import tensorflow.keras as K
from tensorflow_probability import bijectors as tfb


# class definitions ############################################################
class ConstantLayer(K.layers.Layer):
    """A layer to generate constant values of predefined shape trainable in the model.

    Parameters
    ----------
    parameter_shape
        Shape of the constant parameter.

    """

    def __init__(self, parameter_shape: Tuple[int, ...], **kwargs):
        """Initialize a constant layer.

        Parameters
        ----------
        parameter_shape
            Shape of the constant parameter.
        kwargs
            Additional key-word arguments passed to the initializer of `K.layers.Layer`.

        """
        super().__init__(**kwargs)
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
        return self.theta_0[None, ...] * tf.ones_like(inputs)[..., :1, None]


# function definitions #########################################################
def _build_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    submodel_build_fn: Callable[..., K.Model],
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "ar",
    **kwargs,
) -> K.Model:
    """Build an autoregressive model.

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

    assert input_shape[0] == output_shape[0], (
        "First dim of input and output shapes must be equal"
    )
    fc = [ConstantLayer(output_shape[1:], name=f"{name}_constant_layer")(inputs)]
    for i in range(1, input_shape[0]):
        x = inputs[..., :i]
        fc_out = submodel_build_fn(
            input_shape=[i],
            output_shape=[1] + list(output_shape[1:] if len(output_shape) > 1 else [1]),
            name=f"{name}_sub_net_{i}",
            dtype=dtype,
            **kwargs,
        )(x)

        fc.append(fc_out)

    out = K.layers.Concatenate(axis=-2)(fc)
    out = K.layers.Reshape(output_shape)(out)

    return K.models.Model(inputs=inputs, outputs=out, name=name)


def build_fully_connected_net(
    input_shape: Tuple[int, ...],
    hidden_units: List[int],
    activation: str,
    batch_norm: bool,
    output_shape: Tuple[int, ...],
    dropout: float,
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
    x = K.Input(shape=input_shape, name="/".join((name, "input")), dtype=dtype)
    inputs = [x]

    if batch_norm:
        x = K.layers.BatchNormalization(
            name="/".join((name, "batch_norm")), dtype=dtype
        )(x)

    for i, h in enumerate(hidden_units):
        if dropout > 0:
            x = K.layers.Dropout(dropout, name="/".join((name, f"dropout_{i}")))(x)
        x = K.layers.Dense(
            h,
            activation=None,
            name="/".join((name, f"hidden_layer_{i}")),
            dtype=dtype,
            **kwargs,
        )(x)
        x = K.layers.Activation(
            activation, name="/".join((name, f"{activation}_{i}")), dtype=dtype
        )(x)
        if batch_norm:
            x = K.layers.BatchNormalization(
                name="/".join((name, f"batch_norm_{i}")), dtype=dtype
            )(x)

    pv = K.layers.Dense(
        tf.reduce_prod(output_shape),
        activation="linear",
        name="/".join((name, "parameter_vector")),
        dtype=dtype,
        **kwargs,
    )(x)
    pv_reshaped = K.layers.Reshape(output_shape, name="reshape_pv", dtype=dtype)(pv)
    return K.Model(inputs=inputs, outputs=pv_reshaped, name=name)


def build_conditional_net(
    input_shape: Tuple[int],
    conditional_event_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    parameter_net: K.Model,
    conditioning_net_build_fn: Callable[..., K.Model] = build_fully_connected_net,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "net",
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
    name = "/".join(("conditional", name))
    inputs = []
    normal_input = K.Input(
        shape=input_shape, name="/".join((name, "input")), dtype=dtype
    )
    conditional_input = K.Input(
        shape=conditional_event_shape,
        name="/".join((name, "conditional_input")),
        dtype=dtype,
    )
    inputs = [normal_input, conditional_input]

    out = parameter_net(normal_input)

    conditional_fc_out = conditioning_net_build_fn(
        input_shape=conditional_event_shape,
        output_shape=output_shape,
        name="/".join((name, "conditioning_net")),
        **kwargs,
    )(conditional_input)

    out = K.layers.Add(name="/".join((name, "add_conditional_out")))(
        [out, conditional_fc_out]
    )

    return K.models.Model(inputs=inputs, outputs=out, name=name)


def build_masked_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Tuple[int] = None,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "made",
    **kwargs,
) -> K.Model:
    """Build an masked autoregressive model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    conditional
        If True, network is conditional, by default False.
    conditional_event_shape
        Shape of additional conditional input.
    dtype
        The dtype of the operation, by default tf.float32.
    name
        Name of the Keras model.
    kwargs
        Additional keyword arguments passed to `tfb.AutoregressiveNetwork`.


    Returns
    -------
        The maksed autoregressive parameter network model.

    """

    def reduce_prod(iterable):
        return reduce(lambda x, r: x * r, tuple(iterable), 1)

    assert input_shape[0] == output_shape[0], (
        "First dim of input and output shapes must be equal"
    )

    params = reduce_prod(output_shape[1:])

    normal_input = K.Input(
        shape=input_shape, name="/".join((name, "input")), dtype=dtype
    )
    inputs = [normal_input]
    made_layer = tfb.AutoregressiveNetwork(
        event_shape=input_shape,
        params=params,
        name="/".join((name, "network")),
        **kwargs,
    )

    if conditional:
        conditional_input = K.Input(
            shape=conditional_event_shape,
            name="/".join((name, "conditional_input")),
            dtype=dtype,
        )
        inputs.append(conditional_input)
        made_out = made_layer(normal_input, conditional_input=conditional_input)
    else:
        made_out = made_layer(normal_input)

    out = K.layers.Reshape(output_shape, name="/".join((name, "reshape_pv")))(made_out)

    return K.Model(inputs=inputs, outputs=out, name=name)


def build_fully_connected_autoregressive_net(
    input_shape: Tuple[int],
    output_shape: Tuple[int, ...],
    name: str = "ar_res_net",
    **kwargs,
) -> K.Model:
    """Build an autoregressive fully connected model.

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
    return _build_autoregressive_net(
        input_shape=input_shape,
        output_shape=output_shape,
        submodel_build_fn=build_fully_connected_net,
        name=name,
        **kwargs,
    )


def build_fully_connected_res_net(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    res_blocks: int,
    hidden_features: int,
    activation: str,
    batch_norm: bool,
    dropout: float,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "fc_res_net",
    **kwargs,
) -> K.Model:
    """Build a ResNet pre-activation model.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    res_blocks
        Number of residual blocks.
    hidden_features
        Hidden features of residual blocks.
    activation
        Activation function for the hidden units.
    batch_norm
        If True, apply batch normalization.
    dropout
        Dropout rate, between 0 and 1.
    dtype
        The dtype of the operation, by default `tf.float32`.
    name
        Name of the Keras model, by default `"fc_res_net"`.
    kwargs
        Additional keyword arguments passed to `K.layers.Dense`.

    Reference
    ---------
    - K. He, X. Zhang, S. Ren, and J. Sun, Identity mappings in deep residual networks,
      European Conference on Computer Vision, 2016.
    - C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, Neural Spline Flows,
      in Advances in Neural Information Processing Systems, 2019.

    Returns
    -------
        The parameter network model.

    """
    inputs = K.Input(shape=input_shape, name="res_net_input", dtype=dtype)

    x = inputs
    projection = K.layers.Dense(
        hidden_features,
        activation=activation,
        name="/".join((name, "projection")),
        dtype=dtype,
    )(x)
    x = projection
    for r in range(res_blocks):
        if batch_norm:
            x = K.layers.BatchNormalization(
                name="/".join((name, f"batch_norm_{r}_1")), dtype=dtype
            )(x)
        x = K.layers.Activation(
            activation, name="/".join((name, f"{activation}_{r}_1")), dtype=dtype
        )(x)
        x = K.layers.Dense(
            hidden_features,
            activation="linear",
            name="/".join((name, f"hidden_layer_{r}_1")),
            dtype=dtype,
            **kwargs,
        )(x)
        if batch_norm:
            x = K.layers.BatchNormalization(
                name="/".join((name, f"batch_norm_{r}_2")), dtype=dtype
            )(x)
        x = K.layers.Activation(
            activation, name="/".join((name, f"{activation}_{r}_2")), dtype=dtype
        )(x)
        if dropout > 0:
            x = K.layers.Dropout(dropout, name="/".join((name, f"dropout_{r}")))(x)
        x = K.layers.Dense(
            hidden_features,
            activation="linear",
            name="/".join((name, f"hidden_layer_{r}_2")),
            dtype=dtype,
            **kwargs,
        )(x)
        x = K.layers.Add(name=f"add_res{r}")([projection, x])
    out = K.layers.Dense(
        tf.reduce_prod(output_shape),
        activation="linear",
        name="/".join((name, "output_layer")),
        dtype=dtype,
        **kwargs,
    )(x)
    out_reshaped = K.layers.Reshape(output_shape, name="reshaped_output", dtype=dtype)(
        out
    )
    return K.models.Model(inputs=inputs, outputs=out_reshaped, name=name)
