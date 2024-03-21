# %% imports
from typing import Callable, List, Tuple

import tensorflow as tf
from mctm.parameters import (
    build_fully_connected_net,
    get_masked_autoregressive_network_fn,
)
from tensorflow import keras as K


# %% functions
class ConstantLayer(K.layers.Layer):
    """A layer to generate constant values of predefined shape trainable in the model.

    Parameters
    ----------
    parameter_shape
        Shape of the constant parameter.

    """

    def __init__(self, parameter_shape: Tuple[int, ...]):
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
            fc_out = build_fully_connected_net(
                input_shape=input_shape,
                output_shape=input_shape,
                activation=activation,
                name=f"fc_res{r}",
                dtype=dtype,
                **kwargs,
            )(x)
            res_out = K.layers.Add(name=f"add_res{r}")([fc_out, x])
            x = K.layers.Activation(activation, name=f"{activation}_res{r}")(res_out)
    out = build_fully_connected_net(
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
    conditioning_net_build_fn: Callable[..., K.Model] = build_fully_connected_net,
    dtype: tf.dtypes.DType = tf.float32,
    name: str = "conditional_net",
    **kwargs,
) -> K.Model:
    """Combines a given model with an additional conditional input.

    Parameters
    ----------
    input_shape
        The shape of the input data.
    output_shape
        Shape of the output layer.
    conditional_event_shape
        Shape of additional conditional input.
    dtype
        The dtype of the operation, by default tf.float32.
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


def get_autoregressive_res_net_parameter_network_fn(
    parameter_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Tuple[int, ...] = (),
    name: str = "autoregressive_res_net",
    dtype: tf.dtypes.DType = tf.float32,
    **kwargs,
) -> Tuple[Callable[..., K.Model], List[tf.Variable]]:
    """Generate an autoregressive ResNet model.

    Parameters
    ----------
    parameter_shape
        Shape of the parameter vector.
    res_blocks
        Number of residual blocks, by default 0.
    activation
        Activation function for the hidden units, by default "relu".
    conditional
        If True, network is conditional, by default False.
    conditional_event_shape
        Shape of additional conditional input, by default ().
    dtype
        The dtype of the operation, by default tf.float32.
    kwargs
        Additional keyword arguments passed to Dense layers.


    Returns
    -------
        The parameter network model as a callable and a list of its trainable variables.

    """
    dims = parameter_shape[0]
    output_shape = [1] + list(parameter_shape[1:])

    parameter_network = _get_autoregressive_res_net(
        input_shape=[dims],
        output_shape=output_shape,
        dtype=dtype,
        name=name,
        **kwargs,
    )

    if conditional:
        assert (
            conditional_event_shape is not None
        ), "Conditional event shape must be provided if network is conditional."

        parameter_network = _get_conditional_net(
            input_shape=[dims],
            conditional_event_shape=conditional_event_shape,
            output_shape=output_shape,
            parameter_net=parameter_network,
            conditioning_net_build_fn=_get_res_net,
            name=name,
            **kwargs,
        )

        # def parameter_network_fn(conditional_input, **kwargs):
        #     return lambda x: parameter_network([x, conditional_input], **kwargs)
    # else:

    # def parameter_network_fn(conditional_input, **kwargs):
    #     return lambda x: parameter_network(x, **kwargs)
    def parameter_network_fn(*args, **kwargs):
        return parameter_network

    return parameter_network_fn, parameter_network.trainable_variables


# %% res net
res_net = _get_res_net(
    input_shape=[2],
    output_shape=[2, 3],
    res_blocks=3,
    res_block_units=[50],
    batch_norm=True,
    dropout=0.2,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    res_net,
    to_file="res_net.png",
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)

# %% testing
params = dict(parameter_shape=(3, 2), hidden_units=[100, 100], activation="relu")
x = tf.ones((100, 3))
model_made, _ = get_masked_autoregressive_network_fn(**params)

model_made()(x).shape

model_ar, _ = get_autoregressive_res_net_parameter_network_fn(
    **params,
    batch_norm=True,
    dropout=0.2,
    res_blocks=2,
    conditional=True,
    conditional_event_shape=[3],
)
model = model_ar()
model(x).shape
model.summary()
K.utils.plot_model(
    model,
    to_file="model.png",
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
    show_layer_names=True,
)
