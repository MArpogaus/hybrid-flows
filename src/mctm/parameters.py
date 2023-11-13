# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : parameters.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-24 16:15:23 (Marcel Arpogaus)
# changed : 2023-08-24 16:15:23 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
"""Functions defining ANNs.

The 'parameters' module contains functions for defining ANNs based on certain
parameters as used in various distribution models.

It includes functions for creating simple fully connected networks, parameter
vectors, and autoregressive parameter networks.

"""

from functools import partial

import tensorflow as tf
from tensorflow import keras as K
from tensorflow_probability import bijectors as tfb


# PRIVATE FUNCTIONS ############################################################
def __get_simple_fully_connected_network__(
    input_shape,
    hidden_units,
    activation,
    batch_norm,
    output_shape,
    conditional=False,
    conditional_event_shape=None,
    dtype=tf.float32,
    **kwds,
):
    """Create a simple fully connected neural network for parameter estimation.

    :param tuple input_shape: The shape of the input.
    :param list hidden_units: List of integers specifying the number of hidden
                              units in each layer.
    :param str activation: Activation function to use in hidden layers.
    :param bool batch_norm: Whether to use batch normalization.
    :param tuple output_shape: The shape of the output parameter vector.
    :param bool conditional: Flag indicating whether the network is conditional.
    :param tuple conditional_event_shape: The shape of conditional input, if
                                          conditional is True.
    :param dtype: Data type for the network.
    :param **kwds: Additional keyword arguments.
    :return: A Keras model representing the parameter network.
    :rtype: keras.Model
    """
    x = K.Input(input_shape, name="input", dtype=dtype)
    inputs = [x]

    if conditional:
        c = K.Input(conditional_event_shape, name="conditional_input", dtype=dtype)
        inputs += [c]
    if batch_norm:
        x = K.layers.BatchNormalization(name="batch_norm", dtype=dtype)(x)
        if conditional:
            c = K.layers.BatchNormalization(name="conditional_batch_norm", dtype=dtype)(
                c
            )

    for i, h in enumerate(hidden_units):
        x = K.layers.Dense(
            h, activation=None, name=f"hidden{i}_layer", dtype=dtype, **kwds
        )(x)
        if conditional:
            c_out = K.layers.Dense(
                h,
                activation=None,
                name=f"conditional_hidden{i}_layer",
                dtype=dtype,
                **kwds,
            )(c)
            x = K.layers.Add(name=f"add_c_out{i}", dtype=dtype)([x, c_out])
        x = K.layers.Activation(activation, dtype=dtype)(x)

    pv = K.layers.Dense(
        tf.reduce_prod(output_shape),
        activation="linear",
        name="parameter_vector",
        dtype=dtype,
        **kwds,
    )(x)
    pv_reshaped = K.layers.Reshape(output_shape, dtype=dtype)(pv)
    return K.Model(inputs=inputs, outputs=pv_reshaped)


# PUBLIC FUNCTIONS #############################################################
def get_parameter_vector_lambda(parameters_shape, dtype):
    """Create a parameter vector as a trainable TensorFlow variable.

    :param tuple parameters_shape: The shape of the parameter vector.
    :param dtype: Data type of the parameter vector.
    :return: A callable that returns the parameter vector.
    :rtype: Callable
    """
    parameter_vector = tf.Variable(
        tf.random.normal(parameters_shape, dtype=dtype), trainable=True
    )
    return lambda *_, **__: parameter_vector, parameter_vector


def get_simple_fully_connected_parameter_network_lambda(
    parameter_shape, input_shape, **kwds
):
    """Create a simple fully connected parameter network for density estimation.

    :param tuple parameter_shape: The shape of the parameter.
    :param tuple input_shape: The shape of the input to the network.
    :param **kwds: Additional keyword arguments.
    :return: A tuple containing a callable parameter network and its
             trainable variables.
    :rtype: Tuple
    """
    parameter_network = __get_simple_fully_connected_network__(
        input_shape=input_shape, output_shape=parameter_shape, **kwds
    )
    parameter_network.build(input_shape)

    if kwds.get("conditional", False):

        def parameter_network_lambda(conditional_input=None, **kwds):
            return lambda x: parameter_network([x, conditional_input], **kwds)

    else:

        def parameter_network_lambda(conditional_input=None, **kwds):
            return parameter_network(conditional_input, **kwds)

    return parameter_network_lambda, parameter_network.trainable_variables


def get_parameter_vector_or_simple_network_lambda(parameter_shape, conditional, **kwds):
    """Create parameter vector.
    
    Create either a parameter vector or a simple fully connected parameter
    network based on conditional flag.

    :param tuple parameter_shape: The shape of the parameter.
    :param bool conditional: Flag indicating whether to use conditional
                             networks.
    :param **kwds: Additional keyword arguments.
    :return: A tuple containing a callable parameter network and its
             trainable variables.
    :rtype: Tuple
    """
    if conditional:
        return get_simple_fully_connected_parameter_network_lambda(
            parameter_shape, **kwds
        )
    else:
        return get_parameter_vector_lambda(parameter_shape, **kwds)


def get_autoregressive_parameter_network_lambda(parameter_shape, **kwds):
    """Create an autoregressive parameter network.

    :param tuple parameter_shape: The shape of the parameter.
    :param **kwds: Additional keyword arguments.
    :return: A tuple containing a callable autoregressive parameter network
             and its trainable variables.
    :rtype: Tuple
    """
    dims = parameter_shape[:1]
    params = tf.reduce_prod(parameter_shape[1:])
    parameter_network = tfb.AutoregressiveNetwork(
        params=params, event_shape=dims, **kwds
    )
    parameter_network.build(dims)

    def parameter_network_lambda(conditional_input=None, **kwds):
        return partial(parameter_network, conditional_input=conditional_input, **kwds)

    return parameter_network_lambda, parameter_network.trainable_variables


def get_autoregressive_parameter_network_with_additive_conditioner_lambda(
    parameter_shape, made_kwds, x0_kwds
):
    """Create an autoregressive parameter network with an additive conditioner.

    :param tuple parameter_shape: The shape of the parameter.
    :param made_kwds: Keyword arguments for MADE (Masked Autoregressive Flow
                      with Autoregressive Conditioner).
    :param x0_kwds: Keyword arguments for the additive conditioner network.
    :return: A tuple containing a callable parameter network and its
             trainable variables.
    :rtype: Tuple
    """
    (
        masked_autoregressive_parameter_network_lambda,
        masked_autoregressive_trainable_variables,
    ) = get_autoregressive_parameter_network_lambda(
        parameter_shape,
        **made_kwds,
    )

    (
        x0_parameter_network_lambda,
        x0_trainable_variables,
    ) = get_simple_fully_connected_parameter_network_lambda(
        parameter_shape=parameter_shape, input_shape=1, **x0_kwds
    )

    def parameter_lambda(conditional_input=None, **kwds):
        made_net = masked_autoregressive_parameter_network_lambda(
            conditional_input, **kwds
        )
        x0_net = x0_parameter_network_lambda  # (conditional_input, **kwds)

        def call(x, conditional_input):
            pv1 = made_net(x)
            pv2 = x0_net(conditional_input)
            return pv1 + pv2

        return call

    trainable_parameters = (
        masked_autoregressive_trainable_variables + x0_trainable_variables
    )
    return parameter_lambda, trainable_parameters
