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
):
    x = K.Input(input_shape, name="input")
    inputs = [x]

    if conditional:
        c = K.Input(conditional_event_shape, name="conditional_input")
        inputs += [c]
    if batch_norm:
        x = K.layers.BatchNormalization(name="batch_norm")(x)
        if conditional:
            c = K.layers.BatchNormalization(name="conditional_batch_norm")(c)

    for i, h in enumerate(hidden_units):
        x = K.layers.Dense(h, activation=None, name=f"hidden{i}_layer")(x)
        if conditional:
            c_out = K.layers.Dense(
                h, activation=None, name=f"conditional_hidden{i}_layer"
            )(c)
            x = K.layers.Add(name=f"add_c_out{i}")([x, c_out])
        x = K.layers.Activation(activation)(x)

    pv = K.layers.Dense(
        tf.reduce_prod(output_shape), activation="linear", name="parameter_vector"
    )(x)
    pv_reshaped = K.layers.Reshape(output_shape)(pv)
    return K.Model(inputs=inputs, outputs=pv_reshaped)


# PUBLIC FUNCTIONS #############################################################
def get_parameter_vector_lambda(parameters_shape, dtype):
    parameter_vector = tf.Variable(
        tf.random.normal(parameters_shape, dtype=dtype), trainable=True
    )
    return lambda *_, **__: parameter_vector, parameter_vector


def get_simple_fully_connected_parameter_network_lambda(
    parameter_shape, input_shape, **kwds
):
    parameter_network = __get_simple_fully_connected_network__(
        input_shape=input_shape, output_shape=parameter_shape, **kwds
    )
    parameter_network.build(input_shape)
    return parameter_network, parameter_network.trainable_variables


def get_autoregressive_parameter_network_lambda(parameter_shape, **kwds):
    dims = parameter_shape[:1]
    params = tf.reduce_prod(parameter_shape[1:])
    parameter_network = tfb.AutoregressiveNetwork(
        params=params, event_shape=dims, **kwds
    )
    parameter_network.build(dims)

    def parameter_network_lambda(conditional_input=None, **kwds):
        return partial(parameter_network, conditional_input=conditional_input, **kwds)

    return parameter_network_lambda, parameter_network.trainable_variables
