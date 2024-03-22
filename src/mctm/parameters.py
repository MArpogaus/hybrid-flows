# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : parameters.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-24 16:15:23 (Marcel Arpogaus)
# changed : 2024-03-22 15:41:16 (Marcel Arpogaus)
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

from typing import Any, Callable, Dict, List, Tuple, Union

import tensorflow as tf
from tensorflow import keras as K

from .nn import (
    build_conditional_net,
    build_fully_connected_autoregressive_net,
    build_fully_connected_net,
    build_fully_connected_res_net,
    build_masked_autoregressive_net,
)


def get_parameter_vector_fn(
    parameter_shape: Tuple[int, ...], dtype: tf.dtypes.DType = tf.float32
) -> Tuple[Callable[..., tf.Variable], tf.Variable]:
    """Create a TensorFlow parameter vector with a given shape and dtype.

    Parameters
    ----------
    parameter_shape
        The shape of the parameter vector.
    dtype
        The data type of the parameter vector, by default tf.float32.

    Returns
    -------
         A tuple with a callable that returns the parameter vector and the parameter
        vector itself.

    """
    parameter_vector = tf.Variable(
        tf.random.normal(parameter_shape, dtype=dtype), trainable=True
    )
    return lambda *_, **__: parameter_vector, parameter_vector


def get_fully_connected_network_fn(
    parameter_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Tuple[int, ...] = (),
    **kwargs,
) -> Tuple[
    Callable[tf.Variable, Callable[tf.Variable, tf.Variable]], List[tf.Variable]
]:
    """Create a simple fully connected parameter network.

    Parameters
    ----------
    parameter_shape
        Shape of the parameter to output by the network.
    input_shape
        Shape of the input data.
    conditional
        If True, network is conditional, by default False.
    conditional_event_shape
        Shape of additional conditional input, by default ().
    kwargs
        Additional keyword arguments for network configuration.

    Returns
    -------
        A tuple of a callable that returns the parameter network and a list of its
        trainable variables.

    """
    parameter_network = build_fully_connected_net(
        input_shape=input_shape, output_shape=parameter_shape, **kwargs
    )
    parameter_network.build(input_shape)

    if conditional:
        assert (
            conditional_event_shape is not None
        ), "Conditional event shape must be provided if network is conditional."

        parameter_network = build_conditional_net(
            input_shape=input_shape,
            conditional_event_shape=conditional_event_shape,
            output_shape=parameter_shape,
            parameter_net=parameter_network,
            conditioning_net_build_fn=build_fully_connected_net,
            **kwargs,
        )

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network([x, conditional_input], **kwargs)
    else:

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables


def get_parameter_vector_or_simple_network_fn(
    parameter_shape: Tuple[int, ...], conditional: bool, **kwargs
) -> Union[
    Tuple[K.Model, List[tf.Variable]],
    Tuple[Callable[tf.Variable, tf.Variable], tf.Variable],
]:
    """Create either a parameter vector or a fully connected parameter network.

    Parameters
    ----------
    parameter_shape
        Shape of the parameter.
    conditional
        If True, creates a conditional network; otherwise, creates a parameter vector.
    kwargs
        Additional keyword arguments.

    Returns
    -------
        A tuple consisting of either a Keras model and its trainable variables, or a
        callable parameter vector and the parameter vector itself.

    """
    if conditional:
        input_shape = kwargs.pop("input_shape")
        parameter_network = build_fully_connected_net(
            output_shape=parameter_shape, input_shape=input_shape, **kwargs
        )
        parameter_network.build(input_shape=input_shape)
        return parameter_network, parameter_network.trainable_variables
    else:
        return get_parameter_vector_fn(parameter_shape=parameter_shape, **kwargs)


def get_fully_connected_res_net_fn(
    parameter_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Tuple[int, ...] = (),
    name: str = "res_net",
    dtype: tf.dtypes.DType = tf.float32,
    **kwargs,
) -> Tuple[Callable[..., K.Model], List[tf.Variable]]:
    """Generate an fully connected ResNet model.

    Parameters
    ----------
    parameter_shape
        Shape of the parameter vector.
    input_shape
        The shape of the input data.
    conditional
        If True, network is conditional, by default False.
    conditional_event_shape
        Shape of additional conditional input, by default ().
    name
        Name of the Keras model.
    dtype
        The dtype of the operation, by default tf.float32.
    kwargs
        Additional keyword arguments passed to Dense layers.


    Returns
    -------
        The parameter network model as a callable and a list of its trainable variables.

    """
    parameter_network = build_fully_connected_res_net(
        input_shape=input_shape,
        output_shape=parameter_shape,
        dtype=dtype,
        name=name,
        **kwargs,
    )

    if conditional:
        assert (
            conditional_event_shape is not None
        ), "Conditional event shape must be provided if network is conditional."

        parameter_network = build_conditional_net(
            input_shape=conditional_event_shape,
            conditional_event_shape=conditional_event_shape,
            output_shape=parameter_shape,
            parameter_net=parameter_network,
            conditioning_net_build_fn=build_fully_connected_res_net,
            name=name,
            **kwargs,
        )

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network([x, conditional_input], **kwargs)
    else:

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables


def get_masked_autoregressive_network_fn(
    parameter_shape: Tuple[int, ...], conditional: bool = False, **kwargs
) -> Tuple[
    Callable[tf.Variable, Callable[tf.Variable, tf.Variable]], List[tf.Variable]
]:
    """Create an autoregressive parameter network.

    Parameters
    ----------
    parameter_shape
        Shape of the parameters.
    conditional
        If True, network is conditional, by default False.
    kwargs
        Additional keyword arguments.

    Returns
    -------
        A tuple of callable autoregressive parameter network function and its
        trainable variables.

    """
    dims = parameter_shape[:1]
    parameter_network = build_masked_autoregressive_net(
        input_shape=dims,
        output_shape=parameter_shape,
        conditional=conditional,
        **kwargs,
    )
    parameter_network.build(dims)

    if conditional:

        def parameter_network_fn(conditional_input=None, **kwargs):
            return lambda x: parameter_network([x, conditional_input], **kwargs)
    else:

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables


def get_masked_autoregressive_network_with_additive_conditioner_fn(
    parameter_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    made_kwargs: Dict[str, Any],
    x0_kwargs: Dict[str, Any],
) -> Tuple[
    Callable[
        tf.Variable,
        Callable[tf.Variable, Callable[[tf.Variable, tf.Variable], tf.Variable]],
    ],
    List[tf.Variable],
]:
    """Create an autoregressive parameter network with an additive conditioner.

    Parameters
    ----------
    parameter_shape
        The shape of the parameter.
    input_shape
        The shape of the input.
    made_kwargs
        Keyword arguments for MADE (Masked Autoregressive Flow with
        Autoregressive Conditioner).
    x0_kwargs
        Keyword arguments for the additive conditioner network.

    Returns
    -------
        A tuple containing a callable parameter network and its trainable variables.

    """
    (
        masked_autoregressive_parameter_network_fn,
        masked_autoregressive_trainable_variables,
    ) = get_masked_autoregressive_network_fn(
        parameter_shape,
        **made_kwargs,
    )

    (
        x0_parameter_network_fn,
        x0_trainable_variables,
    ) = get_fully_connected_network_fn(
        parameter_shape=parameter_shape, input_shape=input_shape, **x0_kwargs
    )

    def parameter_fn(
        conditional_input: tf.Variable = None, **kwargs
    ) -> Callable[[tf.Variable, tf.Variable], tf.Variable]:
        made_net = masked_autoregressive_parameter_network_fn(
            conditional_input, **kwargs
        )
        x0_net = x0_parameter_network_fn(conditional_input, **kwargs)

        def call(x: tf.Variable, conditional_input: tf.Variable) -> tf.Variable:
            pv1 = made_net(x)
            pv2 = x0_net(conditional_input)
            return pv1 + pv2

        return call

    trainable_parameters = (
        masked_autoregressive_trainable_variables + x0_trainable_variables
    )
    return parameter_fn, trainable_parameters


def get_fully_connected_autoregressive_network_fn(
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
    name
        Name of the Keras model.
    dtype
        The dtype of the operation, by default tf.float32.
    kwargs
        Additional keyword arguments passed to Dense layers.


    Returns
    -------
        The parameter network model as a callable and a list of its trainable variables.

    """
    dims = parameter_shape[0]

    parameter_network = build_fully_connected_autoregressive_net(
        input_shape=[dims],
        output_shape=parameter_shape,
        dtype=dtype,
        name=name,
        **kwargs,
    )

    if conditional:
        assert (
            conditional_event_shape is not None
        ), "Conditional event shape must be provided if network is conditional."

        parameter_network = build_conditional_net(
            input_shape=[dims],
            conditional_event_shape=conditional_event_shape,
            output_shape=parameter_shape,
            parameter_net=parameter_network,
            conditioning_net_build_fn=build_fully_connected_net,
            name=name,
            **kwargs,
        )

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network([x, conditional_input], **kwargs)
    else:

        def parameter_network_fn(conditional_input, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables
