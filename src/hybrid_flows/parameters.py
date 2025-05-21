# %% Description ###############################################################
"""Functions defining ANNs.

The 'parameters' module contains functions for defining ANNs based on certain
parameters as used in various distribution models.

It includes functions for creating simple fully connected networks, parameter
vectors, and autoregressive parameter networks.

"""

# %% imports ###################################################################
from typing import Any, Callable, Dict, List, Tuple, Union

import tensorflow as tf
import tensorflow.keras as K
from bernstein_flow.bijectors import BernsteinPolynomial

from .nn import (
    build_conditional_net,
    build_fully_connected_autoregressive_net,
    build_fully_connected_net,
    build_fully_connected_res_net,
    build_masked_autoregressive_net,
)


# %% functions #################################################################
def get_parameter_vector_fn(
    parameter_shape: Tuple[int, ...],
    initializer: Callable[[Tuple[int, ...], ...], tf.Tensor] = tf.random.normal,
    dtype: tf.dtypes.DType = tf.float32,
) -> Tuple[Callable[..., tf.Variable], List[tf.Variable]]:
    """Create a TensorFlow parameter vector with a given shape and dtype.

    Parameters
    ----------
    parameter_shape : tuple of int
        The shape of the parameter vector.
    initializer : callable, optional
        Function returning initial parameters, by default tf.random.normal.
    dtype : tf.dtypes.DType, optional
        The data type of the parameter vector, by default tf.float32.

    Returns
    -------
    tuple
        A tuple with a callable that returns the parameter vector and a list containing
        the parameter vector itself.

    """
    # HACK: Initialize keras initializer class
    if isinstance(initializer, type):
        initializer = initializer()
    parameter_vector = tf.Variable(
        initializer(parameter_shape, dtype=dtype), trainable=True
    )
    return lambda *_, **__: parameter_vector, [parameter_vector], []


def get_fully_connected_network_fn(
    parameter_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Union[Tuple[int, ...], None] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Callable[..., tf.Tensor], List[tf.Variable]]:
    """Create a simple fully connected parameter network.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameter to output by the network.
    input_shape : tuple of int
        Shape of the input data.
    conditional : bool, optional
        If True, network is conditional, by default False.
    conditional_event_shape : tuple of int, optional
        Shape of additional conditional input, by default ().
    kwargs : dict, optional
        Additional keyword arguments for network configuration.

    Returns
    -------
    tuple
        A tuple of a callable that returns the parameter network and a list of its
        trainable variables.

    """
    parameter_network = build_fully_connected_net(
        input_shape=input_shape, output_shape=parameter_shape, **kwargs
    )
    parameter_network.build(input_shape)

    if conditional:
        assert conditional_event_shape is not None, (
            "Conditional event shape must be provided if network is conditional."
        )

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

        def parameter_network_fn(*_, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables, []


def get_parameter_vector_or_simple_network_fn(
    parameter_shape: Tuple[int, ...], conditional: bool, **kwargs: Dict[str, Any]
) -> Union[
    Tuple[K.Model, List[tf.Variable]],
    Tuple[Callable[..., tf.Variable], tf.Variable],
]:
    """Create either a parameter vector or a fully connected parameter network.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameter.
    conditional : bool
        If True, creates a conditional network; otherwise, creates a parameter vector.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    tuple
        A tuple consisting of either a Keras model and its trainable variables, or a
        callable parameter vector and the parameter vector itself.

    """
    if conditional:
        input_shape = kwargs.pop("conditional_event_shape")
        parameter_network = build_fully_connected_net(
            output_shape=parameter_shape, input_shape=input_shape, **kwargs
        )
        parameter_network.build(input_shape=input_shape)
        return parameter_network, parameter_network.trainable_variables, []
    else:
        return get_parameter_vector_fn(parameter_shape=parameter_shape, **kwargs)


def get_fully_connected_res_net_fn(
    parameter_shape: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Union[Tuple[int, ...], None] = None,
    name: str = "res_net",
    dtype: tf.dtypes.DType = tf.float32,
    **kwargs: Dict[str, Any],
) -> Tuple[Callable[..., K.Model], List[tf.Variable]]:
    """Generate a fully connected ResNet model.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameter vector.
    input_shape : tuple of int
        The shape of the input data.
    conditional : bool, optional
        If True, network is conditional, by default False.
    conditional_event_shape : tuple of int, optional
        Shape of additional conditional input, by default ().
    name : str, optional
        Name of the Keras model, by default "res_net".
    dtype : tf.dtypes.DType, optional
        The dtype of the operation, by default tf.float32.
    kwargs : dict, optional
        Additional keyword arguments passed to Dense layers.

    Returns
    -------
    tuple
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
        assert conditional_event_shape is not None, (
            "Conditional event shape must be provided if network is conditional."
        )

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

        def parameter_network_fn(*_, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables, []


def get_masked_autoregressive_network_fn(
    parameter_shape: Tuple[int, ...],
    conditional: bool = False,
    **kwargs: Dict[str, Any],
) -> Tuple[Callable[..., Callable[..., tf.Tensor]], List[tf.Variable]]:
    """Create an autoregressive parameter network.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameters.
    conditional : bool, optional
        If True, network is conditional, by default False.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    tuple
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

        def parameter_network_fn(*_, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables, []


def get_fully_connected_autoregressive_network_fn(
    parameter_shape: Tuple[int, ...],
    conditional: bool = False,
    conditional_event_shape: Union[Tuple[int, ...], None] = None,
    name: str = "autoregressive_res_net",
    dtype: tf.dtypes.DType = tf.float32,
    **kwargs: Dict[str, Any],
) -> Tuple[Callable[..., K.Model], List[tf.Variable]]:
    """Generate a fully connected autoregressive ResNet model.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameter vector.
    conditional : bool, optional
        If True, network is conditional, by default False.
    conditional_event_shape : tuple of int, optional
        Shape of additional conditional input, by default ().
    name : str, optional
        Name of the Keras model, by default "autoregressive_res_net".
    dtype : tf.dtypes.DType, optional
        The dtype of the operation, by default tf.float32.
    kwargs : dict, optional
        Additional keyword arguments passed to Dense layers.

    Returns
    -------
    tuple
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
        assert conditional_event_shape is not None, (
            "Conditional event shape must be provided if network is conditional."
        )

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

        def parameter_network_fn(*_, **kwargs):
            return lambda x: parameter_network(x, **kwargs)

    return parameter_network_fn, parameter_network.trainable_variables, []


def get_bernstein_polynomial_fn(
    parameter_shape: Tuple[int, ...],
    polynomial_order: int,
    conditional_event_shape: Tuple[int, ...],
    dtype: tf.dtypes.DType,
    initializer: Callable[[Tuple[int, ...], ...], tf.Tensor] = tf.random.normal,
    thetas_constrain_fn: Callable[[tf.Tensor], tf.Tensor] = tf.identity,
    **kwargs: Dict[str, Any],
) -> Tuple[Callable[[tf.Tensor], tf.Tensor], List[tf.Variable]]:
    """Create a Bernstein polynomial parameter lambda.

    Parameters
    ----------
    parameter_shape : tuple of int
        Shape of the parameter.
    polynomial_order : int
        Order of the polynomial.
    conditional_event_shape : tuple of int
        Shape of the conditional event.
    dtype : tf.dtypes.DType
        Data type of the parameter.
    initializer : callable, optional
        Function returning initial parameters, by default tf.random.normal.
    thetas_constrain_fn : callable, optional
        Function to constrain thetas, by default tf.identity.
    kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    tuple
        The parameter lambda and the parameter vector.

    """
    parameter_vector_shape = (
        [conditional_event_shape] + parameter_shape + [polynomial_order + 1]
    )
    parameter_vector = get_parameter_vector_fn(
        parameter_shape=parameter_vector_shape,
        dtype=dtype,
        initializer=initializer,
    )[1][0]

    def get_parameter_fn(conditional_input, **_):
        b_poly = BernsteinPolynomial(thetas_constrain_fn(parameter_vector), **kwargs)
        shape = [...] + [tf.newaxis for _ in range(len(parameter_shape))]
        # TODO: This is wring for conditional_event_shape != 1
        return b_poly(conditional_input[shape])

    return get_parameter_fn, [parameter_vector], []


def get_test_parameters_fn(input_shape, param_shape):
    """Test parameter function."""
    return lambda x: tf.ones([*input_shape, *param_shape]) * x[..., None], [], []


def get_test_parameters_nested_fn(input_shape, param_shape):
    """Test parameter function."""
    parameter_fn, vars, _ = get_test_parameters_fn(input_shape, param_shape)
    return lambda x: lambda xx: x * parameter_fn(xx), vars, []


def get_lu_parameters_fn(event_size, seed=None, dtype=tf.float32):
    event_size = tf.convert_to_tensor(
        event_size, dtype_hint=tf.int32, name="event_size"
    )
    random_matrix = tf.random.uniform(
        shape=[event_size, event_size], dtype=dtype, seed=seed
    )
    random_orthonormal = tf.linalg.qr(random_matrix)[0]
    lower_upper, permutation = tf.linalg.lu(random_orthonormal)
    lower_upper = tf.Variable(
        initial_value=lower_upper, trainable=True, name="lower_upper"
    )
    # Initialize a non-trainable variable for the permutation indices so
    # that its value isn't re-sampled from run-to-run.
    permutation = tf.Variable(
        initial_value=permutation, trainable=False, name="permutation"
    )
    return lambda *_, **__: [lower_upper, permutation], [lower_upper], [permutation]
