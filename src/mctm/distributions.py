# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2024-06-25 19:25:04 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
"""Functions for probability distributions.

The 'distributions' module provides functions for defining and parametrizing
probability distributions. They get used in the 'models' module.

The module defines a list of private base functions that get used to compose
the final model in many cases.
"""

import logging
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinBijector
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from mctm import parameters as parameters_lib

from .activations import get_spline_param_constrain_fn, get_thetas_constrain_fn
from .parameters import (
    get_fully_connected_network_fn,
    get_masked_autoregressive_network_fn,
    get_masked_autoregressive_network_with_additive_conditioner_fn,
    get_parameter_vector_or_simple_network_fn,
)
from .utils import getattr_from_module

# MODULE GLOBAL OBJECTS ########################################################
__LOGGER__ = logging.getLogger(__name__)


# FUNCTIONS ####################################################################
def _get_multivariate_normal_fn(
    dims: int,
) -> Tuple[Callable[[tf.Tensor], tfd.Distribution], Tuple[int, ...]]:
    """Get function to parametrize Multivariate Normal distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.

    Returns
    -------
    dist
        A function to parametrize the Multivariate Normal distribution.
    parameters_shape
        The shape of the parameter vector.

    """
    parameters_shape = (dims + np.sum(np.arange(dims + 1)),)

    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        loc = parameters[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, parameters_shape


def _get_trainable_distribution(
    dims: int,
    get_distribution_fn: Callable[..., Any],
    distribution_kwargs: Dict[str, Any],
    get_parameter_fn: Callable[..., Any],
    parameter_kwargs: Dict[str, Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    Tuple[int, ...],
]:
    """Get functions and variables to fit a distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    get_distribution_fn
        A function to get the distribution lambda.
    distribution_kwargs
        Keyword arguments for the distribution.
    get_parameter_fn
        A function to get the parameter lambda.
    parameter_kwargs
        Keyword arguments for the parameters.

    Returns
    -------
    distribution_fn
        A function to parametrize the distribution
    parameter_fn
        A function to obtain the parameters
    trainable_parameters
        List of trainable parameters

    """
    distribution_fn, parameters_shape = get_distribution_fn(
        dims=dims, **distribution_kwargs
    )
    parameter_fn, trainable_parameters = get_parameter_fn(
        parameters_shape, **parameter_kwargs
    )
    return distribution_fn, parameter_fn, trainable_parameters


def _get_base_distribution(
    dims: int = 0, distribution_name: str = "normal", **kwargs: Any
) -> tfd.Distribution:
    """Get the default base distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_name
        The type of distribution (e.g., "normal", "lognormal", "uniform",
                                  "kumaraswamy").
    kwargs
        Keyword arguments for the distribution.

    Returns
    -------
    dist
        The default base distribution.

    """
    if distribution_name == "normal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Normal(**default_kwargs)
    elif distribution_name == "truncated_normal":
        default_kwargs = dict(loc=0.0, scale=1.0, low=-4, high=4)
        default_kwargs.update(**kwargs)
        dist = tfd.TruncatedNormal(**default_kwargs)
    elif distribution_name == "lognormal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.LogNormal(**default_kwargs)
    elif distribution_name == "logistic":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Logistic(**default_kwargs)
    else:
        dist = getattr_from_module(distribution_name)(**kwargs)

    if dims:
        dist = tfd.Sample(dist, sample_shape=[dims])
    return dist


def _get_parametrized_bijector_fn(
    bijector_name: str,
    parameter_constrain_fn: Union[Callable[[tf.Tensor], tf.Tensor], None] = None,
    **kwargs: Any,
) -> Tuple[Callable[[tf.Tensor], tfb.Bijector], int]:
    """Get a function to parametrize a Bijector function and its parameter shape.

    Parameters
    ----------
    bijector_name
        Name of the bijector
    parameter_constrain_fn
        Function for constraining parameters
    kwargs
        Keyword arguments for the parameter constrain function.

    Returns
    -------
    bijector_fn
        Bijector parametrization function
    num_parameters
        Parameters shape

    """
    if bijector_name == "bernstein_poly":
        num_parameters = kwargs.pop("order")
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_thetas_constrain_fn(**kwargs)

        def bijector_fn(unconstrained_parameters: tf.Tensor) -> tfb.Bijector:
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = BernsteinBijector(constrained_parameters, name=bijector_name)

            return bijector

    elif bijector_name == "quadratic_spline":
        num_parameters = kwargs["nbins"] * 3 - 1
        range_min = kwargs.pop("range_min")
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_spline_param_constrain_fn(**kwargs)

        def bijector_fn(unconstrained_parameters: tf.Tensor) -> tfb.Bijector:
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = tfb.RationalQuadraticSpline(
                **constrained_parameters, range_min=range_min, name=bijector_name
            )

            return bijector

    else:
        raise ValueError(f"Unknown bijector type: {bijector_name}")

    return bijector_fn, num_parameters


def _get_flow_parametrization_fn(
    scale: Union[bool, tf.Tensor],
    shift: Union[bool, tf.Tensor],
    bijector_name: str,
    get_parametrized_bijector_fn: Callable[
        [str, Union[Callable[[tf.Tensor], tf.Tensor], None], Any],
        Tuple[Callable[[tf.Tensor], tfb.Bijector], int],
    ] = _get_parametrized_bijector_fn,
    **kwargs: Any,
) -> Tuple[Callable[[tf.Tensor], tfb.Bijector], int]:
    """Get function to parametrize a normalizing flow.

    Parameters
    ----------
    scale
        The scale of the flow.
    shift
        The shift of the flow.
    bijector_name
        A string that defines the name of the bijector
    get_parametrized_bijector_fn
        An optional function that generates the parameterized bijector function
    kwargs
        Additional keyword arguments.

    Returns
    -------
    flow_parametrization_fn
        The flow parametrization function
    num_parameters
        Its parameter shape.

    """
    (
        bijector_fn,
        num_parameters,
    ) = get_parametrized_bijector_fn(bijector_name, **kwargs)

    learnable_scale = scale is True
    if learnable_scale:
        num_parameters += 1

    learnable_shift = shift is True
    if learnable_shift:
        num_parameters += 1

    def get_parameters(
        unconstrained_parameters: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nonlocal scale, shift
        if learnable_scale:
            scale = tf.abs(unconstrained_parameters[..., 0]) + 0.01
            unconstrained_parameters = unconstrained_parameters[..., 1:]

        if learnable_shift:
            shift = unconstrained_parameters[..., 0]
            unconstrained_parameters = unconstrained_parameters[..., 1:]
        return scale, shift, unconstrained_parameters

    def flow_parametrization_fn(unconstrained_parameters: tf.Tensor) -> tfb.Bijector:
        scale, shift, unconstrained_parameters = get_parameters(
            unconstrained_parameters
        )

        bijectors = []

        # ŷ = a1(x)*(y + b1(x)) = f2(f1(y,x),x)
        # f1 = y + b1(x)
        if shift is not False:
            shift_bj = tfb.Shift(
                shift,
                name="shift",
            )
            bijectors.append(shift_bj)

        # f2 = a1(x) * y
        if scale is not False:
            scale_bj = tfb.Scale(
                scale,
                name="scale",
            )
            bijectors.append(scale_bj)

        # f3: Flexible Transformation Function
        bijectors.append(bijector_fn(unconstrained_parameters))

        if len(bijectors) == 1:
            flow = bijectors[0]
        else:
            # the Chain bijector uses reversed list in the forward call.
            # We change the direction by first reversing the list to get f₃ ∘ f₂ ∘ f₁
            # and then invert it to get T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂
            bijectors = list(reversed(bijectors))
            flow = tfb.Chain(bijectors)

        return tfb.Invert(flow)

    return flow_parametrization_fn, num_parameters


def _get_transformed_distribution_fn(
    flow_parametrization_fn: Callable[[tf.Tensor], tfb.Bijector],
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    **kwargs: Any,
) -> Callable[[tf.Tensor], tfd.TransformedDistribution]:
    """Get function to parametrize a transformed distribution.

    Parameters
    ----------
    flow_parametrization_fn
        The flow parametrization function.
    get_base_distribution
        Function that returns base distribution if provided;
        otherwise, use default base distribution.
    **kwargs
        Additional keyword parameters.

    Returns
    -------
    distribution_fn
        The transformed distribution function.

    """

    def distribution_fn(parameters: tf.Tensor) -> tfd.TransformedDistribution:
        if isinstance(parameters, tuple) and len(parameters) == 2:
            parameters, base_parameters = parameters
            kwargs.update(base_parameters)
            __LOGGER__.debug("got parameters for base distribution.")

        __LOGGER__.debug("base distribution kwargs: %s", str(kwargs))
        base_distribution = get_base_distribution(**kwargs)
        bijector = flow_parametrization_fn(parameters)
        return tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=bijector,
        )

    return distribution_fn


def _get_elementwise_flow(
    dims: int,
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    **kwargs: Any,
) -> Tuple[Callable[[tf.Tensor], tfd.Distribution], Tuple[int, ...]]:
    """Get a function to parametrize a elementwise transformed distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    get_base_distribution
        The base distribution lambda.
    base_distribution_kwargs
        Keyword arguments for the base distribution.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    distribution_fn
        The parametrization function of the transformed distribution
    pv_shape
        Its parameter shape.

    """
    flow_parametrization_fn, num_parameters = _get_flow_parametrization_fn(**kwargs)
    pv_shape = (dims, num_parameters)

    distribution_fn = _get_transformed_distribution_fn(
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        dims=0,
        **base_distribution_kwargs,
    )

    return distribution_fn, pv_shape


def _get_multivariate_flow_fn(
    dims: int, **kwargs: Any
) -> Tuple[Callable[[tf.Tensor], tfd.Distribution], Tuple[int, ...]]:
    """Get a function to parametrize a multivariate normalizing flow distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    distribution_fn
        A function to parametrize the multivariate normalizing flow distribution
    pv_shape
        The shape of the parameter vector.

    """
    flow_parametrization_fn, num_parameters = _get_flow_parametrization_fn(**kwargs)
    pv_shape = (num_parameters * dims + np.sum(np.arange(dims + 1)),)

    def distribution_fn(parameters: tf.Tensor) -> tfd.Distribution:
        bs = prefer_static.shape(parameters)[:-1]
        shape = tf.concat((bs, [dims, num_parameters]), 0)

        unconstrained_parameters = tf.reshape(
            parameters[..., : num_parameters * dims], shape
        )
        scale_tril = tfp.bijectors.FillScaleTriL()(
            parameters[..., num_parameters * dims :]
        )

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=flow_parametrization_fn(
                unconstrained_parameters,
            ),
        )

    return distribution_fn, pv_shape


def _get_stacked_flow_parametrization_fn(
    dims: int,
    num_layers: int,
    get_parameter_fn_for_layer: Callable[[int, int], Any],
    get_bijectors_for_layer: Callable[
        [int, Callable[..., Any], Callable[..., Any]], Any
    ],
    **kwargs: Any,
) -> Tuple[
    Callable[[List[Callable[..., Any]]], tfb.Bijector],
    Callable[[tf.Tensor, Any], Any],
    List[tf.Variable],
]:
    """Get function to parametrize a stacked normalizing flow.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    num_layers
        The number of layers in the flow.
    get_parameter_fn_for_layer
        Function that returns parameter function for each layer.
    get_bijectors_for_layer
        Function that returns bijector function for each layer.
    kwargs
        Keyword arguments.

    Returns
    -------
    stacked_flow_parametrization_fn
        Parametrization function
    parameter_fn
        Parameter function
    trainable_variables
        Trainable variables

    """
    flow_kwargs = kwargs.copy()

    # scale and shift have to be applied on all dimensions before the first
    # coupling layers so let's remove them from the kwargs to skip them in the
    # following calls of __get_bijector_fn__
    scale = flow_kwargs.pop("scale", False)
    shift = flow_kwargs.pop("shift", False)
    scale_to_domain = flow_kwargs.pop("scale_to_domain", False)

    flow_parametrization_fns: List[
        Callable[[Union[tf.Tensor, bool]], tfb.Bijector]
    ] = []
    parameter_networks: List[Callable[..., Any]] = []
    trainable_variables: List[tf.Variable] = []
    for layer in range(num_layers):
        if layer > 0:
            if scale_to_domain:
                low = flow_kwargs["low"]
                high = flow_kwargs["high"]
                flow_kwargs.update(scale=1 / (high - low), shift=-low)
            else:
                flow_kwargs.update(scale=False, shift=False)
        else:
            flow_kwargs.update(scale=scale, shift=shift)

        flow_parametrization_fn, num_parameters = _get_flow_parametrization_fn(
            **flow_kwargs
        )
        flow_parametrization_fns.append(flow_parametrization_fn)

        network, variables = get_parameter_fn_for_layer(layer, num_parameters)
        parameter_networks.append(network)
        trainable_variables.append(variables)

    def parameter_fn(conditional_input: tf.Tensor, **kwargs):
        return list(
            map(lambda net: net(conditional_input, **kwargs), parameter_networks)
        )

    def stacked_flow_parametrization_fn(
        parameter_networks: List[Callable[..., Any]],
    ) -> tfb.Bijector:
        bijectors = []

        # Stack bijectors
        for layer, (parameter_network, flow_parametrization_fn) in enumerate(
            zip(parameter_networks, flow_parametrization_fns)
        ):
            bijectors.extend(
                get_bijectors_for_layer(
                    layer, parameter_network, flow_parametrization_fn
                )
            )

        chain = tfb.Chain(bijectors)
        return chain

    return (
        stacked_flow_parametrization_fn,
        parameter_fn,
        list(chain.from_iterable(trainable_variables)),
    )


def _get_stacked_flow(
    dims: int,
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    **kwargs: Any,
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
]:
    """Get a function to parametrize a stacked normalizing flow distribution.

    Parameters
    ----------
    dims
        event dimensions of distribution.
    get_base_distribution
        Callable to retrieve the base distribution.
    base_distribution_kwargs
        Keyword arguments passed to get_base_distribution.
    kwargs
        Keyword arguments passed to `_get_stacked_flow`.

    Returns
    -------
    distribution_fn
        Parametrization function
    parameter_fn
        Parameter function
    trainable_variables
        Trainable variables

    """
    (
        flow_parametrization_fn,
        parameter_fn,
        trainable_variables,
    ) = _get_stacked_flow_parametrization_fn(dims, **kwargs)

    distribution_fn = _get_transformed_distribution_fn(
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, trainable_variables


def _get_num_masked(dims: int, layer: int) -> int:
    """Compute the number of masked dimensions.

    Parameters
    ----------
    dims
        The total number of dimensions.
    layer
        The layer number.

    Returns
    -------
    num_masked
        The number of masked dimensions.

    """
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


def _get_bijector_fn(
    network: Callable[[tf.Tensor, Any], tf.Tensor],
    flow_parametrization_fn: Callable[[tf.Tensor], tfb.Bijector],
) -> Callable[[tf.Tensor, Any], tfb.Bijector]:
    """Get a bijector function as a callable.

    Parameters
    ----------
    network
        The network to use for the bijector function.
    flow_parametrization_fn
        Function initializing the flow from unconstrained parameters.

    Returns
    -------
    bijector_fn
        A function to parametrize the bijector function.

    """

    def bijector_fn(y: tf.Tensor, *args: Any, **kwargs: Any) -> tfb.Bijector:
        with tf.name_scope("bnf_bjector"):
            pvector = network(y, **kwargs)
            flow = flow_parametrization_fn(pvector)

        return flow

    return bijector_fn


# PUBLIC FUNCTIONS #############################################################
def get_coupling_flow(
    dims: int,
    distribution_kwargs: Dict[str, Any],
    parameter_kwargs: Dict[str, Any],
    num_masked: Union[int, None] = None,
    get_bijector_fn: Callable[
        [Callable[[tf.Tensor, Any], tf.Tensor], Callable[..., Any]],
        Callable[..., Any],
    ] = _get_bijector_fn,
    get_parameter_fn: Callable[..., Any] = get_fully_connected_network_fn,
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
]:
    """Get a Coupling Flow distribution as a callable.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_kwargs
        Keyword arguments for the distribution.
    parameter_kwargs
        Keyword arguments for the parameters.
    num_masked
        Number of dimensions to mask.
    get_bijector_fn
        A function to get the bijector.
    get_parameter_fn
        A function to get the parameter.

    Returns
    -------
    distribution_fn
        A function to parametrize the distribution
    parameter_fn
        A callable for parameter networks
    trainable_parameters
        A list of trainable parameters.

    """
    distribution_kwargs = distribution_kwargs.copy()
    num_layers = distribution_kwargs.pop("num_layers", 1)
    get_base_distribution = distribution_kwargs.pop(
        "get_base_distribution", _get_base_distribution
    )
    base_distribution_kwargs = distribution_kwargs.pop("base_distribution_kwargs", {})

    def get_parameter_fn_for_layer(layer: int, num_parameters: int):
        nm = num_masked if num_masked is not None else _get_num_masked(dims, layer)
        parameter_shape = (dims - nm, num_parameters)
        return get_parameter_fn(
            input_shape=(nm,),
            parameter_shape=parameter_shape,
            **parameter_kwargs,
        )

    def get_bijectors_for_layer(
        layer: int,
        network: Callable[[tf.Tensor, Any], tf.Tensor],
        flow_parametrization_fn: Callable[[tf.Tensor], tfb.Bijector],
    ) -> List[tfb.Bijector]:
        nm = num_masked if num_masked is not None else _get_num_masked(dims, layer)

        bijectors = []
        bijectors.append(
            tfb.RealNVP(
                num_masked=nm,
                bijector_fn=get_bijector_fn(network, flow_parametrization_fn),
            )
        )

        permutation = list(range(nm, dims)) + list(range(nm))
        if num_layers % 2 != 0 and layer == (num_layers - 1):
            print("uneven number of coupling layers -> skipping last permutation")
        else:
            bijectors.append(tfb.Permute(permutation=permutation))

        return bijectors

    return _get_stacked_flow(
        dims=dims,
        num_layers=num_layers,
        get_parameter_fn_for_layer=get_parameter_fn_for_layer,
        get_bijectors_for_layer=get_bijectors_for_layer,
        get_base_distribution=get_base_distribution,
        base_distribution_kwargs=base_distribution_kwargs,
        **distribution_kwargs,
    )


def _get_masked_autoregressive_flow_parametrization_fn(
    dims: int,
    distribution_kwargs: Dict[str, Any],
    parameter_kwargs: Dict[str, Any],
    get_bijector_fn: Callable[
        [Callable[[tf.Tensor, Any], tf.Tensor], Callable[..., Any]],
        Callable[..., Any],
    ],
    get_parameter_fn: Callable[..., Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
]:
    """Get a Masked Autoregressive Flow distribution as a callable.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_kwargs
        Keyword arguments for the distribution.
    parameter_kwargs
        Keyword arguments for the parameters.
    get_bijector_fn
        A function to get the bijector.
    get_parameter_fn
        A function to get the parameter.

    Returns
    -------
    distribution_fn
        A function to parametrize the distribution
    parameter_fn
        A callable for parameter networks
    trainable_parameters
        A list of trainable parameters.

    """

    def get_parameter_fn_for_layer(_: int, num_parameters: int):
        parameter_shape = (dims, num_parameters)
        return get_parameter_fn(parameter_shape, **parameter_kwargs)

    def get_bijectors_for_layer(
        _: int,
        network: Callable[[tf.Tensor, Any], tf.Tensor],
        flow_parametrization_fn: Callable[[tf.Tensor], tfb.Bijector],
    ) -> List[tfb.MaskedAutoregressiveFlow]:
        bijector_fn = get_bijector_fn(
            network=network,
            flow_parametrization_fn=flow_parametrization_fn,
        )

        return [tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn)]

    return _get_stacked_flow_parametrization_fn(
        dims=dims,
        get_parameter_fn_for_layer=get_parameter_fn_for_layer,
        get_bijectors_for_layer=get_bijectors_for_layer,
        **distribution_kwargs,
    )


def get_masked_autoregressive_flow(
    dims: int,
    distribution_kwargs: Dict[str, Any],
    parameter_kwargs: Dict[str, Any],
    get_bijector_fn: Callable[
        [Callable[[tf.Tensor, Any], tf.Tensor], Callable[..., Any]],
        Callable[..., Any],
    ] = _get_bijector_fn,
    get_parameter_fn: Callable[..., Any] = get_masked_autoregressive_network_fn,
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
]:
    """Get a Masked Autoregressive Flow distribution as a callable.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_kwargs
        Keyword arguments for the distribution.
    parameter_kwargs
        Keyword arguments for the parameters.
    get_bijector_fn
        A function to get the bijector.
    get_parameter_fn
        A function to get the parameter.

    Returns
    -------
    distribution_fn
        A function to parametrize the distribution
    parameter_fn
        A callable for parameter networks
    trainable_parameters
        A list of trainable parameters.

    """
    distribution_kwargs = distribution_kwargs.copy()
    get_base_distribution = distribution_kwargs.pop(
        "get_base_distribution", _get_base_distribution
    )
    base_distribution_kwargs = distribution_kwargs.pop("base_distribution_kwargs", {})
    if "dims" not in base_distribution_kwargs:
        base_distribution_kwargs["dims"] = dims
    print(base_distribution_kwargs)

    (
        flow_parametrization_fn,
        parameter_fn,
        trainable_variables,
    ) = _get_masked_autoregressive_flow_parametrization_fn(
        dims=dims,
        distribution_kwargs=distribution_kwargs,
        parameter_kwargs=parameter_kwargs,
        get_bijector_fn=get_bijector_fn,
        get_parameter_fn=get_parameter_fn,
    )
    distribution_fn = _get_transformed_distribution_fn(
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, trainable_variables


def get_masked_autoregressive_flow_first_dim_masked(
    dims: int,
    distribution_kwargs: Dict[str, Any],
    parameter_kwargs: Dict[str, Any],
    get_bijector_fn: Callable[
        [Callable[[tf.Tensor, Any], tf.Tensor], Callable[..., Any]],
        Callable[..., Any],
    ] = _get_bijector_fn,
    get_parameter_fn: Callable[
        ..., Any
    ] = get_masked_autoregressive_network_with_additive_conditioner_fn,  # noqa: E501
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
]:
    """Get a Masked Autoregressive Bernstein Flow with the first dimension masked.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_kwargs
        Keyword arguments for the distribution.
    parameter_kwargs
        Keyword arguments for the parameters.
    get_bijector_fn
        A function to get the bijector.
    get_parameter_fn
        A function to get the parameters.

    Returns
    -------
        A function to parametrize the Masked Autoregressive Bernstein Flow
        distribution with the first dimension masked, a callable for parameter networks,
        and a list of trainable parameters.

    """
    distribution_kwargs = distribution_kwargs.copy()
    get_base_distribution = distribution_kwargs.pop(
        "get_base_distribution", _get_base_distribution
    )
    base_distribution_kwargs = distribution_kwargs.pop("base_distribution_kwargs", {})

    (
        flow_parametrization_fn,
        parameter_fn,
        trainable_variables,
    ) = _get_masked_autoregressive_flow_parametrization_fn(
        dims=dims - 1,
        distribution_kwargs=distribution_kwargs,
        parameter_kwargs={"input_shape": (1,), **parameter_kwargs},
        get_bijector_fn=get_bijector_fn,
        get_parameter_fn=get_parameter_fn,
    )

    def masked_flow_parametrization_fn(
        parameter_networks: List[Callable[[tf.Tensor, Any], tf.Tensor]],
    ) -> tfb.RealNVP:
        def bijector_fn(x0: tf.Tensor, *arg: Any, **kwargs: Any) -> tfb.Bijector:
            conditioned_parameter_networks = list(
                map(lambda net: partial(net, conditional_input=x0), parameter_networks)
            )
            stacked_maf_bijector = flow_parametrization_fn(
                conditioned_parameter_networks
            )
            return stacked_maf_bijector

        return tfb.RealNVP(num_masked=1, bijector_fn=bijector_fn)

    distribution_fn = _get_transformed_distribution_fn(
        masked_flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, trainable_variables


def _get_parametrized_bijector(
    bijector: Union[tfb.Bijector, type, str],
    parameters: tf.Tensor,
    parameters_constraint_fn: Callable = None,
    invert: bool = False,
    bijector_kwargs: Dict[str, Any] = {},
) -> tfp.bijectors.Bijector:
    """Get a parametrized bijector instance.

    Parameters
    ----------
    bijector
        Bijectro class, instance or name as string.
    parameters
        Parameters to pass to the bijector.
    parameters_constraint_fn
        Function to constrain the parameters.
    invert
        If `True` the parametrized bijector gets inverted, default: False
    bijector_kwargs
        Additional keyword arguments to pass to the bijector.

    Returns
    -------
        The parametrized bijector instance.

    """
    # import bijector class
    if not isinstance(bijector, tfb.Bijector):
        if isinstance(bijector, type):
            bijector_cls = bijector
        elif bijector == "BernsteinBijector":
            bijector_cls = BernsteinBijector
        elif "." not in bijector:
            bijector_cls = getattr_from_module("tfb." + bijector)
        else:
            bijector_cls = getattr_from_module(bijector)

        # instantiate bijector
        if (
            bijector_cls in (tfb.MaskedAutoregressiveFlow, tfb.RealNVP)
            and "bijector" in bijector_kwargs
        ):
            bijector_kwargs = bijector_kwargs.copy()
            nested_bijector = bijector_kwargs.pop("bijector")
            nested_bijector_kwargs = bijector_kwargs.pop("bijector_kwargs", {})

            def bijector_fn(y, *args, **kwargs):
                pvector = parameters(y, **kwargs)
                bijector = _get_parametrized_bijector(
                    bijector=nested_bijector,
                    parameters=pvector,
                    parameters_constraint_fn=parameters_constraint_fn,
                    bijector_kwargs=nested_bijector_kwargs,
                )

                return bijector

            bijector = bijector_cls(bijector_fn=bijector_fn, **bijector_kwargs)
        elif parameters_constraint_fn:
            constrained_parameters = parameters_constraint_fn(parameters)
            if isinstance(constrained_parameters, list):
                bijector = bijector_cls(*constrained_parameters, **bijector_kwargs)
            elif isinstance(constrained_parameters, dict):
                bijector = bijector_cls(**constrained_parameters, **bijector_kwargs)
            else:
                bijector = bijector_cls(constrained_parameters, **bijector_kwargs)
        else:
            bijector = bijector_cls(parameters, **bijector_kwargs)

    if invert:
        return tfb.Invert(bijector)
    else:
        return bijector


def _get_parameter_fn(
    parameters: tf.Tensor = None,
    parameters_fn: Callable = None,
    **parameters_fn_kwargs,
) -> Tuple[Callable, Dict]:
    """Get a parameter function and its trainable parameters.

    Parameters
    ----------
    parameters
        Parameters to pass to the parameter function.
    parameters_fn
        Function to create the parameters.
    parameters_fn_kwargs
        Keyword arguments to pass to the parameter function.

    Returns
    -------
        The parameter function and its trainable parameters.

    Raises
    ------
    ValueError
        If both or none of `parameters` and `parameters_fn` are provided.

    """
    if parameters is not None and parameters_fn is not None:
        raise ValueError(
            "The arguments 'parameters' and 'parameter_fn' are "
            "mutually exclusive. Only provide either one of them"
        )
    elif parameters is None and parameters_fn is None:
        raise ValueError(
            "Either 'parameters' or 'parameter_fn' " "have to be provided."
        )

    # get parameter fn
    if parameters:
        __LOGGER__.debug("got constant parameters %s", str(parameters))

        def parameter_fn(*_, **__):  # noqa: E731
            return parameters

    else:
        if callable(parameters_fn):
            __LOGGER__.debug("using provided callable as parameter function")
            get_parameter_fn = parameters_fn
        else:
            __LOGGER__.debug("parameter function: %s", parameters_fn)
            get_parameter_fn = getattr(parameters_lib, f"get_{parameters_fn}_fn")

        __LOGGER__.debug("initializing parametrization function")
        parameter_fn, trainable_parameters = get_parameter_fn(
            **parameters_fn_kwargs,
        )

    return parameter_fn, trainable_parameters


def _get_parameters_constraint_fn(
    parameters_constraint_fn: Callable,
    **parameters_constraint_fn_kwargs,
) -> Callable:
    """Get a parameters constraint function.

    Parameters
    ----------
    parameters_constraint_fn
        The parameters constraint function.
    parameters_constraint_fn_kwargs
        Keyword arguments to pass to the parameters constraint function.

    Returns
    -------
        The parameters constraint function.

    """
    if callable(parameters_constraint_fn):
        __LOGGER__.debug("using provided callable as parameter constraint function")
    else:
        __LOGGER__.debug(
            "importing parameter constraint function '%s'", parameters_constraint_fn
        )
        parameters_constraint_fn = getattr_from_module(parameters_constraint_fn)

    if len(parameters_constraint_fn_kwargs) > 0 or isinstance(
        parameters_constraint_fn, type
    ):
        parameters_constraint_fn = parameters_constraint_fn(
            **parameters_constraint_fn_kwargs
        )

    return parameters_constraint_fn


def get_normalizing_flow(
    bijectors: List[Dict],
    reverse_flow: bool = True,
    inverse_flow: bool = True,
    get_base_distribution: Callable = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    default_parameters_constraint_fn: Callable[
        [tf.Tensor], Union[Dict[str, tf.Tensor], List[tf.Tensor], tf.Tensor]
    ] = tf.identity,
) -> Tuple[Callable, Callable, List]:
    """Get a function to parametrize a elementwise transformed distribution.

    Parameters
    ----------
    bijectors
        List of dictionaries describing bijective transformations.
    reverse_flow:
        Reverse chain of bijectors.
    inverse_flow
        Invert flow to transform from the data to the base distribution.
    get_base_distribution
        The base distribution lambda.
    base_distribution_kwargs
        Keyword arguments for the base distribution.
    default_parameters_constraint_fn
        Default constraining function to use, if not provided. Default: `tf.identity`

    Returns
    -------
        The parametrization function of the transformed distribution,
        the parameter function and a list of trainable parameters.

    """
    bijector_name_key = "bijector"
    bijector_kwargs_key = "bijector_kwargs"
    parameters_key = "parameters"
    parameters_fn_key = "parameters_fn"
    parameters_fn_kwargs_key = "parameters_fn_kwargs"
    parameters_constraint_fn_key = "parameters_constraint_fn"
    parameters_constraint_fn_kwargs_key = "parameters_constraint_fn_kwargs"

    parameter_fns = {}
    parameters_constraint_fns = {}
    trainable_parameters = {}

    for bijector in bijectors:
        bijector_name = bijector[bijector_name_key]
        __LOGGER__.debug("processing definition of bijector %s", bijector_name)
        (
            parameter_fns[bijector_name],
            trainable_parameters[bijector_name],
        ) = _get_parameter_fn(
            parameters=bijector.pop(parameters_key, None),
            parameters_fn=bijector.pop(parameters_fn_key, None),
            **bijector.pop(parameters_fn_kwargs_key, {}),
        )
        if parameters_constraint_fn_key in bijector:
            parameters_constraint_fns[bijector_name] = _get_parameters_constraint_fn(
                parameters_constraint_fn=bijector.pop(
                    parameters_constraint_fn_key, None
                ),
                **bijector.pop(parameters_constraint_fn_kwargs_key, {}),
            )
        else:
            parameters_constraint_fns[bijector_name] = default_parameters_constraint_fn

    __LOGGER__.debug("parameter function %s", str(parameter_fns))

    if (
        parameters_key in base_distribution_kwargs
        or parameters_fn_key in base_distribution_kwargs
    ):
        (
            base_distribution_parameter_fn,
            base_distribution_trainable_parameters,
        ) = _get_parameter_fn(
            parameters=base_distribution_kwargs.pop(parameters_key, None),
            parameters_fn=base_distribution_kwargs.pop(parameters_fn_key, None),
            **base_distribution_kwargs.pop(parameters_fn_kwargs_key, {}),
        )
        trainable_parameters["base_distribution"] = (
            base_distribution_trainable_parameters
        )
        if parameters_constraint_fn_key in base_distribution_kwargs:
            base_distribution_parameter_constraint_fns = _get_parameters_constraint_fn(
                parameters_constraint_fn=base_distribution_kwargs.pop(
                    parameters_constraint_fn_key, None
                ),
                **base_distribution_kwargs.pop(parameters_constraint_fn_kwargs_key, {}),
            )
        else:
            base_distribution_parameter_constraint_fns = (
                default_parameters_constraint_fn
            )
    else:
        base_distribution_parameter_fn = None

    def parameter_fn(*args, **kwargs):
        params = [fn(*args, **kwargs) for fn in parameter_fns.values()]
        if base_distribution_parameter_fn:
            base_params = base_distribution_parameter_constraint_fns(
                base_distribution_parameter_fn(*args, **kwargs)
            )
            return params, base_params
        else:
            return params

    def flow_parametrization_fn(all_parameters):
        bijectors_list = []
        for bijector, bijector_parameters in zip(bijectors, all_parameters):
            bijector_name = bijector[bijector_name_key]
            bijector_kwargs = bijector.get(bijector_kwargs_key, {})
            bijectors_list.append(
                _get_parametrized_bijector(
                    bijector=bijector_name,
                    parameters=bijector_parameters,
                    parameters_constraint_fn=parameters_constraint_fns[bijector_name],
                    bijector_kwargs=bijector_kwargs,
                )
            )

        if reverse_flow:
            # The Chain bijector uses the reversed list in the forward call.
            # We change the direction here to get T = f₃ ∘ f₂ ∘ f₁.
            bijectors_list = list(reversed(bijectors_list))

        if len(bijectors_list) == 1:
            flow = bijectors_list[0]
        else:
            flow = tfb.Chain(bijectors_list)

        if inverse_flow:
            # If we invert the reversed flow we get
            # T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂.
            flow = tfb.Invert(flow)

        return flow

    distribution_fn = _get_transformed_distribution_fn(
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, list(trainable_parameters.values())


# actual functions that are composed of base functions
get_elementwise_flow = partial(
    _get_trainable_distribution,
    get_distribution_fn=_get_elementwise_flow,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
get_multivariate_flow = partial(
    _get_trainable_distribution,
    get_distribution_fn=_get_multivariate_flow_fn,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
get_multivariate_normal = partial(
    _get_trainable_distribution,
    get_distribution_fn=_get_multivariate_normal_fn,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
