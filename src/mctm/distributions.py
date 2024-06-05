# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2024-06-05 19:02:24 (Marcel Arpogaus)
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

import importlib
import logging
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinBijector
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from mctm import parameters

from .activations import get_spline_param_constrain_fn, get_thetas_constrain_fn
from .parameters import (
    get_fully_connected_network_fn,
    get_masked_autoregressive_network_fn,
    get_masked_autoregressive_network_with_additive_conditioner_fn,
    get_parameter_vector_or_simple_network_fn,
)

# MODULE GLOBAL OBJECTS ########################################################
__LOGGER__ = logging.getLogger(__name__)


# FUNCTIONS ####################################################################
def _get_multivariate_normal_fn(
    dims: int,
) -> Tuple[Callable[tf.Variable, tfd.Distribution], Tuple[int, ...]]:
    """Get function to parametrize Multivariate Normal distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.

    Returns
    -------
    tuple
        A function to parametrize the Multivariate Normal distribution and the shape
        of the parameter vector.

    """
    parameters_shape = [dims + np.sum(np.arange(dims + 1))]

    def dist(parameters):
        loc = parameters[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, parameters_shape


def _get_trainable_distribution(
    dims: int,
    get_distribution_fn: Callable,
    distribution_kwargs: Dict,
    get_parameter_fn: Callable,
    parameter_kwargs: Dict,
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
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
        A function to parametrize the distribution, a function to obtain the parameters
        and list of trainable parameters.

    """
    distribution_fn, parameters_shape = get_distribution_fn(
        dims=dims, **distribution_kwargs
    )
    parameter_fn, trainable_parameters = get_parameter_fn(
        parameters_shape, **parameter_kwargs
    )
    return distribution_fn, parameter_fn, trainable_parameters


def _get_base_distribution(
    dims: int, distribution_type: str = "normal", is_joint: bool = True, **kwargs
) -> tfd.Distribution:
    """Get the default base distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    distribution_type
        The type of distribution (e.g., "normal", "lognormal", "uniform",
                                  "kumaraswamy").
    is_joint
        If set to True, return the Joint Distribution.
    kwargs
        Keyword arguments for the distribution.

    Returns
    -------
        The default base distribution.

    """
    if distribution_type == "normal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Normal(**default_kwargs)
    elif distribution_type == "truncated_normal":
        default_kwargs = dict(loc=0.0, scale=1.0, low=-4, high=4)
        default_kwargs.update(**kwargs)
        dist = tfd.TruncatedNormal(**default_kwargs)
    elif distribution_type == "lognormal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.LogNormal(**default_kwargs)
    elif distribution_type == "logistic":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Logistic(**default_kwargs)
    elif distribution_type == "uniform":
        dist = tfd.Uniform(**kwargs)
    elif distribution_type == "kumaraswamy":
        dist = tfd.Kumaraswamy(**kwargs)
    else:
        raise ValueError(f"Unsupported distribution type {distribution_type}.")
    if is_joint:
        dist = tfd.Sample(dist, sample_shape=[dims])
    return dist


def _get_parametrized_bijector_fn(
    bijector_name: str, parameter_constrain_fn=None, **kwargs
) -> Tuple[Callable[tf.Variable, tfb.Bijector], int]:
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
        Bijector parametrization function and parameters shape

    """
    if bijector_name == "bernstein_poly":
        num_parameters = kwargs.pop("order")
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_thetas_constrain_fn(**kwargs)

        def bijector_fn(unconstrained_parameters):
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = BernsteinBijector(constrained_parameters, name=bijector_name)

            return bijector

    elif bijector_name == "quadratic_spline":
        num_parameters = kwargs["nbins"] * 3 - 1
        range_min = kwargs.pop("range_min")
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_spline_param_constrain_fn(**kwargs)

        def bijector_fn(unconstrained_parameters):
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = tfb.RationalQuadraticSpline(
                **constrained_parameters, range_min=range_min, name=bijector_name
            )

            return bijector

    else:
        raise ValueError(f"Unknown bijector type: {bijector_name}")

    return bijector_fn, num_parameters


def _get_flow_parametrization_fn(
    scale: bool,
    shift: bool,
    bijector_name: str,
    get_parametrized_bijector_fn=_get_parametrized_bijector_fn,
    **kwargs,
) -> Tuple[Callable[tf.Variable, tfb.Bijector], List[int]]:
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
        The flow parametrization function and its parameter shape.

    """
    (
        bijector_fn,
        num_parameters,
    ) = get_parametrized_bijector_fn(bijector_name, **kwargs)

    learnable_scale = scale and isinstance(scale, bool)
    if learnable_scale:
        num_parameters += 1

    learnable_shift = shift and isinstance(shift, bool)
    if learnable_shift:
        num_parameters += 1

    def get_parameters(unconstrained_parameters):
        nonlocal scale, shift
        if learnable_scale:
            scale = tf.abs(unconstrained_parameters[..., 0]) + 0.01
            unconstrained_parameters = unconstrained_parameters[..., 1:]

        if learnable_shift:
            shift = unconstrained_parameters[..., 0]
            unconstrained_parameters = unconstrained_parameters[..., 1:]
        return scale, shift, unconstrained_parameters

    def flow_parametrization_fn(unconstrained_parameters):
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
    dims: int,
    flow_parametrization_fn: Callable,
    get_base_distribution=_get_base_distribution,
    **kwargs,
) -> Callable[tf.Variable, tfd.TransformedDistribution]:
    """Get function to parametrize a transformed distribution.

    Parameters
    ----------
    dims
        The dimensions of the distribution.
    flow_parametrization_fn
        The flow parametrization function.
    get_base_distribution
        Function that returns base distribution if provided;
        otherwise, use default base distribution.
    **kwargs : Dict
        Additional keyword parameters.

    Returns
    -------
        The transformed distribution function.

    """

    def distribution_fn(parameters):
        if isinstance(parameters, tuple) and len(parameters) == 2:
            parameters, base_parameters = parameters
            kwargs.update(base_parameters)
            __LOGGER__.debug("got parameters for base distribution.")

        __LOGGER__.debug("base distribution kwargs : %s", str(kwargs))
        base_distribution = get_base_distribution(dims, **kwargs)
        bijector = flow_parametrization_fn(parameters)
        return tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=bijector,
        )

    return distribution_fn


def _get_elementwise_flow(
    dims: int,
    get_base_distribution=_get_base_distribution,
    base_distribution_kwargs: Dict = {},
    **kwargs,
) -> Tuple[Callable[tf.Variable, tfd.Distribution], Tuple[int, ...]]:
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
        The parametrization function of the transformed distribution and its
        parameter shape.

    """
    flow_parametrization_fn, num_parameters = _get_flow_parametrization_fn(**kwargs)
    pv_shape = [dims, num_parameters]

    distribution_fn = _get_transformed_distribution_fn(
        dims,
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        is_joint=False,
        **base_distribution_kwargs,
    )

    return distribution_fn, pv_shape


def _get_multivariate_flow_fn(
    dims: int, **kwargs
) -> Tuple[Callable[tf.Variable, tfd.Distribution], Tuple[int, ...]]:
    """Get a function to parametrize a multivariate normalizing flow distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    kwargs
        Additional keyword arguments.

    Returns
    -------
        A function to parametrize the multivariate normalizing flow distribution and the
        shape of the parameter vector.

    """
    flow_parametrization_fn, num_parameters = _get_flow_parametrization_fn(**kwargs)
    pv_shape = [num_parameters * dims + np.sum(np.arange(dims + 1))]

    def distribution_fn(parameters):
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
    get_parameter_fn_for_layer: Callable,
    get_bijectors_for_layer: Callable,
    **kwargs,
) -> Tuple[
    Callable[tf.Variable, tfb.Bijector],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        Parametrization function, parameter function, and trainable variables.

    """
    flow_kwargs = kwargs.copy()

    # scale and shift have to be applied on all dimensions before the first
    # coupling layers so let's remove them from the kwargs to skip them in the
    # following calls of __get_bijector_fn__
    scale = flow_kwargs.pop("scale", False)
    shift = flow_kwargs.pop("shift", False)
    scale_to_domain = flow_kwargs.pop("scale_to_domain", False)

    flow_parametrization_fns = []
    parameter_networks = []
    trainable_variables = []
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

    def parameter_fn(conditional_input, **kwargs):
        return list(
            map(lambda net: net(conditional_input, **kwargs), parameter_networks)
        )

    def stacked_flow_parametrization_fn(parameter_networks):
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
    base_distribution_kwargs: Dict = {},
    **kwargs,
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        Parametrization function, parameter function, and trainable variables.

    """
    (
        flow_parametrization_fn,
        parameter_fn,
        trainable_variables,
    ) = _get_stacked_flow_parametrization_fn(dims, **kwargs)

    distribution_fn = _get_transformed_distribution_fn(
        dims,
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
        The number of masked dimensions.

    """
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


def _get_bijector_fn(
    network: Callable, flow_parametrization_fn: Callable
) -> Callable[tf.Variable, tfb.Bijector]:
    """Get a bijector function as a callable.

    Parameters
    ----------
    network
        The network to use for the bijector function.
    flow_parametrization_fn
        Function initializing the flow from unconstrained parameters.

    Returns
    -------
    callable
        A function to parametrize the bijector function.

    """

    def bijector_fn(y, *args, **kwargs):
        with tf.name_scope("bnf_bjector"):
            pvector = network(y, **kwargs)
            flow = flow_parametrization_fn(pvector)

        return flow

    return bijector_fn


# PUBLIC FUNCTIONS #############################################################
def get_coupling_flow(
    dims: int,
    distribution_kwargs: Dict,
    parameter_kwargs: Dict,
    num_masked: int = None,
    get_bijector_fn: Callable = _get_bijector_fn,
    get_parameter_fn: Callable = get_fully_connected_network_fn,
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        A function to parametrize the distribution, a callable for
        parameter networks, and a list of trainable parameters.

    """
    distribution_kwargs = distribution_kwargs.copy()
    num_layers = distribution_kwargs.pop("num_layers", 1)
    get_base_distribution = distribution_kwargs.pop(
        "get_base_distribution", _get_base_distribution
    )
    base_distribution_kwargs = distribution_kwargs.pop("base_distribution_kwargs", {})

    def get_parameter_fn_for_layer(layer, num_parameters):
        nm = num_masked if num_masked else _get_num_masked(dims, layer)
        parameter_shape = [dims - nm, num_parameters]
        return get_parameter_fn(
            input_shape=nm,
            parameter_shape=parameter_shape,
            **parameter_kwargs,
        )

    def get_bijectors_for_layer(layer, network, flow_parametrization_fn):
        nm = num_masked if num_masked else _get_num_masked(dims, layer)

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
    distribution_kwargs: Dict,
    parameter_kwargs: Dict,
    get_bijector_fn: Callable,
    get_parameter_fn: Callable,
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        A function to parametrize the distribution, a callable for
        parameter networks, and a list of trainable parameters.

    """

    def get_parameter_fn_for_layer(_, num_parameters):
        parameter_shape = [dims, num_parameters]
        return get_parameter_fn(parameter_shape, **parameter_kwargs)

    def get_bijectors_for_layer(_, network, flow_parametrization_fn):
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
    distribution_kwargs: Dict,
    parameter_kwargs: Dict,
    get_bijector_fn: Callable = _get_bijector_fn,
    get_parameter_fn: Callable = get_masked_autoregressive_network_fn,
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        A function to parametrize the distribution, a callable for
        parameter networks, and a list of trainable parameters.

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
        dims=dims,
        distribution_kwargs=distribution_kwargs,
        parameter_kwargs=parameter_kwargs,
        get_bijector_fn=get_bijector_fn,
        get_parameter_fn=get_parameter_fn,
    )
    distribution_fn = _get_transformed_distribution_fn(
        dims,
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, trainable_variables


def get_masked_autoregressive_flow_first_dim_masked(
    dims: int,
    distribution_kwargs: Dict,
    parameter_kwargs: Dict,
    get_bijector_fn: Callable = _get_bijector_fn,
    get_parameter_fn: Callable = get_masked_autoregressive_network_with_additive_conditioner_fn,  # noqa: E501
) -> Tuple[
    Callable[tf.Variable, tfd.Distribution],
    Callable[tf.Variable, tf.Variable],
    Tuple[int, ...],
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
        parameter_kwargs={"input_shape": [1], **parameter_kwargs},
        get_bijector_fn=get_bijector_fn,
        get_parameter_fn=get_parameter_fn,
    )

    def masked_flow_parametrization_fn(parameter_networks):
        def bijector_fn(x0, *arg, **kwargs):
            conditioned_parameter_networks = list(
                map(lambda net: partial(net, conditional_input=x0), parameter_networks)
            )
            stacked_maf_bijector = flow_parametrization_fn(
                conditioned_parameter_networks
            )
            return stacked_maf_bijector

        return tfb.RealNVP(num_masked=1, bijector_fn=bijector_fn)

    distribution_fn = _get_transformed_distribution_fn(
        dims,
        masked_flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )

    return distribution_fn, parameter_fn, trainable_variables


def _get_parametrized_bijector(bijector_name, parameters, **kwds):
    if bijector_name == "bernstein_poly":
        bijector_cls = BernsteinBijector
    else:
        bijector_class_name = "".join(map(str.title, bijector_name.split("_")))
        bijector_cls = getattr(tfb, bijector_class_name)
    return bijector_cls(parameters, **kwds)


def _get_parameter_fn(dims, key_prefix="", ignore_undefined=False, **kwargs):
    const_parameters_key = key_prefix + "parameters"
    parameter_fn_name_key = key_prefix + "parameter_fn"
    parameters_shape_key = key_prefix + "parameters_shape"
    parameter_fn_kwargs_key = key_prefix + "parameter_fn_kwargs"
    parameter_constraints_key = key_prefix + "parameter_constraints"
    # kwargs
    const_parameters = kwargs.get(const_parameters_key, None)
    parameter_fn_name = kwargs.get(parameter_fn_name_key, None)
    if const_parameters is not None and parameter_fn_name is not None:
        raise ValueError(
            f"The augments '{const_parameters_key}' and '{parameter_fn_name_key}' are "
            "mutually exclusive. Only provide either one of them"
        )
    elif const_parameters is None and parameter_fn_name is None:
        if ignore_undefined:
            __LOGGER__.debug("Ignoring undefined parameter function")
            return None, []
        else:
            raise ValueError(
                f"Either '{const_parameters_key}' or '{parameter_fn_name_key}' "
                "have to be provided."
            )

    # get parameter fn
    if const_parameters:
        __LOGGER__.debug("got constant parameters %s", str(const_parameters))
        parameter_fn = lambda *_, **__: const_parameters  # noqa: E731
    else:
        if callable(parameter_fn_name):
            __LOGGER__.debug("using provided callable as parameter function")
            get_parameter_fn = parameter_fn_name
        else:
            __LOGGER__.debug("parameter function: %s", parameter_fn_name)
            get_parameter_fn = getattr(parameters, f"get_{parameter_fn_name}_fn")

        __LOGGER__.debug("initializing parametrization function")
        parameters_shape = kwargs.get(parameters_shape_key, [])  # TODO: guess form kwds
        parameter_fn_kwargs = kwargs.get(parameter_fn_kwargs_key, {})

        parameter_fn, trainable_parameters = get_parameter_fn(
            parameter_shape=[dims] + parameters_shape,
            **parameter_fn_kwargs,
        )

    # get parameter constraint fn
    parameter_constraints = kwargs.get(
        parameter_constraints_key, False
    )  # TODO: guess from name
    if parameter_constraints:
        if callable(parameter_constraints):
            constrain_fn = parameter_constraints
            __LOGGER__.debug("using provided callable as parameter constraint function")
        else:
            if isinstance(parameter_constraints, dict):
                parameter_constrains_fn_name = parameter_constraints.pop("name")
                parameter_constrains_fn_kwargs = parameter_constraints
            else:
                parameter_constrains_fn_name = parameter_constraints
                parameter_constrains_fn_kwargs = {}
                __LOGGER__.debug(
                    "add parameter constraint function %s",
                    parameter_constrains_fn_name,
                )
            parameter_constrains_fn_name = parameter_constrains_fn_name.replace(
                "tfb.", "tensorflow_probability.python.bijectors.", 1
            ).replace("tf.", "tensorflow.", 1)
            module_name, obj_name = parameter_constrains_fn_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            constrain_fn = getattr(module, obj_name)
            if len(parameter_constrains_fn_kwargs) > 0 or isinstance(
                constrain_fn, type
            ):
                constrain_fn = constrain_fn(**parameter_constrains_fn_kwargs)

        constrained_parameter_fn = lambda *args, **kwargs: constrain_fn(  # noqa: E731
            parameter_fn(*args, **kwargs)
        )
        return constrained_parameter_fn, trainable_parameters
    else:
        return parameter_fn, trainable_parameters


def get_normalizing_flow(
    dims: int,
    transformations: List,
    inverse_flow: bool = True,
    get_base_distribution=_get_base_distribution,
    base_distribution_kwargs: Dict = {},
) -> Tuple[Callable[tf.Variable, tfd.Distribution], Tuple[int, ...]]:
    """Get a function to parametrize a elementwise transformed distribution.

    Parameters
    ----------
    dims
        The dimension of the distribution.
    transformations
        List of bijective transformation.
    inverse_flow
        Reverse direction of flow, to transform from the data to the base distribution.
    get_base_distribution
        The base distribution lambda.
    base_distribution_kwargs
        Keyword arguments for the base distribution.

    Returns
    -------
        The parametrization function of the transformed distribution and its
        parameter shape.

    """
    parameter_fns = {}
    trainable_parameters = {}

    for transformation in transformations:
        bijector_name = transformation["bijector_name"]
        __LOGGER__.debug("processing definition of transformation %s", bijector_name)
        parameter_fns[bijector_name], trainable_parameters[bijector_name] = (
            _get_parameter_fn(
                dims=dims,
                parameters=transformation.pop("parameters", None),
                parameter_fn=transformation.pop("parameter_fn", None),
                parameters_shape=transformation.pop("parameters_shape", None),
                parameter_fn_kwargs=transformation.pop("parameter_fn_kwargs", None),
                parameter_constraints=transformation.pop("parameter_constraints", None),
            )
        )

    __LOGGER__.debug("parameter function %s", str(parameter_fns))

    base_distribution_parameter_fn, base_distribution_trainable_parameters = (
        _get_parameter_fn(
            dims=dims,
            key_prefix="base_distribution",
            ignore_undefined=True,
            **base_distribution_kwargs,
        )
    )

    trainable_parameters["base_dsitribution"] = base_distribution_trainable_parameters

    def parameter_fn(*args, **kwargs):
        params = [fn(*args, **kwargs) for fn in parameter_fns.values()]
        if base_distribution_parameter_fn:
            base_params = base_distribution_parameter_fn(*args, **kwargs)
            return params, base_params
        else:
            return params

    def flow_parametrization_fn(all_parameters):
        bijectors = []
        for bi_kwds, pj_params in zip(transformations, all_parameters):
            bijectors.append(
                _get_parametrized_bijector(parameters=pj_params, **bi_kwds)
            )

        if inverse_flow:
            # the Chain bijector uses reversed list in the forward call.
            # We change the direction by first reversing the list to get f₃ ∘ f₂ ∘ f₁
            # and then invert it to get T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂
            bijectors = list(reversed(bijectors))

        if len(bijectors) == 1:
            flow = bijectors[0]
        else:
            flow = tfb.Chain(bijectors)

        if inverse_flow:
            flow = tfb.Invert(flow)

        return flow

    distribution_fn = _get_transformed_distribution_fn(
        dims,
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        is_joint=False,
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
