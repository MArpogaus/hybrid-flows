# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2024-03-07 16:14:34 (Marcel Arpogaus)
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

from functools import partial
from itertools import chain

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinBijector
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from .activations import get_spline_param_constrain_fn, get_thetas_constrain_fn
from .parameters import (
    get_autoregressive_parameter_network_fn,
    get_autoregressive_parameter_network_with_additive_conditioner_fn,
    get_parameter_vector_or_simple_network_fn,
    get_simple_fully_connected_parameter_network_fn,
)


# FUNCTIONS ####################################################################
def __get_multivariate_normal_fn__(dims):
    """Get a Multivariate Normal distribution as a callable.

    :param int dims: The dimension of the distribution.
    :return: A callable representing the Multivariate Normal distribution.
    :rtype: callable
    :return: The shape of the parameter vector.
    :rtype: list
    """
    pv_shape = [dims + np.sum(np.arange(dims + 1))]

    def dist(parameters):
        loc = parameters[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, pv_shape


def __get_trainable_distribution__(
    dims,
    get_distribution_fn,
    distribution_kwds,
    get_parameter_fn,
    parameter_kwds,
):
    """Get a trainable distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param callable get_distribution_fn: A function to get the
                                               distribution lambda.
    :param dict distribution_kwds: Keyword arguments for the distribution.
    :param callable get_parameter_fn: A function to get the
                                             parameter lambda.
    :param dict parameter_kwds: Keyword arguments for the parameters.
    :return: A callable representing the distribution.
    :rtype: callable
    :return: A callable for parameter vectors.
    :rtype: callable
    :return: List of trainable parameters.
    :rtype: list
    """
    distribution_fn, parameters_shape = get_distribution_fn(
        dims=dims, **distribution_kwds
    )
    parameter_fn, trainable_parameters = get_parameter_fn(
        parameters_shape, **parameter_kwds
    )
    return distribution_fn, parameter_fn, trainable_parameters


def __get_base_distribution__(dims, distribution_type="normal", is_joint=True, **kwds):
    """Get the default base distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param str distribution_type: The type of distribution
                    (e.g., "normal", "lognormal", "uniform", "kumaraswamy").
    :param **kwds: Additional keyword arguments for the distribution.
    :return: The default base distribution.
    :rtype: Distribution
    """
    if distribution_type == "normal":
        default_kwds = dict(loc=0.0, scale=1.0)
        default_kwds.update(**kwds)
        dist = tfd.Normal(**default_kwds)
    elif distribution_type == "lognormal":
        default_kwds = dict(loc=0.0, scale=1.0)
        default_kwds.update(**kwds)
        dist = tfd.LogNormal(**default_kwds)
    elif distribution_type == "uniform":
        dist = tfd.Uniform(**kwds)
    elif distribution_type == "kumaraswamy":
        dist = tfd.Kumaraswamy(**kwds)
    else:
        raise ValueError(f"Unsupported distribution type {distribution_type}.")
    if is_joint:
        dist = tfd.Sample(dist, sample_shape=[dims])
    return dist


def __get_parametrized_bijector_fn__(
    bijector_name, parameter_constrain_fn=None, **kwds
):
    if bijector_name == "bernstein_poly":
        parameters_shape = [kwds.pop("order")]
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_thetas_constrain_fn(**kwds)

        def bijector_fn(unconstrained_parameters):
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = BernsteinBijector(constrained_parameters, name=bijector_name)

            return bijector

    elif bijector_name == "quadratic_spline":
        parameters_shape = [kwds["nbins"] * 3 - 1]
        range_min = kwds.pop("range_min")
        if parameter_constrain_fn is None:
            parameter_constrain_fn = get_spline_param_constrain_fn(**kwds)

        def bijector_fn(unconstrained_parameters):
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = tfb.RationalQuadraticSpline(
                **constrained_parameters, range_min=range_min, name=bijector_name
            )

            return bijector

    else:
        raise ValueError(f"Unknown bijector type: {bijector_name}")

    return bijector_fn, parameters_shape


def __get_flow_parametrization_fn__(
    scale,
    shift,
    bijector_name,
    get_parametrized_bijector_fn=__get_parametrized_bijector_fn__,
    **kwds,
):
    """Get a parametrized flow as a callable.

    :param scale: The scale of the flow.
    :param shift: The shift of the flow.
    :param unconstrained_bernstein_coefficents: The unconstrained
    Bernstein coefficients.
    :param clip_to_bernstein_domain: Whether to clip to the Bernstein domain.
    :param **kwds: Additional keyword arguments.
    :return: The parametrized flow bijector.
    :rtype: bijector
    """
    (
        bijector_fn,
        parameter_shape,
    ) = get_parametrized_bijector_fn(bijector_name, **kwds)

    def flow_parametrization_fn(unconstrained_parameters):
        bijectors = []

        # ŷ = a1(x)*(y + b1(x)) = f2(f1(y,x),x)
        # f1 = y + b1(x)
        if shift:
            shift_bj = tfb.Shift(
                tf.convert_to_tensor(
                    shift, dtype=unconstrained_parameters.dtype, name="shift"
                ),
                name="shift",
            )
            bijectors.append(shift_bj)

        # f2 = a1(x) * y
        if scale:
            scale_bj = tfb.Scale(
                tf.convert_to_tensor(
                    scale, dtype=unconstrained_parameters.dtype, name="scale"
                ),
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

    return flow_parametrization_fn, parameter_shape


def __get_transformed_distribution_fn__(
    dims,
    flow_parametrization_fn,
    get_base_distribution=__get_base_distribution__,
    **kwds,
):
    def distribution_fn(parameters):
        base_distribution = get_base_distribution(dims, **kwds)
        bijector = flow_parametrization_fn(parameters)
        return tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=bijector,
        )

    return distribution_fn


def __get_elementwise_flow__(
    dims,
    get_base_distribution=__get_base_distribution__,
    base_distribution_kwds={},
    **kwds,
):
    """Get a Bernstein Flow distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param int order: The order of the Bernstein Flow.
    :param get_base_distribution: The base distribution lambda.
    :param **kwds: Additional keyword arguments.
    :return: The Bernstein Flow distribution.
    :rtype: Distribution
    """
    flow_parametrization_fn, parameters_shape = __get_flow_parametrization_fn__(**kwds)
    pv_shape = [dims] + parameters_shape

    distribution_fn = __get_transformed_distribution_fn__(
        dims,
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        is_joint=False,
        **base_distribution_kwds,
    )

    return distribution_fn, pv_shape


def __get_multivariate_flow_fn__(dims, **kwds):
    """Get a Multivariate flow distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param **kwds: Additional keyword arguments.
    :return: A callable representing the Multivariate Bernstein Flow distribution.
    :rtype: callable
    :return: The shape of the parameter vector.
    :rtype: list
    """
    flow_parametrization_fn, parameters_shape = __get_flow_parametrization_fn__(**kwds)
    num_params = np.sum(parameters_shape)
    pv_shape = [num_params * dims + np.sum(np.arange(dims + 1))]

    def distribution_fn(parameters):
        bs = prefer_static.shape(parameters)[:-1]
        shape = tf.concat((bs, [dims, num_params]), 0)

        unconstrained_parameters = tf.reshape(
            parameters[..., : num_params * dims], shape
        )
        scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., num_params * dims :])

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=flow_parametrization_fn(
                unconstrained_parameters,
            ),
        )

    return distribution_fn, pv_shape


def __get_stacked_flow_parametrization_fn__(
    dims,
    num_layers,
    get_parameter_fn_for_layer,
    get_bijectors_for_layer,
    **kwds,
):
    flow_kwds = kwds.copy()

    # scale and shift have to be applied on all dimensions before the first
    # coupling layers so let's remove them from the kwds to skip them in the
    # following calls of __get_bijector_fn__
    scale = flow_kwds.pop("scale", False)
    shift = flow_kwds.pop("shift", False)

    flow_kwds.update(scale=False, shift=False)

    flow_parametrization_fn, parameters_shape = __get_flow_parametrization_fn__(
        **flow_kwds
    )

    parameter_networks = []
    trainable_variables = []
    for layer in range(num_layers):
        network, variables = get_parameter_fn_for_layer(layer, parameters_shape)
        parameter_networks.append(network)
        trainable_variables.append(variables)

    def parameter_fn(conditional_input, **kwds):
        return list(map(lambda net: net(conditional_input, **kwds), parameter_networks))

    def stacked_flow_parametrization_fn(parameter_networks):
        bijectors = []
        # tfp uses the invers T⁻¹ to calculate the log_prob
        # so we use the invers here to use scale parameters inferred from the data
        if shift:
            shift_t = tf.convert_to_tensor(shift, name="shift")[None, ...]
            f1_shift = tfb.Shift(shift_t, name="f1_shift")
            bijectors.append(tfb.Invert(f1_shift))
        if scale:
            scale_t = tf.convert_to_tensor(scale, name="scale")[None, ...]
            f1_scale = tfb.Scale(scale_t, name="f1_scale")
            bijectors.append(tfb.Invert(f1_scale))

        # Stack bijectors
        for layer, network in enumerate(parameter_networks):
            bijectors.extend(
                get_bijectors_for_layer(layer, network, flow_parametrization_fn)
            )

        return tfb.Chain(bijectors)

    return (
        stacked_flow_parametrization_fn,
        parameter_fn,
        chain.from_iterable(trainable_variables),
    )


def __get_stacked_flow__(
    dims,
    get_base_distribution=__get_base_distribution__,
    base_distribution_kwds={},
    **kwds,
):
    (
        flow_parametrization_fn,
        parameter_fn,
        trainable_variables,
    ) = __get_stacked_flow_parametrization_fn__(
        dims,
        **kwds,
    )

    distribution_fn = __get_transformed_distribution_fn__(
        dims,
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwds,
    )

    return distribution_fn, parameter_fn, trainable_variables


def __get_num_masked__(dims, layer):
    """Compute the number of masked dimensions.

    :param int dims: The total number of dimensions.
    :param int layer: The layer number.
    :return: The number of masked dimensions.
    :rtype: int
    """
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


def __get_bijector_fn__(network, flow_parametrization_fn):
    """Get a bijector function as a callable.

    :param callable network: The network to use for the bijector function.
    :param callable flow_parametrization_fn: function initializing the flow
                                                 from unconstrained parameters.
    :return: A callable representing the bijector function.
    :rtype: callable
    """

    def bijector_fn(y, *arg, **kwds):
        with tf.name_scope("bnf_bjector"):
            pvector = network(y, **kwds)
            flow = flow_parametrization_fn(pvector)

            return flow

    return bijector_fn


# PUBLIC FUNCTIONS #############################################################
def get_coupling_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    num_masked=None,
    get_bijector_fn=__get_bijector_fn__,
    get_parameter_fn=get_simple_fully_connected_parameter_network_fn,
):
    """Get a Coupling Flow distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param dict distribution_kwds: Keyword arguments for the distribution.
    :param dict parameter_kwds: Keyword arguments for the parameters.
    :param callable get_parameter_fn: A function to get the parameter
    lambda.
    :return: A callable representing the distribution.
    :rtype: callable
    :return: A callable for parameter networks.
    :rtype: callable
    :return: List of trainable parameters.
    :rtype: list
    """
    distribution_kwds = distribution_kwds.copy()
    num_layers = distribution_kwds.pop("num_layers", 1)
    get_base_dsitribution = distribution_kwds.pop(
        "get_base_distribution", __get_base_distribution__
    )
    base_distribution_kwds = distribution_kwds.pop("base_distribution_kwds", {})

    def get_parameter_fn_for_layer(layer, parameters_shape):
        nm = num_masked if num_masked else __get_num_masked__(dims, layer)
        parameter_shape = [dims - nm] + parameters_shape
        return get_parameter_fn(
            input_shape=nm,
            parameter_shape=parameter_shape,
            **parameter_kwds,
        )

    def get_bijectors_for_layer(layer, network, flow_parametrization_fn):
        nm = num_masked if num_masked else __get_num_masked__(dims, layer)

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

    return __get_stacked_flow__(
        dims=dims,
        num_layers=num_layers,
        get_parameter_fn_for_layer=get_parameter_fn_for_layer,
        get_bijectors_for_layer=get_bijectors_for_layer,
        get_base_distribution=get_base_dsitribution,
        base_distribution_kwds=base_distribution_kwds,
        **distribution_kwds,
    )


def get_masked_autoregressive_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_bijector_fn=__get_bijector_fn__,
    get_parameter_fn=get_autoregressive_parameter_network_fn,
):
    """Get a Masked Autoregressive Flow distribution as a callable.

    :param int dims: The dimension of the distribution.
    :param dict distribution_kwds: Keyword arguments for the distribution.
    :param dict parameter_kwds: Keyword arguments for the parameters.
    :param callable get_parameter_fn: A function to get the parameter
    lambda.
    :return: A callable representing the distribution.
    :rtype: callable
    :return: A callable for parameter networks.
    :rtype: callable
    :return: List of trainable parameters.
    :rtype: list
    """
    distribution_kwds = distribution_kwds.copy()
    num_layers = distribution_kwds.pop("num_layers", 1)
    get_base_dsitribution = distribution_kwds.pop(
        "get_base_distribution", __get_base_distribution__
    )
    base_distribution_kwds = distribution_kwds.pop("base_distribution_kwds", {})

    def get_parameter_fn_for_layer(_, parameters_shape):
        parameter_shape = [dims] + parameters_shape
        return get_parameter_fn(parameter_shape, **parameter_kwds)

    def get_bijectors_for_layer(_, network, flow_parametrization_fn):
        bijector_fn = get_bijector_fn(
            network=network,
            flow_parametrization_fn=flow_parametrization_fn,
        )

        return [tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn)]

    return __get_stacked_flow__(
        dims=dims,
        num_layers=num_layers,
        get_parameter_fn_for_layer=get_parameter_fn_for_layer,
        get_bijectors_for_layer=get_bijectors_for_layer,
        get_base_distribution=get_base_dsitribution,
        base_distribution_kwds=base_distribution_kwds,
        **distribution_kwds,
    )


def get_masked_autoregressive_flow_first_dim_masked(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_fn=get_autoregressive_parameter_network_with_additive_conditioner_fn,  # noqa: E501
):
    """Get a Masked Autoregressive Bernstein Flow.

    Distribution with the first dimension masked as a callable.

    :param int dims: The dimension of the distribution.
    :param dict distribution_kwds: Keyword arguments for the distribution.
    :param dict parameter_kwds: Keyword arguments for the parameters.
    :param callable get_parameter_fn: A function to get the parameters.
    :return: A callable representing the Masked Autoregressive Bernstein Flow
             distribution with the first dimension masked.
    :rtype: callable
    :return: A callable for parameter networks.
    :rtype: callable
    :return: List of trainable parameters.
    :rtype: list
    """
    distribution_kwds.update(num_layers=1)

    def get_bijector_fn(parameter_network, flow_parametrization_fn):
        bijector_fn = __get_bijector_fn__(
            network=parameter_network,
            flow_parametrization_fn=flow_parametrization_fn,
        )
        maf_bijector = tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn)

        def bijector_fn(x0, *arg, **kwds):
            with tf.name_scope("bernstein_bjector"):
                return tfb.Inline(
                    forward_fn=partial(maf_bijector.forward, conditional_input=x0),
                    inverse_fn=partial(maf_bijector.inverse, conditional_input=x0),
                    inverse_log_det_jacobian_fn=partial(
                        maf_bijector.inverse_log_det_jacobian, conditional_input=x0
                    ),
                    forward_min_event_ndims=1,
                    # is_increasing=True,
                    name="maf_cond",
                )

        return bijector_fn

    return get_coupling_flow(
        dims=dims,
        distribution_kwds=distribution_kwds,
        parameter_kwds=parameter_kwds,
        num_masked=1,
        get_bijector_fn=get_bijector_fn,
        get_parameter_fn=get_parameter_fn,
    )


# actual functions that are composed of base functions
get_elementwise_flow = partial(
    __get_trainable_distribution__,
    get_distribution_fn=__get_elementwise_flow__,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
get_multivariate_flow = partial(
    __get_trainable_distribution__,
    get_distribution_fn=__get_multivariate_flow_fn__,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
get_multivariate_normal = partial(
    __get_trainable_distribution__,
    get_distribution_fn=__get_multivariate_normal_fn__,
    get_parameter_fn=get_parameter_vector_or_simple_network_fn,
)
