# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2023-10-17 09:29:33 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinBijectorLinearExtrapolate
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from .parameters import (
    get_autoregressive_parameter_network_lambda,
    get_autoregressive_parameter_network_with_additive_conditioner_lambda,
    get_parameter_vector_or_simple_network_lambda,
    get_simple_fully_connected_parameter_network_lambda,
)


# FUNCTIONS ####################################################################
def __default_base_distribution_lambda__(dims, distribution_type="normal", **kwds):
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
    return tfd.Sample(dist, sample_shape=[dims])


def get_spline_param_constrain_fn(nbins, interval_width, min_bin_width, min_slope):
    def _bin_positions(x):
        return (
            tf.math.softmax(x, axis=-1) * (interval_width - nbins * min_bin_width)
            + min_bin_width
        )

    def _slopes(x):
        return tf.math.softplus(x) + min_slope

    def constrain_fn(unconstrained_parameters):
        # shape: [dims, 3*nbins - 1 ]
        bin_widths, bin_heights, knot_slopes = tf.split(
            unconstrained_parameters, [nbins, nbins, nbins - 1], axis=-1
        )

        return dict(
            bin_widths=_bin_positions(bin_widths),
            bin_heights=_bin_positions(bin_heights),
            knot_slopes=_slopes(knot_slopes),
        )

    return constrain_fn


def __get_parametrized_bijector_fn__(bijector_name, **kwds):
    if bijector_name == "bernstein_poly":
        parameters_shape = [kwds.pop("order")]
        parameter_constrain_fn = get_thetas_constrain_fn(**kwds)

        def bijector_fn(unconstrained_parameters):
            constrained_parameters = parameter_constrain_fn(unconstrained_parameters)
            bijector = BernsteinBijectorLinearExtrapolate(
                constrained_parameters, name=bijector_name
            )

            return bijector

    elif bijector_name == "quadratic_spline":
        parameters_shape = [kwds["nbins"] * 3 - 1]
        range_min = kwds.pop("range_min")
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


def __get_flow_parametrization_lambda__(
    scale,
    shift,
    bijector_name,
    get_parametrized_bijector_fn=__get_parametrized_bijector_fn__,
    **kwds,
):
    (
        bijector_fn,
        parameter_shape,
    ) = get_parametrized_bijector_fn(bijector_name, **kwds)

    def flow_parametrization_lambda(unconstrained_parameters):
        bijectors = []

        # ŷ = a1(x)*(y + b1(x))
        if shift:
            shift_bj = tfb.Shift(
                tf.convert_to_tensor(
                    shift, dtype=unconstrained_parameters.dtype, name="shift"
                )[None, ...],
                name="shift",
            )
            bijectors.append(shift_bj)

        if scale:
            scale_bj = tfb.Scale(
                tf.convert_to_tensor(
                    scale, dtype=unconstrained_parameters.dtype, name="scale"
                )[None, ...],
                name="scale",
            )
            bijectors.append(scale_bj)

        # Flexible Transformation Function
        bijectors.append(bijector_fn(unconstrained_parameters))

        # tfp uses the invers T⁻¹ to calculate the log_prob
        # lets change the direction here by first reversing the list to get f₃ ∘ f₂ ∘ f₁
        bijectors = list(reversed(bijectors))

        # and now invert it to get T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂
        return tfb.Invert(tfb.Chain(bijectors))

    return flow_parametrization_lambda, parameter_shape


def __get_elementwise_flow__(
    dims, base_distribution_lambda=__default_base_distribution_lambda__, **kwds
):
    flow_parametrization_lambda, parameters_shape = __get_flow_parametrization_lambda__(
        **kwds
    )
    pv_shape = [dims] + parameters_shape

    def dist(pv_lambda):
        pv = pv_lambda()
        return tfd.TransformedDistribution(
            distribution=base_distribution_lambda(
                dims, **kwds.pop("base_distribution_kwds", {})
            ),
            bijector=flow_parametrization_lambda(pv),
        )

    return dist, pv_shape


def __get_multivariate_flow_lambda__(dims, **kwds):
    flow_parametrization_lambda, parameters_shape = __get_flow_parametrization_lambda__(
        **kwds
    )
    num_params = np.sum(parameters_shape)
    pv_shape = [num_params * dims + np.sum(np.arange(dims + 1))]

    def dist(pv_lambda):
        pv = pv_lambda()
        bs = prefer_static.shape(pv)[:-1]
        shape = tf.concat((bs, [dims, num_params]), 0)

        unconstrained_parameters = tf.reshape(pv[..., : num_params * dims], shape)
        scale_tril = tfp.bijectors.FillScaleTriL()(
            pv[..., num_params * dims:]  # fmt: skip
        )

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=flow_parametrization_lambda(
                unconstrained_parameters,
            ),
        )

    return dist, pv_shape


def __get_multivariate_normal_lambda__(dims):
    pv_shape = [dims + np.sum(np.arange(dims + 1))]

    def dist(pv_lambda):
        pv = pv_lambda()

        loc = pv[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(pv[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, pv_shape


def __get_trainable_distribution__(
    dims,
    get_distribution_lambda_fn,
    distribution_kwds,
    get_parameter_lambda_fn,
    parameter_kwds,
):
    distribution_lambda, parameters_shape = get_distribution_lambda_fn(
        dims=dims, **distribution_kwds
    )
    parameter_vector_lambda, trainable_parameters = get_parameter_lambda_fn(
        parameters_shape, **parameter_kwds
    )
    return distribution_lambda, parameter_vector_lambda, trainable_parameters


def __get_bijector_fn__(network, flow_parametrization_lambda):
    def bijector_fn(y, *arg, **kwds):
        with tf.name_scope("bnf_bjector"):
            pvector = network(y, **kwds)

            return flow_parametrization_lambda(pvector)

    return bijector_fn


def __get_num_masked__(dims, layer):
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


# PUBLIC FUNCTIONS #############################################################
def get_masked_autoregressive_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_bijector_fn=__get_bijector_fn__,
    get_parameter_lambda_fn=get_autoregressive_parameter_network_lambda,
):
    distribution_kwds = distribution_kwds.copy()
    base_distribution_lambda = distribution_kwds.pop(
        "base_distribution_lambda", __default_base_distribution_lambda__
    )
    base_distribution_kwds = distribution_kwds.pop("base_distribution_kwds", {})

    flow_parametrization_lambda, parameters_shape = __get_flow_parametrization_lambda__(
        **distribution_kwds
    )

    parameter_shape = [dims] + parameters_shape
    parameter_network_lambda, trainable_variables = get_parameter_lambda_fn(
        parameter_shape, **parameter_kwds
    )

    def distribution_lambda(parameter_network):
        bijector_fn = get_bijector_fn(
            network=parameter_network,
            flow_parametrization_lambda=flow_parametrization_lambda,
        )

        distribution = tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims, **base_distribution_kwds),
            bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn),
        )
        return distribution

    return distribution_lambda, parameter_network_lambda, trainable_variables


def get_coupling_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    num_masked=None,
    get_bijector_fn=__get_bijector_fn__,
    get_parameter_lambda_fn=get_simple_fully_connected_parameter_network_lambda,
):
    distribution_kwds = distribution_kwds.copy()
    base_distribution_lambda = distribution_kwds.pop(
        "base_distribution_lambda", __default_base_distribution_lambda__
    )
    base_distribution_kwds = distribution_kwds.pop("base_distribution_kwds", {})
    coupling_layers = distribution_kwds.pop("coupling_layers")

    # scale and shift have to be applied on all dimensions before the first
    # coupling layers so let's remove them from the kwds to skip them in the
    # following calls of __get_bijector_fn__
    scale = distribution_kwds.pop("scale", False)
    shift = distribution_kwds.pop("shift", False)

    distribution_kwds.update(scale=False, shift=False)

    flow_parametrization_lambda, parameters_shape = __get_flow_parametrization_lambda__(
        **distribution_kwds
    )

    parameter_shape = [dims] + parameters_shape

    parameter_networks = []
    trainable_variables = []
    for layer in range(coupling_layers):
        nm = num_masked if num_masked else __get_num_masked__(dims, layer)
        parameter_shape = [dims - nm] + parameters_shape
        network, variables = get_parameter_lambda_fn(
            input_shape=nm,
            parameter_shape=parameter_shape,
            **parameter_kwds,
        )
        parameter_networks.append(network)
        trainable_variables.append(variables)

    def parameter_lambda(conditional_input=None, **kwds):
        return list(map(lambda net: net(conditional_input, **kwds), parameter_networks))

    def distribution_lambda(parameter_networks):
        bijectors = []
        # tfp uses the invers T⁻¹ to calculate the log_prob
        # the Chain bijector uses reversed list in the forward call so we want the
        # chained bijectors in the order f₁ ∘ f₂ ∘ … ∘ fᵢ and use their inverse
        # to get T = f₃⁻¹ ∘ … ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂

        if shift:
            shift_t = tf.convert_to_tensor(shift, name="shift")[None, ...]
            f1_shift = tfb.Shift(shift_t, name="f1_shift")
            bijectors.append(tfb.Invert(f1_shift))
        if scale:
            scale_t = tf.convert_to_tensor(scale, name="scale")[None, ...]
            f1_scale = tfb.Scale(scale_t, name="f1_scale")
            bijectors.append(tfb.Invert(f1_scale))

        for layer, network in enumerate(parameter_networks):
            nm = num_masked if num_masked else __get_num_masked__(dims, layer)
            permutation = list(range(nm, dims)) + list(range(nm))
            bijectors.append(
                tfb.RealNVP(
                    num_masked=nm,
                    bijector_fn=get_bijector_fn(network, flow_parametrization_lambda),
                )
            )
            if coupling_layers % 2 != 0 and layer == (coupling_layers - 1):
                print("uneven number of coupling layers -> skipping last permutation")
            else:
                bijectors.append(tfb.Permute(permutation=permutation))

        return tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims, **base_distribution_kwds),
            bijector=tfb.Chain(bijectors),
        )

    return distribution_lambda, parameter_lambda, trainable_variables


def get_masked_autoregressive_flow_first_dim_masked(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_autoregressive_parameter_network_with_additive_conditioner_lambda,  # noqa: E501
):
    distribution_kwds.update(coupling_layers=1)

    def get_bijector_fn(parameter_network, flow_parametrization_lambda):
        bijector_fn = __get_bijector_fn__(
            network=parameter_network,
            flow_parametrization_lambda=flow_parametrization_lambda,
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
        get_parameter_lambda_fn=get_parameter_lambda_fn,
    )


get_elementwise_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_elementwise_flow__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
get_multivariate_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_flow_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
get_multivariate_normal = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_normal_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
