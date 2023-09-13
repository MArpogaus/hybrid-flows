# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2023-06-21 17:18:39 (Marcel Arpogaus)
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


# PRIVATE GLOBAL OBJECTS #######################################################
def __DEFAULT_BASE_DISTRIBUTION_LAMBDA__(dims):
    return tfd.Sample(tfd.Normal(0.0, 1.0), sample_shape=[dims])


# FUNCTIONS ####################################################################
def __get_parametrized_flow__(
    scale, shift, unconstrained_bernstein_coefficents, clip_to_bernstein_domain, **kwds
):
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    bijectors = []

    # f1: ŷ = a1(x)*(y - b1(x))
    if shift:
        shift = tf.convert_to_tensor(
            shift, dtype=unconstrained_bernstein_coefficents.dtype, name="shift"
        )[None, ...]
        f1_shift = tfb.Shift(shift, name="f1_shift")
        bijectors.append(f1_shift)

    if scale:
        scale = tf.convert_to_tensor(
            scale, dtype=unconstrained_bernstein_coefficents.dtype, name="scale"
        )[None, ...]
        f1_scale = tfb.Scale(scale, name="f1_scale")
        bijectors.append(f1_scale)

    # clip to domain [0, 1]
    if clip_to_bernstein_domain:
        bijectors.append(tfb.Sigmoid(name="sigmoid"))

    # f2: ẑ = Bernstein Polynomial
    bernstein_coefficents = thetas_constrain_fn(unconstrained_bernstein_coefficents)
    f2 = BernsteinBijectorLinearExtrapolate(bernstein_coefficents, name="bpoly")
    bijectors.append(f2)

    # tfp uses the invers T⁻¹ to calculate the log_prob
    # lets change the direction here by first reversing the list to get f₃ ∘ f₂ ∘ f₁
    bijectors = list(reversed(bijectors))

    # and now invert it to get T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂
    return tfb.Invert(tfb.Chain(bijectors))


def __get_bernstein_flow_lambda__(
    dims, order, base_distribution_lambda=__DEFAULT_BASE_DISTRIBUTION_LAMBDA__, **kwds
):
    pv_shape = [dims, order]

    def dist(pv):
        return tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims),
            bijector=__get_parametrized_flow__(
                unconstrained_bernstein_coefficents=pv, **kwds
            ),
        )

    return dist, pv_shape


def __get_multivariate_bernstein_flow_lambda__(dims, order, **kwds):
    pv_shape = [order * dims + np.sum(np.arange(dims + 1))]

    def dist(pv):
        bs = prefer_static.shape(pv)[:-1]
        shape = tf.concat((bs, [dims, order]), 0)

        unconstrained_bernstein_coeficents = tf.reshape(pv[..., : order * dims], shape)
        scale_tril = tfp.bijectors.FillScaleTriL()(pv[..., order * dims:])  # fmt: skip

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=__get_parametrized_flow__(
                unconstrained_bernstein_coefficents=unconstrained_bernstein_coeficents,
                **kwds,
            ),
        )

    return dist, pv_shape


def __get_multivariate_normal_lambda__(dims):
    pv_shape = [dims + np.sum(np.arange(dims + 1))]

    def dist(pv):
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


def __get_bijector_fn__(network, **flow_kwds):
    def bijector_fn(y, *arg, **kwds):
        with tf.name_scope("bnf_bjector"):
            pvector = network(y, **kwds)

            return __get_parametrized_flow__(
                unconstrained_bernstein_coefficents=pvector, **flow_kwds
            )

    return bijector_fn


def __get_num_masked__(dims, layer):
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


# PUBLIC FUNCTIONS #############################################################
def get_masked_autoregressive_bernstein_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_autoregressive_parameter_network_lambda,
):
    distribution_kwds = distribution_kwds.copy()
    order = distribution_kwds.pop("order")
    base_distribution_lambda = distribution_kwds.pop(
        "base_distribution_lambda", __DEFAULT_BASE_DISTRIBUTION_LAMBDA__
    )

    parameter_shape = (dims, order)
    parameter_network_lambda, trainable_variables = get_parameter_lambda_fn(
        parameter_shape, **parameter_kwds
    )

    def distribution_lambda(parameter_network):
        bijector_fn = __get_bijector_fn__(
            network=parameter_network, **distribution_kwds
        )

        distribution = tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims),
            bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn),
        )
        return distribution

    return distribution_lambda, parameter_network_lambda, trainable_variables


def get_coupling_bernstein_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_simple_fully_connected_parameter_network_lambda,
):
    distribution_kwds = distribution_kwds.copy()
    order = distribution_kwds.pop("order")
    base_distribution_lambda = distribution_kwds.pop(
        "base_distribution_lambda", __DEFAULT_BASE_DISTRIBUTION_LAMBDA__
    )
    coupling_layers = distribution_kwds.pop("coupling_layers")

    # scale and shift have to be applied on all dimensions before the first
    # coupling layers so let's remove them from the kwds to skip them in the
    # following calls of __get_bijector_fn__
    scale = distribution_kwds.pop("scale", False)
    shift = distribution_kwds.pop("shift", False)

    distribution_kwds.update(scale=False, shift=False)

    parameter_networks = []
    trainable_variables = []
    for layer in range(coupling_layers):
        num_masked = __get_num_masked__(dims, layer)
        parameter_shape = (dims - num_masked, order)
        network, variables = get_parameter_lambda_fn(
            input_shape=num_masked,
            parameter_shape=parameter_shape,
            **parameter_kwds,
        )
        parameter_networks.append(network)
        trainable_variables.append(variables)

    def parameter_lambda(conditional_input, **kwds):
        if parameter_kwds.get("conditional", False):
            return list(
                map(lambda net: net(conditional_input, **kwds), parameter_networks)
            )
        else:
            return parameter_networks

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
            num_masked = __get_num_masked__(dims, layer)
            permutation = list(range(num_masked, dims)) + list(range(num_masked))
            bijectors.append(
                tfb.RealNVP(
                    num_masked=num_masked,
                    bijector_fn=__get_bijector_fn__(network, **distribution_kwds),
                )
            )
            if coupling_layers % 2 != 0 and layer == (coupling_layers - 1):
                print("uneven number of coupling layers -> skipping last permutation")
            else:
                bijectors.append(tfb.Permute(permutation=permutation))

        return tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims),
            bijector=tfb.Chain(bijectors),
        )

    return distribution_lambda, parameter_lambda, trainable_variables


def get_masked_autoregressive_bernstein_flow_first_dim_masked(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_autoregressive_parameter_network_with_additive_conditioner_lambda,  # noqa: E501
):
    distribution_kwds = distribution_kwds.copy()
    order = distribution_kwds.pop("order")
    base_distribution_lambda = distribution_kwds.pop(
        "base_distribution_lambda", __DEFAULT_BASE_DISTRIBUTION_LAMBDA__
    )

    parameter_shape = (dims - 1, order)
    parameter_lambda, trainable_parameters = get_parameter_lambda_fn(
        parameter_shape, **parameter_kwds
    )

    def get_bijector_fn(parameter_network):
        bijector_fn = __get_bijector_fn__(
            network=parameter_network, **distribution_kwds
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

    def distribution_lambda(parameter_network):
        bijector = tfb.RealNVP(
            num_masked=1,
            bijector_fn=get_bijector_fn(parameter_network),
        )
        return tfd.TransformedDistribution(
            distribution=base_distribution_lambda(dims),
            bijector=bijector,
        )

    return distribution_lambda, parameter_lambda, trainable_parameters


get_bernstein_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_bernstein_flow_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
get_multivariate_bernstein_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_bernstein_flow_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
get_multivariate_normal = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_normal_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_or_simple_network_lambda,
)
