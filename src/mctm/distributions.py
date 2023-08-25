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
    get_parameter_vector_lambda,
    get_simple_fully_connected_parameter_network_lambda,
)

# PRIVATE GLOBAL OBJECTS #######################################################
__DEFAULT_BASE_DISTRIBUTION__ = tfd.Normal(0.0, 1.0)


# FUNCTIONS ####################################################################
def __get_bernstein_flow_lambda__(
    dims, order, base_distribution=__DEFAULT_BASE_DISTRIBUTION__, **kwds
):
    pv_shape = [dims, order]
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    def dist(pv):
        thetas = thetas_constrain_fn(pv)

        return tfd.TransformedDistribution(
            distribution=tfd.Sample(base_distribution, sample_shape=[dims]),
            bijector=tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas)),
        )

    return dist, pv_shape


def __get_multivariate_bernstein_flow_lambda__(dims, order, **kwds):
    pv_shape = [order * dims + np.sum(np.arange(dims + 1))]
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    def dist(pv):
        bs = prefer_static.shape(pv)[:-1]
        shape = tf.concat((bs, [dims, order]), 0)

        thetas = thetas_constrain_fn(tf.reshape(pv[..., : order * dims], shape))
        scale_tril = tfp.bijectors.FillScaleTriL()(pv[..., order * dims:])  # fmt: skip

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas)),
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


def __get_bijector_fn__(network, thetas_constrain_fn, **kwds):
    def bijector_fn(y, *arg, **kwds):
        with tf.name_scope("bnf_made_bjector"):
            pvector = network(y, **kwds)  # todo: add conditionals
            thetas = thetas_constrain_fn(pvector)

            return tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas))

    return bijector_fn


# PUBLIC FUNCTIONS #############################################################
def get_masked_autoregressive_bernstein_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_autoregressive_parameter_network_lambda,
):
    order = distribution_kwds.pop("order")
    base_distribution = distribution_kwds.pop(
        "base_distribution", __DEFAULT_BASE_DISTRIBUTION__
    )

    parameter_shape = (dims, order)
    parameter_network, trainable_variables = get_parameter_lambda_fn(
        parameter_shape, **parameter_kwds
    )

    thetas_constrain_fn = get_thetas_constrain_fn(**distribution_kwds)

    def distribution_lambda(parameter_network):
        bijector_fn = __get_bijector_fn__(
            network=parameter_network, thetas_constrain_fn=thetas_constrain_fn
        )

        distribution = tfd.TransformedDistribution(
            distribution=tfd.Sample(base_distribution, sample_shape=[dims]),
            bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn),
        )
        return distribution

    return distribution_lambda, parameter_network, trainable_variables


def get_coupling_bernstein_flow(
    dims,
    distribution_kwds,
    parameter_kwds,
    get_parameter_lambda_fn=get_simple_fully_connected_parameter_network_lambda,
):
    order = distribution_kwds.pop("order")
    base_distribution = distribution_kwds.pop(
        "base_distribution", __DEFAULT_BASE_DISTRIBUTION__
    )
    coupling_layers = distribution_kwds.pop("coupling_layers")

    parameter_shape = (dims // 2, order)

    thetas_constrain_fn = get_thetas_constrain_fn(**distribution_kwds)

    parameter_networks = []
    trainable_variables = []
    for _ in range(coupling_layers):
        network, variables = get_parameter_lambda_fn(
            input_shape=dims // 2, parameter_shape=parameter_shape, **parameter_kwds
        )
        parameter_networks.append(network)
        trainable_variables.append(variables)

    def parameter_lambda(conditional_input=None, **kwds):
        if parameter_kwds.get("conditional", False):
            return list(
                map(
                    lambda net: lambda x: net([x, conditional_input], **kwds),
                    parameter_networks,
                )
            )
        else:
            return parameter_networks

    def distribution_lambda(parameter_networks):
        bijectors = []
        for layer, network in enumerate(parameter_networks):
            bijectors.append(
                tfb.RealNVP(
                    num_masked=(dims // 2),
                    bijector_fn=__get_bijector_fn__(network, thetas_constrain_fn),
                )
            )
            if coupling_layers % 2 != 0 and layer == (coupling_layers - 1):
                print("uneven number of coupling layers -> skipping last permutation")
            else:
                bijectors.append(tfb.Permute(permutation=[1, 0]))

        return tfd.TransformedDistribution(
            distribution=tfd.Sample(base_distribution, sample_shape=[dims]),
            bijector=tfb.Chain(bijectors),
        )

    return distribution_lambda, parameter_lambda, trainable_variables


get_bernstein_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_bernstein_flow_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_lambda,
    parameter_kwds={"dtype": tf.float32},
)
get_multivariate_bernstein_flow = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_bernstein_flow_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_lambda,
    parameter_kwds={"dtype": tf.float32},
)
get_multivariate_normal = partial(
    __get_trainable_distribution__,
    get_distribution_lambda_fn=__get_multivariate_normal_lambda__,
    get_parameter_lambda_fn=get_parameter_vector_lambda,
    parameter_kwds={"dtype": tf.float32},
)
