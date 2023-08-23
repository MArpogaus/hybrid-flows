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

from .utils.tensorflow import get_simple_fully_connected_network

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


def __get_unconditional_trainable_distribution__(
    get_distribution_lambda_fn, dims, dtype=tf.float32, **distribution_kwds
):
    distribution_lambda, parameters_shape = get_distribution_lambda_fn(
        dims=dims, **distribution_kwds
    )
    trainable_parameters = tf.Variable(
        tf.random.normal(parameters_shape, dtype=dtype), trainable=True
    )
    return distribution_lambda, trainable_parameters


def __get_conditional_trainable_distribution__(
    get_distribution_lambda_fn, dims, conditioner, **distribution_kwds
):
    raise NotImplementedError()


def __get_trainable_distribution__(
    get_disttribution_lambda_fn, dims, conditioner=None, **kwds
):
    if conditioner:
        return __get_conditional_trainable_distribution__(
            get_disttribution_lambda_fn, dims=dims, conditioner=conditioner, **kwds
        )
    else:
        return __get_unconditional_trainable_distribution__(
            get_disttribution_lambda_fn, dims=dims, **kwds
        )


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
    order,
    hidden_units,
    activation,
    base_distribution=__DEFAULT_BASE_DISTRIBUTION__,
    conditional=False,
    conditional_event_shape=None,
    conditional_input_layers="all_layers",
    **kwds,
):
    made_net = tfb.AutoregressiveNetwork(
        params=order,
        hidden_units=hidden_units,
        event_shape=(dims,),
        activation=activation,
        conditional=conditional,
        conditional_event_shape=conditional_event_shape,
        conditional_input_layers=conditional_input_layers,
    )
    made_net.build((dims,))

    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    bijector_fn = __get_bijector_fn__(
        network=made_net, thetas_constrain_fn=thetas_constrain_fn
    )

    distribution = tfd.TransformedDistribution(
        distribution=tfd.Sample(base_distribution, sample_shape=[dims]),
        bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn),
    )

    return lambda _: distribution, made_net.trainable_variables


def get_coupling_bernstein_flow(
    dims,
    order,
    hidden_units,
    activation,
    batch_norm,
    coupling_layers,
    base_distribution=__DEFAULT_BASE_DISTRIBUTION__,
    **kwds,
):
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    trainable_variables = []
    bijectors = []
    for layer in range(coupling_layers):
        network = get_simple_fully_connected_network(
            input_shape=dims // 2,
            hidden_units=hidden_units,
            activation=activation,
            batch_norm=batch_norm,
            output_shape=(dims // 2, order),
        )
        trainable_variables += network.trainable_variables
        bijectors.append(
            tfb.RealNVP(
                num_masked=(dims // 2),
                bijector_fn=__get_bijector_fn__(network, thetas_constrain_fn),
            )
        )
        if coupling_layers % 2 != 0 and layer == (coupling_layers - 1):
            print("uneven number of coupling layers -> skipping last permuataion")
        else:
            bijectors.append(tfb.Permute(permutation=[1, 0]))

    distribution = tfd.TransformedDistribution(
        distribution=tfd.Sample(base_distribution, sample_shape=[dims]),
        bijector=tfb.Chain(bijectors),
    )

    return lambda _: distribution, trainable_variables


get_bernstein_flow = partial(
    __get_trainable_distribution__,
    __get_bernstein_flow_lambda__,
)
get_multivariate_bernstein_flow = partial(
    __get_trainable_distribution__,
    __get_multivariate_bernstein_flow_lambda__,
)
get_multivariate_normal = partial(
    __get_trainable_distribution__,
    __get_multivariate_normal_lambda__,
)
