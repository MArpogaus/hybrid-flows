# %% imports

import tensorflow as tf
from mctm.data.sklearn_datasets import get_dataset
from tensorflow import keras as K
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tqdm import trange


# %% functions
def get_param_net(hidden_layers, params):
    network = K.Sequential(
        [K.layers.Input((dims // 2))]
        + [K.layers.Dense(d, activation="relu") for d in hidden_layers]
        + [K.layers.Dense(params)]
    )

    return network


def get_shift_and_log_scale_fn(hidden_layers):
    network = get_param_net(hidden_layers, 2)

    shift_and_log_scale_fn = lambda x, input_depth: tf.split(
        network(x), 2 * input_depth, axis=-1
    )
    return shift_and_log_scale_fn, network.trainable_variables


def fit_real_nvp(
    data,
    dims,
    epochs,
    seed,
    learning_rate,
    coupling_layers,
    hidden_units,
    activation,
    distribution,
):
    set_seed(seed)

    opt = tf.optimizers.Adam(learning_rate)

    trainable_variables = []
    bijectors = []
    for _ in range(coupling_layers):
        shift_and_log_scale_fn, variables = get_shift_and_log_scale_fn(hidden_units)
        trainable_variables += variables
        bijectors.append(
            tfb.RealNVP(
                num_masked=(dims // 2), shift_and_log_scale_fn=shift_and_log_scale_fn
            )
        )
        bijectors.append(tfb.Permute(permutation=[1, 0]))

    deep_nvp = tfd.TransformedDistribution(
        distribution=distribution,
        bijector=tfb.Chain(bijectors),
    )

    print("Number of Parameters:", sum(map(tf.size, trainable_variables)).numpy())

    it = trange(epochs)
    for i in it:
        nll = lambda: -deep_nvp.log_prob(data)
        opt.minimize(nll, var_list=trainable_variables)
        it.set_postfix(loss=nll().numpy().mean(), lr=get_lr(opt, i))

    return nll().numpy().mean(), deep_nvp


# %% get data
(y, x), dims = get_dataset("moons", n_samples=2000, scale=(0, 1))


# %% train
nll, nvp = fit_real_nvp(
    data,
    dims=dims,
    epochs=200,
    seed=1,
    learning_rate=0.005,
    hidden_units=(16, 16),
    coupling_layers=4,
    activation="relu",
    distribution=tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=[2]),
)
