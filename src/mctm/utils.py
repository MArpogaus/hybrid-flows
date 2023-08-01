# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : utils.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 14:44:17 (Marcel Arpogaus)
# changed : 2023-06-19 17:08:07 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
from functools import partial

import numpy as np
import tensorflow as tf

from .models import get_parameter_model


# FUNCTION DEFINITIONS #########################################################
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Construct and fit model.
# @tf.function
def fit_distribution(
    model,
    learning_rate=0.001,
    lr_patience=10,
    monitor="loss",
    loss=lambda y, dist: -dist.log_prob(y),
    **kwds,
):
    set_seed(1)
    print("start debug")
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=lr_patience,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=3 * lr_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    ds = kwds["x"]
    xa, xb = next(iter(ds))
    print(xa.shape, xb.shape)
    y = model.predict(xa)
    print(y.shape)
    # l = loss(xa, )
    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        **kwds,
    )


def pipeline(dist, dist_keywords, output_shape, ds_train, preprocessing):
    model_params = {
        "input_shape": (1,),
        "hidden_layers": [16, 16],
        "activation": "relu",
        "batch_norm": False,
    }

    params = {
        "epochs": 5,
        "learning_rate": 0.00001,
        "lr_patience": 10,
    }

    # get_model, get_dist, get_data + respective keywords
    # + weitere an fit_distribution + fit_distribution keywords
    # + postfix_fn eg

    # create model from dist
    P = partial(get_parameter_model, **model_params)
    model = P(
        output_shape=output_shape,
        dist_lambda=dist(**dist_keywords),
    )

    ds_train = preprocessing(ds_train)
    print(ds_train)
    # train model
    hist = fit_distribution(
        model,
        x=ds_train,
        # validation_data=(val_x, val_y),
        # batch_size=32,
        # steps_per_epoch=2,
        **params,
    )

    return model, hist
