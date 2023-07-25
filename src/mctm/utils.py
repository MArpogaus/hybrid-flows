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
import tensorflow as tf
import numpy as np

from functools import partial
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
    lr_patience=5,
    loss=lambda y, dist: -dist.log_prob(y),
    **kwds,
):
    set_seed(1)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=lr_patience,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3 * lr_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        **kwds,
    )

def pipeline(dist, latentdim, ds_train, preprocessing):
    params = {
        "latentdim": latentdim,
        "hidden_layers": [16],
        "activation": "relu",
        "batch_norm": True,
        "epochs": 500,
        "learning_rate": 0.001, 
        "lr_patience": 20,
    }
    
    # create model from dist
    P = partial(get_parameter_model, input_shape=(1,), **params)
    model = P(
        output_shape=latentdim + np.sum(np.arange(latentdim + 1)),
        dist_lambda=dist(latentdim),
    )
    
    ds_train = preprocessing(ds_train)
    
    # train model
    hist = fit_distribution(
    model,
    x=ds_train,
    # validation_data=(val_x, val_y),
    #batch_size=32,
    epochs=params["epochs"],
    # steps_per_epoch=2,
    learning_rate=params["learning_rate"],
    lr_patience=params["lr_patience"],
    )
    
    return model, hist