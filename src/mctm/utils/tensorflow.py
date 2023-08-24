# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : tensorflow.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-23 15:52:34 (Marcel Arpogaus)
# changed : 2023-08-23 15:52:34 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORT MODULES ###############################################################
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as K


# PUBLIC FUNCTIONS #############################################################
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Construct and fit model.
# @tf.function
def fit_distribution(
    model,
    seed,
    learning_rate,
    lr_patience,
    results_path,
    monitor,
    verbose,
    loss=lambda y, dist: -dist.log_prob(y),
    **kwds,
):
    set_seed(seed)
    print("start debug")
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    callbacks = [
        K.callbacks.ModelCheckpoint(
            os.path.join(results_path, "mcp/weights"),
            monitor="val_loss",
            mode="auto",
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=lr_patience,
            verbose=verbose,
        ),
        K.callbacks.EarlyStopping(
            monitor=monitor,
            patience=3 * lr_patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        K.callbacks.TerminateOnNaN(),
    ]
    kwds["x"]
    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        **kwds,
    )
