"""Tensorflow utils."""
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : tensorflow.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-23 15:52:34 (Marcel Arpogaus)
# changed : 2023-10-06 11:01:38 (Marcel Arpogaus)
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
    """Set the random seed for reproducibility in NumPy and TensorFlow.

    :param int seed: The random seed value to set.
    """
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
    reduce_lr_on_plateau,
    early_stopping,
    loss=lambda y, dist: -dist.log_prob(y),
    callbacks=[],
    **kwds,
):
    """Train model.

    This function compiles and fits a probability distribution model using
    the specified settings,
    including callbacks for early stopping and learning rate reduction.

    :param model: The probability distribution model to fit.
    :param int seed: The random seed for reproducibility.
    :param float learning_rate: The learning rate for the optimizer.
    :param int lr_patience: The patience parameter for learning rate reduction.
    :param str results_path: The path to save model weights and artifacts.
    :param str monitor: The monitored metric for early stopping and
                       learning rate reduction.
    :param int verbose: The verbosity level for training.
    :param bool reduce_lr_on_plateau: Whether to reduce the learning rate on plateau.
    :param bool early_stopping: Whether to enable early stopping.
    :param callable loss: The loss function for the model.
    :param list callbacks: Additional training callbacks.
    :param **kwds: Additional keyword arguments for the model fitting function.
    :return: The training history of the fitted model.
    :rtype: object
    """
    set_seed(seed)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    callbacks += [
        K.callbacks.ModelCheckpoint(
            os.path.join(results_path, "mcp/weights"),
            monitor=monitor,
            mode="auto",
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
        ),
        K.callbacks.TerminateOnNaN(),
    ]
    if reduce_lr_on_plateau:
        callbacks += [
            K.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.1,
                patience=lr_patience,
                verbose=verbose,
            )
        ]
    if early_stopping:
        callbacks += [
            K.callbacks.EarlyStopping(
                monitor=monitor,
                patience=3 * lr_patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        ]
    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
        **kwds,
    )
