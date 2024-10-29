# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : tensorflow.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-10-29 12:55:08 (Marcel Arpogaus)
# changed : 2024-10-29 12:55:08 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Tensorflow utils."""

# %% imports ###################################################################
import logging
import os
from typing import Any, Dict, Tuple, Union

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import mctm.scheduler

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# PUBLIC FUNCTIONS #############################################################
def set_seed(seed: int):
    """Set the random seed for reproducibility in NumPy and TensorFlow.

    :param int seed: The random seed value to set.
    """
    # This sets the Python seed, the NumPy seed, and the TensorFlow seed.
    K.utils.set_random_seed(seed)
    # When op determinism is enabled, TensorFlow ops will be deterministic.
    # This means that if an op is run multiple times with the same inputs on the
    # same hardware, it will have the exact same outputs each time.
    # NOTE: that determinism in general comes at the expense of lower performance
    #       and so your model may run slower when op determinism is enabled.
    tf.config.experimental.enable_op_determinism()


def get_learning_rate(
    fit_kwargs: Dict[str, Any],
) -> Tuple[Union[float, LearningRateSchedule], Dict[str, Any]]:
    """Get learning rate scheduler.

    Parameters
    ----------
    fit_kwargs : dict
        Dictionary containing fit parameters.

    Returns
    -------
    tuple
        Initial learning rate and learning rate scheduler.

    """
    if isinstance(fit_kwargs["learning_rate"], dict):
        scheduler_name = fit_kwargs["learning_rate"]["scheduler_name"]
        schduler_class_name = "".join(map(str.title, scheduler_name.split("_")))
        scheduler_kwargs = fit_kwargs["learning_rate"]["scheduler_kwargs"]
        __LOGGER__.info(f"{scheduler_name=}({scheduler_kwargs})")

        scheduler = getattr(
            mctm.scheduler,
            schduler_class_name,
            getattr(K.optimizers.schedules, schduler_class_name, None),
        )(**scheduler_kwargs)

        fit_kwargs["callbacks"] = [K.callbacks.LearningRateScheduler(scheduler)]
        return scheduler_kwargs["initial_learning_rate"], fit_kwargs["learning_rate"]
    else:
        return fit_kwargs["learning_rate"], {}


# Construct and fit model.
# TODO: implement a custom training loop
def fit_distribution(
    model: K.Model,
    seed: int,
    learning_rate: float,
    results_path: str,
    monitor: str,
    verbose: bool,
    reduce_lr_on_plateau: bool,
    early_stopping: bool,
    lr_patience: int = None,
    lr_reduction_factor: float = None,
    weight_decay: float = None,
    loss=lambda y, dist: -dist.log_prob(y),
    callbacks=[],
    compile_kwargs={},
    **fit_kwargs,
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
    :param int lr_reduction_factor: Factor by which the learning rate will be
                                    reduced. new_lr = lr * factor.
    :param callable loss: The loss function for the model.
    :param list callbacks: Additional training callbacks.
    :return: The training history of the fitted model.
    :rtype: object

    """
    set_seed(seed)
    if weight_decay:
        optimizer = K.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)

    callbacks += [
        K.callbacks.ModelCheckpoint(
            os.path.join(results_path, "model_checkpoint.weights.h5"),
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
                factor=lr_reduction_factor,
                patience=lr_patience,
                verbose=verbose,
            )
        ]
    if early_stopping:
        callbacks += [
            K.callbacks.EarlyStopping(
                monitor=monitor,
                patience=(early_stopping if isinstance(early_stopping, int) else 3)
                * (lr_patience if reduce_lr_on_plateau else 1),
                restore_best_weights=True,
                verbose=verbose,
            )
        ]
    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
        **fit_kwargs,
    )
