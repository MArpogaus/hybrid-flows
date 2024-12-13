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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import mctm.scheduler

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# PUBLIC FUNCTIONS #############################################################
def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility in NumPy and TensorFlow.

    Parameters
    ----------
    seed : int
        The random seed value to set.

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
    learning_rate = fit_kwargs["learning_rate"]
    if isinstance(learning_rate, dict):
        scheduler_name = learning_rate.get("scheduler_name", "")
        scheduler_class_name = "".join(scheduler_name.title().split("_"))
        scheduler_kwargs = learning_rate.get("scheduler_kwargs", {})
        __LOGGER__.info(f"Scheduler: {scheduler_name}({scheduler_kwargs})")

        scheduler = getattr(
            mctm.scheduler,
            scheduler_class_name,
            getattr(K.optimizers.schedules, scheduler_class_name, None),
        )(**scheduler_kwargs)

        fit_kwargs["callbacks"] = [K.callbacks.LearningRateScheduler(scheduler)]
        return scheduler_kwargs["initial_learning_rate"], learning_rate
    else:
        return learning_rate, {}


def fit_distribution(
    model: K.Model,
    seed: int,
    learning_rate: float,
    results_path: str,
    monitor: str,
    verbose: bool,
    reduce_lr_on_plateau: bool,
    early_stopping: bool,
    lr_patience: Optional[int] = None,
    lr_reduction_factor: Optional[float] = None,
    weight_decay: Optional[float] = None,
    loss: Callable = lambda y, dist: -dist.log_prob(y),
    callbacks: list = [],
    compile_kwargs: dict = {},
    **fit_kwargs: Any,
) -> K.callbacks.History:
    """Train model.

    This function compiles and fits a probability distribution model using
    the specified settings, including callbacks for early stopping and
    learning rate reduction.

    Parameters
    ----------
    model : K.Model
        The probability distribution model to fit.
    seed : int
        The random seed for reproducibility.
    learning_rate : float
        The learning rate for the optimizer.
    results_path : str
        The path to save model weights and artifacts.
    monitor : str
        The monitored metric for early stopping and learning rate reduction.
    verbose : bool
        The verbosity level for training.
    reduce_lr_on_plateau : bool
        Whether to reduce the learning rate on plateau.
    early_stopping : bool
        Whether to enable early stopping.
    lr_patience : int, optional
        The patience parameter for learning rate reduction.
    lr_reduction_factor : float, optional
        Factor by which the learning rate will be reduced. new_lr = lr * factor.
    weight_decay : float, optional
        Weight decay for the optimizer.
    loss : callable
        The loss function for the model.
    callbacks : list
        Additional training callbacks.
    compile_kwargs : dict
        Additional compilation parameters.
    fit_kwargs: dict
        Additional keyword arguments passed to `Model.fit`.

    Returns
    -------
    K.callbacks.History
        The training history of the fitted model.

    """
    set_seed(seed)
    if weight_decay:
        optimizer = K.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)

    callbacks.extend(
        [
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
    )

    if reduce_lr_on_plateau:
        callbacks.append(
            K.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=lr_reduction_factor or 0.1,
                patience=lr_patience or 10,
                verbose=verbose,
            )
        )

    if early_stopping:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor=monitor,
                patience=(early_stopping if isinstance(early_stopping, int) else 3)
                * (lr_patience if reduce_lr_on_plateau else 1),
                restore_best_weights=True,
                verbose=verbose,
            )
        )

    return model.fit(
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
        **fit_kwargs,
    )
