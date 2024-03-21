# %% imports
import inspect

import tensorflow as tf
from matplotlib import pyplot as plt

# %% globals
decay_steps = 1000
common_kwds = dict(initial_learning_rate=1.0)


# %% functions
def get_scheduler_by_name(scheduler_name, decay_steps, **scheduler_kwds):
    scheduler_class = getattr(tf.keras.optimizers.schedules, scheduler_name)
    if "decay_steps" in inspect.signature(scheduler_class).parameters:
        scheduler_kwds.update(decay_steps=decay_steps)
    return scheduler_class(**scheduler_kwds)


def plot_scheduler(scheduler_name, decay_steps, **scheduler_kwds):
    if isinstance(scheduler_name, str):
        scheduler = get_scheduler_by_name(
            scheduler_name, decay_steps=decay_steps, **scheduler_kwds
        )
    else:
        scheduler = scheduler_name(decay_steps=decay_steps, **scheduler_kwds)
        scheduler_name = scheduler.__class__.__name__

    steps = list(range(decay_steps + 100))
    lr = [scheduler(step) for step in steps]

    fig = plt.figure()
    plt.plot(steps, lr)
    plt.title(scheduler_name)
    return fig


# %% CosineDecay
fig = plot_scheduler("CosineDecay", decay_steps, alpha=0.5, **common_kwds)
fig.show()

# %% CosineDecayRepeats
fig = plot_scheduler(
    "CosineDecayRestarts", decay_steps, first_decay_steps=100, **common_kwds
)
fig.show()

# %% InverseTimeDecay
fig = plot_scheduler("InverseTimeDecay", decay_steps, decay_rate=2, **common_kwds)
fig.show()

# %% PiecewiseConstantDecay
fig = plot_scheduler(
    "PiecewiseConstantDecay",
    decay_steps,
    boundaries=[100, 500],
    values=[1.0, 0.5, 0.1],
)
fig.show()

# %% ExponentialDecay
fig = plot_scheduler("ExponentialDecay", decay_steps, decay_rate=0.01, **common_kwds)
fig.show()

# %% PolynomialDecay
fig = plot_scheduler(
    "PolynomialDecay", decay_steps, end_learning_rate=0.9, power=0.2, **common_kwds
)
fig.show()


# %% MySchdule
class PolynomialWarmupAndCosineDecay(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """A tensorflow class implementing a custom learning rate scheduler.

    This scheduler starts with a polynomial warmup to the maximum learning rate,
    followed by a stationary period at the maximum learning rate, and finally
    a cosine decay to the end learning rate.

    Attributes
    ----------
    initial_learning_rate : float
        The initial learning rate at the start of the warmup.
    max_learning_rate : float
        The maximum learning rate after the warmup.
    end_learning_rate : float
        The end learning rate after the cosine decay.
    warmup : int
        The number of steps for the warmup.
    stationary : int
        The number of steps to hold the maximum learning rate.
    decay_steps : int
        The total number of training steps.
    warmup_power : float
        The exponent for polynomial warmup.
    warmup_scheduler : tf.keras.optimizers.schedules.PolynomialDecay
        A scheduler for the warmup period.
    cooldown_scheduler : tf.keras.optimizers.schedules.CosineDecay
        A scheduler for the cooldown period.

    """

    def __init__(
        self,
        initial_learning_rate: float,
        max_learning_rate: float,
        end_learning_rate: float,
        warmup: int,
        stationary: int,
        decay_steps: int,
        warmup_power: float,
    ) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.warmup = warmup
        self.stationary = stationary
        self.max_learning_rate = max_learning_rate
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            end_learning_rate=max_learning_rate,
            decay_steps=warmup,
            power=warmup_power,
        )
        self.cooldown_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=max_learning_rate,
            alpha=end_learning_rate / max_learning_rate,
            decay_steps=decay_steps - warmup - stationary,
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Compute the learning rate for the current training step.

        The learning rate starts with a polynomial warmup, holds at
        the maximum learning rate, and decays with cosines.

        Parameters
        ----------
        step : tf.Tensor
            The current training step.

        Returns
        -------
        tf.Tensor
            The learning rate for the current training step.

        """
        # if step <= self.warmup:
        #     return self.warmup_scheduler(step)
        # elif (step > self.warmup) & (step <= self.warmup + self.stationary):
        #     return self.max_learning_rate
        # else:
        #     return self.cooldown_scheduler(step - self.warmup - self.stationary)
        is_warmup = tf.less_equal(step, self.warmup)

        def warmup():
            return self.warmup_scheduler(step)

        is_stationary = ~is_warmup & tf.less_equal(step, self.warmup + self.stationary)

        def stationary():
            return self.max_learning_rate

        def cooldown():
            return self.cooldown_scheduler(step - self.warmup - self.stationary)

        return tf.case(
            [(is_warmup, warmup), (is_stationary, stationary)],
            default=cooldown,
            exclusive=True,
        )


fig = plot_scheduler(
    PolynomialWarmupAndCosineDecay,
    decay_steps=20,
    initial_learning_rate=0.0001,
    max_learning_rate=0.1,
    end_learning_rate=0.00001,
    warmup=5,
    warmup_power=3,
    stationary=10,
)
fig.show()
