import tensorflow as tf


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
        with tf.name_scope(self.__class__.__name__):
            step = tf.convert_to_tensor(step)
            warmup = tf.convert_to_tensor(self.warmup, dtype=step.dtype)
            stationary = tf.convert_to_tensor(self.stationary, dtype=step.dtype)
            tf.print(step)
            # if step <= self.warmup:
            #     return self.warmup_scheduler(step)
            # elif (step > self.warmup) & (step <= self.warmup + self.stationary):
            #     return self.max_learning_rate
            # else:
            #     return self.cooldown_scheduler(step - self.warmup - self.stationary)
            is_warmup = step <= warmup

            def warmup_fn():
                return self.warmup_scheduler(step)

            is_stationary = ~is_warmup & (step <= warmup + stationary)

            def stationary_fn():
                return self.max_learning_rate

            def cooldown_fn():
                return self.cooldown_scheduler(step - warmup - stationary)

            tf.print(is_warmup, is_stationary)
            lr = tf.case(
                [(is_warmup, warmup_fn), (is_stationary, stationary_fn)],
                default=cooldown_fn,
                exclusive=True,
            )

            tf.print(lr)

            return lr
