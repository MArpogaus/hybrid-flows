# %% Description ###############################################################
"""Definitions shared accros all tests."""

# %% imports ###################################################################
import pytest
import tensorflow as tf


# %% fixtures ##################################################################
@pytest.fixture(scope="session", autouse=True)
def disable_gpu():
    """Disable GPU usage at the beginning of the session."""
    tf.random.set_seed(1)
    tf.config.set_visible_devices([], "GPU")


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed at the beginning of all tests."""
    tf.random.set_seed(1)
