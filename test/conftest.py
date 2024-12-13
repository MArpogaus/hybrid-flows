# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : conftest.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-10-31 13:57:25 (Marcel Arpogaus)
# changed : 2024-10-31 13:57:25 (Marcel Arpogaus)

# %% License ###################################################################

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
