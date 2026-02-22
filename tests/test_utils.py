"""
Basic tests for shared utilities.

Run with:  pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import set_seed, moving_average
import numpy as np


def test_set_seed_no_error():
    """set_seed should run without raising an exception."""
    set_seed(0)


def test_moving_average_length():
    values = list(range(20))
    window = 5
    result = moving_average(values, window)
    assert len(result) == len(values) - window + 1


def test_moving_average_values():
    values = [1.0] * 10
    result = moving_average(values, window=3)
    assert np.allclose(result, 1.0)


def test_base_agent_raises():
    from src.agents.base_agent import BaseAgent
    agent = BaseAgent()
    try:
        agent.select_action(None)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass
