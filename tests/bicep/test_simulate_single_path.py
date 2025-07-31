# tests/test_simulate_single_path.py
import numpy as np
from backends.bicep.brownian_motion import simulate_single_path
from backends.bicep.stochastic_control import apply_stochastic_controls

def test_simulate_single_path_basic():
    path = simulate_single_path(1, 100, 0, 0.01, 0, None, np, apply_stochastic_controls)
    assert len(path) == 101, "Path should match n_steps + 1 in length."

def test_simulate_single_path_minimal():
    path = simulate_single_path(1, 1, 0, 1, 0, None, np, apply_stochastic_controls)
    assert len(path) == 2, "Path should handle single-step gracefully."

def test_simulate_single_path_with_controls():
    path = simulate_single_path(1, 100, 0, 0.01, 0, None, np, apply_stochastic_controls)
    assert path[0] == 0, "First path value should match initial value."
