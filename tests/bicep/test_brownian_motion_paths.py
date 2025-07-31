# tests/test_brownian_motion_paths.py
import pytest
from backends.bicep.brownian_motion import brownian_motion_paths

def test_brownian_motion_paths_basic():
    time, paths = brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=10)
    assert len(time) == 101, "Time array should match n_steps + 1."
    assert paths.shape == (10, 101), "Paths shape should match (n_paths, n_steps + 1)."

def test_brownian_motion_paths_zero_paths():
    time, paths = brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=0)
    assert paths.shape == (0, 101), "Paths with n_paths=0 should handle gracefully."

def test_brownian_motion_paths_single_step():
    time, paths = brownian_motion_paths(T=1, n_steps=1, initial_value=0, n_paths=10)
    assert len(time) == 2, "Time array should have 2 elements when n_steps=1."
    assert paths.shape == (10, 2), "Paths should handle single-step paths."
