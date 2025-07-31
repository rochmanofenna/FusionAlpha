# tests/test_stress.py
import pytest
from backends.bicep.brownian_motion import brownian_motion_paths

def test_large_paths():
    time, paths = brownian_motion_paths(T=1, n_steps=100, initial_value=0, n_paths=5000)
    assert paths.shape == (5000, 101), "Large path test should complete without error."
