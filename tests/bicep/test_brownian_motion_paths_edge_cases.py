# tests/test_brownian_motion_paths_edge_cases.py
import pytest
from backends.bicep.brownian_motion import brownian_motion_paths

def test_brownian_motion_paths_zero_time():
    with pytest.raises(ValueError):
        brownian_motion_paths(T=0, n_steps=100, initial_value=0, n_paths=10)

def test_brownian_motion_paths_negative_steps():
    with pytest.raises(ValueError):
        brownian_motion_paths(T=1, n_steps=-10, initial_value=0, n_paths=10)
