# tests/test_calculate_optimal_parameters.py
from backends.bicep.brownian_motion import calculate_optimal_parameters

def test_calculate_optimal_parameters_normal():
    resources = (8, 4, True, 16)  # 8GB RAM, 4 CPUs, GPU with 16GB memory
    batch_size, save_interval, gpu_threshold = calculate_optimal_parameters(1000, 100, *resources)
    assert batch_size > 0, "Batch size should be positive."
    assert save_interval >= 0, "Save interval should be non-negative."
    assert gpu_threshold >= 1000, "GPU threshold should be reasonable."

def test_calculate_optimal_parameters_low_resources():
    resources = (2, 2, False, 0)  # Low RAM, CPU, no GPU
    batch_size, save_interval, gpu_threshold = calculate_optimal_parameters(500, 50, *resources)
    assert batch_size <= 500, "Batch size should adapt to low resources."
    assert gpu_threshold > 500, "GPU threshold should default above n_paths if no GPU."
