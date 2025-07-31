# tests/test_detect_system_resources.py
from backends.bicep.brownian_motion import detect_system_resources

def test_detect_system_resources():
    memory, cpu_count, gpu_available, gpu_memory = detect_system_resources()
    assert memory > 0, "Memory should be detected as positive."
    assert cpu_count > 0, "CPU count should be positive."
    assert isinstance(gpu_available, bool), "GPU availability should be a boolean."
