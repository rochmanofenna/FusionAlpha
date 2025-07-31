import torch
from enn.sparsity_control import dynamic_sparsity_control, event_trigger, low_power_state_collapse

def test_dynamic_sparsity_control():
    neuron_states = torch.rand(10)
    sparse_states = dynamic_sparsity_control(neuron_states, sparsity_threshold=0.5)
    assert sparse_states.shape == neuron_states.shape
    assert torch.all(sparse_states[neuron_states < 0.5] == 0)

def test_event_trigger():
    input_data = torch.rand(10)
    assert event_trigger(input_data, threshold=0.5) in [True, False]  # Ensure it returns a boolean

def test_low_power_state_collapse():
    neuron_states = torch.rand(10)
    collapsed_states = low_power_state_collapse(neuron_states, top_k=3)
    assert torch.sum(collapsed_states > 0) <= 3  # Only top-k should remain active
