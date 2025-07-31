import torch
from enn.memory import state_decay, reset_neuron_state, ShortTermBuffer, temporal_proximity_scaling

def test_state_decay():
    neuron_state = torch.ones(10)
    decayed_state = state_decay(neuron_state, decay_rate=0.1)
    assert torch.all(decayed_state < neuron_state)  # Ensure decay occurs

def test_reset_neuron_state():
    neuron_state = torch.ones(10)
    reset_state = reset_neuron_state(neuron_state)
    assert torch.all(reset_state == 0)

def test_short_term_buffer():
    buffer = ShortTermBuffer(buffer_size=3)
    buffer.add_to_buffer(torch.tensor(1))
    buffer.add_to_buffer(torch.tensor(2))
    assert buffer.get_recent_activations() == [1, 2]
    buffer.add_to_buffer(torch.tensor(3))
    buffer.add_to_buffer(torch.tensor(4))
    assert buffer.get_recent_activations() == [2, 3, 4]  # Check FIFO behavior

def test_temporal_proximity_scaling():
    weights = torch.ones(5)
    scaled_weights = temporal_proximity_scaling(weights, recency_factor=0.9)
    assert scaled_weights[0] > scaled_weights[-1]  # Ensure recency priority
