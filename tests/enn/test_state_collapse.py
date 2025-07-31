import torch
from enn.state_collapse import entropy_based_pruning, interference_based_adjustment, advanced_state_collapse, StateAutoEncoder

def test_entropy_based_pruning():
    neuron_states = torch.rand(10)
    pruned_states = entropy_based_pruning(neuron_states, importance_threshold=0.2)
    assert pruned_states.shape == neuron_states.shape
    assert torch.allclose(pruned_states[neuron_states < 0.2], torch.zeros_like(pruned_states[neuron_states < 0.2]), atol=1e-1)



def test_interference_based_adjustment():
    neuron_states = torch.rand(10)
    adjusted_states = interference_based_adjustment(neuron_states)
    assert adjusted_states.shape == neuron_states.shape

def test_advanced_state_collapse():
    autoencoder = StateAutoEncoder(input_dim=10, compressed_dim=5)
    neuron_states = torch.rand(10)
    collapsed_states = advanced_state_collapse(neuron_states, autoencoder, importance_threshold=0.1)
    assert collapsed_states.shape[0] == 5
