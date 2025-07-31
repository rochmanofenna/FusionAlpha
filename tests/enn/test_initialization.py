import torch
from enn.initialization import context_aware_initialization

def test_context_aware_initialization():
    num_neurons, num_states = 10, 5
    entanglement_matrix = torch.ones(num_neurons, num_states)
    weights = context_aware_initialization(num_neurons, num_states, entanglement_matrix, method="xavier")
    assert weights.shape == (num_neurons, num_states)
    assert torch.all(weights >= -3) and torch.all(weights <= 3)
