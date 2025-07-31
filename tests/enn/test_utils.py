import torch
from enn.utils import entropy_based_prune, context_collapse

def test_entropy_based_prune():
    neuron_state = torch.rand(10)
    pruned_state = entropy_based_prune(neuron_state, importance_threshold=0.5)
    assert pruned_state.shape == neuron_state.shape
    assert torch.all(pruned_state[neuron_state < 0.5] == 0)

def test_context_collapse():
    neuron_state = torch.rand(10)
    collapsed_state = context_collapse(neuron_state, threshold=0.5)
    assert collapsed_state.shape == neuron_state.shape
