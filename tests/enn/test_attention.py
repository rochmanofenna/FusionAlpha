import torch
from enn.attention import attention_gate, probabilistic_path_activation

def test_attention_gate():
    neuron_activations = torch.rand(10, 10)
    gated_activations = attention_gate(neuron_activations, threshold=0.5)
    assert gated_activations.shape == neuron_activations.shape
    assert torch.all(gated_activations[torch.isnan(gated_activations)] == 0)  # Check that non-relevant states are zeroed

def test_probabilistic_path_activation():
    neuron_activations = torch.rand(10, 10)
    activated_paths = probabilistic_path_activation(neuron_activations, activation_probability=0.5)
    assert activated_paths.shape == neuron_activations.shape
    # Check randomness in activation
    assert torch.sum(activated_paths > 0) > 0
