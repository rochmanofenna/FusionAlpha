import torch
import torch.nn.functional as F

def attention_gate(neuron_activations, threshold=0.5):
    query = neuron_activations
    key = neuron_activations.transpose(0, 1)
    
    # Calculate attention scores and ensure it has the shape [10, 5]
    attention_scores = F.softmax(torch.matmul(query, key) / (neuron_activations.size(-1) ** 0.5), dim=-1)
    attention_scores = attention_scores[:, :neuron_activations.size(1)]  # Adjust size to match neuron_activations
    
    gated_activations = torch.where(attention_scores > threshold, neuron_activations, torch.zeros_like(neuron_activations))
    return gated_activations



def probabilistic_path_activation(neuron_activations, activation_probability=0.2):
    """
    Activates neuron paths probabilistically to introduce controlled randomness.

    Parameters:
    - neuron_activations: Tensor representing the neuron activations in the layer.
    - activation_probability: Probability of activating a neuron path.

    Returns:
    - activated_paths: Neuron activations with probabilistic paths applied.
    """
    # Generate a random mask based on the activation probability
    random_mask = torch.rand(neuron_activations.shape) < activation_probability
    activated_paths = neuron_activations * random_mask.float()
    return activated_paths
