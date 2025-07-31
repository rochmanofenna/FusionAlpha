import torch

def dynamic_weight_sharing(neuron_activations, weights, attention_threshold=0.6):
    attention_scores = torch.matmul(neuron_activations, neuron_activations.T) / neuron_activations.size(-1) ** 0.5
    attention_mask = (attention_scores > attention_threshold).float()[:, :weights.size(1)]
    shared_weights = weights * attention_mask
    return shared_weights


