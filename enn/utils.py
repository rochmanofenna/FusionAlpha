import torch

def entropy_based_prune(neuron_state, importance_threshold=0.1):
    return torch.where(neuron_state > importance_threshold, neuron_state, torch.tensor(0.0))

def context_collapse(neuron_state, threshold=0.5):
    entropy = -torch.sum(neuron_state * torch.log(neuron_state + 1e-9))
    if entropy < threshold:
        return torch.mean(neuron_state)
    return neuron_state
