import torch

def compute_data_complexity(data_input):
    complexity_score = torch.var(data_input)
    return torch.clamp(complexity_score, 0, 1)

def dynamic_scaling(data_complexity, max_states=5):
    num_active_states = int(data_complexity * max_states)
    return max(num_active_states, 1)

def activate_neuron(neuron_state, data_input):
    complexity_score = compute_data_complexity(data_input)
    active_states = dynamic_scaling(complexity_score)
    neuron_state[:active_states] = torch.sigmoid(data_input)[:active_states]
    return neuron_state
