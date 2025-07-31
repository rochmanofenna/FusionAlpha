import torch

def context_aware_initialization(num_neurons, num_states, entanglement_matrix, method="xavier"):
    """
    Initializes weights with context-aware entanglement patterns.

    Parameters:
    - num_neurons: Number of neurons.
    - num_states: Number of states per neuron.
    - entanglement_matrix: A matrix indicating entanglement strength between neurons.
    - method: Initialization method, "xavier" or "he".

    Returns:
    - Initialized weight tensor.
    """
    if method == "xavier":
        scale = torch.sqrt(torch.tensor(2.0) / (num_neurons + num_states))
    elif method == "he":
        scale = torch.sqrt(torch.tensor(2.0) / num_neurons)
    else:
        raise ValueError("Unsupported initialization method")

    # Generate weights based on entanglement strength
    weights = scale * torch.randn(num_neurons, num_states)
    for i in range(num_neurons):
        for j in range(num_states):
            weights[i, j] *= entanglement_matrix[i, j] if entanglement_matrix[i, j] > 0 else 1.0

    return weights
