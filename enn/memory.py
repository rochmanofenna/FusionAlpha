import torch
from collections import deque

def state_decay(neuron_state, decay_rate=0.1):
    """
    Applies decay to a neuron's state to reduce the impact of older data.

    Parameters:
    - neuron_state: Tensor representing the neuron's current state.
    - decay_rate: Rate at which the state decays over time.

    Returns:
    - Decayed neuron state.
    """
    return neuron_state * torch.exp(torch.tensor(-decay_rate))

def reset_neuron_state(neuron_state):
    """
    Resets the neuron's state, effectively 'forgetting' outdated entanglements.

    Parameters:
    - neuron_state: Tensor representing the neuron's current state.

    Returns:
    - Reset neuron state (zeroed out).
    """
    return torch.zeros_like(neuron_state)

class ShortTermBuffer:
    def __init__(self, buffer_size=5):
        """
        Initializes a short-term buffer with a limited capacity.

        Parameters:
        - buffer_size: Maximum number of recent activations to store.
        """
        self.buffer = deque(maxlen=buffer_size)

    def add_to_buffer(self, activation):
        """
        Adds a new activation to the buffer, maintaining a short-term history.

        Parameters:
        - activation: New neuron activation to add to the buffer.
        """
        self.buffer.append(activation)

    def get_recent_activations(self):
        """
        Retrieves recent activations stored in the buffer.

        Returns:
        - List of recent activations.
        """
        return list(self.buffer)

def temporal_proximity_scaling(weights, recency_factor=0.9):
    """
    Scales weights based on temporal proximity to prioritize recent connections.

    Parameters:
    - weights: Tensor of weights for neuron connections.
    - recency_factor: Scaling factor where recent weights are prioritized.

    Returns:
    - Scaled weights with emphasis on recent activations.
    """
    time_indices = torch.arange(len(weights), dtype=torch.float32)
    scaling_factors = recency_factor ** time_indices
    return weights * scaling_factors
