class Config:
    num_layers = 3
    num_neurons = 10
    num_states = 5
    decay_rate = 0.1             # Decay rate for neuron states
    recency_factor = 0.9         # Recency factor for temporal proximity scaling
    buffer_size = 5              # Size of the short-term buffer
    importance_threshold = 0.1   # Threshold for entropy pruning
    compressed_dim = 3           # Compressed dimension for auto-encoder
    sparsity_threshold = 0.05    # Threshold for dynamic sparsity control
    low_power_k = 3              # Number of states to retain under low-power conditions
    init_method = "xavier"       # Initialization method
    base_lr = 0.001              # Base learning rate
    max_grad_norm = 1.0          # Gradient clipping max norm
    epochs = 20                  # Number of training epochs
    priority_threshold = 0.5     # Minimum importance level for async neuron updates
    activation_probability = 0.2 # Probability of activating a neuron path in probabilistic activation
    attention_threshold = 0.6    # Threshold for attention gating
    sparsity_mask = torch.ones(num_neurons, num_states)  # Mask for sparse gradient aggregation




