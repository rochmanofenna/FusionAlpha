import torch
from enn.layers import process_entangled_neuron_layer, mini_batch_processing

def test_process_entangled_neuron_layer():
    data_batch = torch.rand(4, 10)
    weights = torch.rand(10, 5)
    neuron_layer = process_entangled_neuron_layer(data_batch, weights, num_neurons=10, num_states=5)
    assert neuron_layer.shape == (4, 10, 5)

def test_mini_batch_processing():
    data_stream = [torch.rand(10) for _ in range(10)]
    batches = list(mini_batch_processing(data_stream, batch_size=4))
    assert len(batches) == 3  # 10 data items with batch size of 4 should yield 3 batches
    assert batches[0].shape == (4, 10)
