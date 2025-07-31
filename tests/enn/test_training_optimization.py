import torch
from enn.training_optimization import MetaLearningRateScheduler, sparse_gradient_aggregation, gradient_clipping

def test_meta_learning_rate_scheduler():
    optimizer = torch.optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.01)
    lr_before = optimizer.param_groups[0]["lr"]
    scheduler = MetaLearningRateScheduler(base_lr=0.01)
    neuron_stability = torch.rand(10)
    entanglement_strength = torch.rand(10)
    scheduler.adjust_learning_rate(optimizer, neuron_stability, entanglement_strength)
    lr_after = optimizer.param_groups[0]["lr"]
    
    # If lr_after is a tensor, check if any element differs from lr_before
    if isinstance(lr_after, torch.Tensor) and lr_after.numel() > 1:
        assert not torch.allclose(lr_after, torch.full_like(lr_after, lr_before))
    else:
        # Single value comparison
        lr_before = lr_before.item() if isinstance(lr_before, torch.Tensor) else lr_before
        lr_after = lr_after.item() if isinstance(lr_after, torch.Tensor) else lr_after
        assert lr_before != lr_after

def test_sparse_gradient_aggregation():
    model = torch.nn.Linear(10, 5)
    data = torch.rand(10)
    output = model(data)
    output.sum().backward()  # Initialize gradients
    sparsity_mask = torch.rand(10) > 0.5
    sparse_gradient_aggregation(model, sparsity_mask)
    assert all(param.grad is not None for param in model.parameters())


def test_gradient_clipping():
    model = torch.nn.Linear(10, 5)
    output = model(torch.rand(1, 10)).sum()  # Make output scalar
    output.backward()
    gradient_clipping(model, max_norm=0.1)
    for param in model.parameters():
        if param.grad is not None:
            assert torch.norm(param.grad) <= 0.1
