import torch
import pytest
from enn.event_processing import async_neuron_update, handle_event
from enn.scheduler import PriorityTaskScheduler

@pytest.mark.asyncio
async def test_async_neuron_update():
    neuron_state = torch.zeros(10)
    data_input = torch.rand(10)
    updated_state = await async_neuron_update(neuron_state, data_input, priority_threshold=0.5)
    assert updated_state.shape == neuron_state.shape

@pytest.mark.asyncio
async def test_handle_event():
    neuron_state = torch.zeros(10)
    data_input = torch.rand(10)
    scheduler = PriorityTaskScheduler()
    await handle_event(neuron_state, data_input, scheduler, priority=1)
    assert len(scheduler.task_queue) == 0  # Verify that tasks are processed
