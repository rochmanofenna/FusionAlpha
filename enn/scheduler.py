import heapq
import asyncio

class PriorityTaskScheduler:
    def __init__(self):
        self.task_queue = []

    def add_task(self, task, priority):
        # Add tasks with priority to the queue (lower values indicate higher priority)
        heapq.heappush(self.task_queue, (priority, task))

    async def process_tasks(self):
        while self.task_queue:
            priority, task = heapq.heappop(self.task_queue)
            await task
