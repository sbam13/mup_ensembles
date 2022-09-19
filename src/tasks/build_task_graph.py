from .task import Task, TaskId
from graphlib import TopologicalSorter, CycleError

def order_tasks(tasks: dict[TaskId, Task], num_devices: int):
    id_graph = {tid: task.dependencies for tid, task in tasks}
    
    ts = TopologicalSorter(id_graph)
    try:
        return list(ts.static_order())
    except CycleError:
        raise ValueError('A dependency cycle exists between tasks.')

        