from typing import Set
from .task import Task, TaskId
from graphlib import TopologicalSorter, CycleError

def order_tasks(dependency_graph: dict[TaskId, Set[TaskId]]):    
    ts = TopologicalSorter(dependency_graph)
    try:
        return list(ts.static_order())
    except CycleError:
        raise ValueError('A dependency cycle exists between tasks.')

        