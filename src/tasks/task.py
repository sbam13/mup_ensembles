from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Mapping, NewType, Set
from enum import Enum

from jax.random import PRNGKey
from jaxlib.xla_extension import DeviceArray

TaskId = NewType('TaskId', int) # task 


class Status(Enum):
    WAITING = 0
    SUBMITTED = 1
    DONE = 2


@dataclass
class Task:
    model: str
    dataset: str
    
    model_params: Mapping
    training_params: Mapping

    type_: int
    seed: PRNGKey

    apply_callback: Callable
    # save_callback: Callable[[str, dict], None] # path, result -> None

    repeat: int = 1 
    parallelize: bool = False
    dependencies: Set = field(default_factory=set)
    
    _status: Status = field(default=Status.WAITING, init=False)
    
    _id: TaskId = field(init=False) 
    _count: ClassVar[int] = 0

    def __post_init__(self):
        self._id = TaskId(Task._count)
        Task._count += 1


@dataclass
class Task_ConfigSubset:
    model: str
    dataset: str
    model_params: Mapping
    training_params: Mapping

    
