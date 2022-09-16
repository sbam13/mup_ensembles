from ast import Call
from dataclasses import dataclass
from typing import Callable, Dict, List, NewType, Set
from enum import Enum

TaskId = NewType('TaskId', int) # task 

class TaskType(Enum):
    PROCESS_DATA = 0
    TRAIN_NN = 1
    TRAIN_NTK = 2
    COMPUTE_STATISTICS = 3

class Status(Enum):
    WAITING = 0
    SUBMITTED = 1
    DONE = 2

@dataclass
class Task:
    dependencies: Set = None
    hyperparams: Dict = None # alpha, k, N, P
    repeat: int = 1
    type: TaskType = None
    status: Status = Status.WAITING
    single_gpu: bool = True
    _id: TaskId = None

    apply_callback: Callable = None
    save_callback: Callable = None


    
