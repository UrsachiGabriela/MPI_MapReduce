from dataclasses import dataclass
from enum import Enum

class Phase(Enum):
    PREPROCESSING = 0
    MAP = 1
    INTERMEDIATE = 2
    REDUCE = 3
    FINISH = 4


class Tag(Enum):
    FREE_WORKER = 0

    MAP_OPERATION = 2
    REDUCE_OPERATION = 3
    MAP_FINISH = 4
    REDUCE_FINISH = 5
    FINISH = 6


@dataclass
class MapMessage:
    files_to_process: list
    output_path: str


@dataclass
class ReduceMessage:
    input_path:str
    files_to_process: list
    output_path: str
    max_dir_index:int
