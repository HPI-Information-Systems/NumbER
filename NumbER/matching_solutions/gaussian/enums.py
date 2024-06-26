from enum import Enum

class Phase(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3
    
class Stage(Enum):
    PRETRAIN = 1
    FINETUNING = 2