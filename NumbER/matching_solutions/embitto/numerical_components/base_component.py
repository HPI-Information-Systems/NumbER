from abc import ABC, abstractmethod
from NumbER.matching_solutions.embitto.enums import Stage

class BaseNumericComponent(ABC):
    def __init__(self, train_data, val_data, test_data, dimensions, should_pretrain):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.should_pretrain = should_pretrain
        
    @abstractmethod
    def __call__(self, numbers, phase):
        raise NotImplementedError("__call__() is not implemented.")
    
    def get_outputshape(self):
        return self.output_shape
    
    def set_outputshape(self, output_shape):
        self.output_shape = output_shape
    
    @abstractmethod
    def get_formatter(stage: Stage):
        raise NotImplementedError("get_formatter() is not implemented.")