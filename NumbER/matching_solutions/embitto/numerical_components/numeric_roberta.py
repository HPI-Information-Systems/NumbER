from torch import nn
from transformers import RobertaModel, AutoTokenizer
from NumbER.matching_solutions.embitto.enums import Phase, Stage
from NumbER.matching_solutions.embitto.formatters import ditto_formatter, pair_based_ditto_formatter
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, numeric_prompt_formatter, complete_prompt_formatter, pair_based_ditto_formatter
from NumbER.matching_solutions.embitto.numerical_components.base_component import BaseNumericComponent
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
import torch
import numpy as np

class NumericRoberta(BaseNumericComponent):
    def __init__(self, train_data, valid_data, test_data, dimensions):
        super().__init__(train_data, valid_data, test_data, dimensions)
        self.roberta = BaseRoberta(256, 256, Stage.FINETUNING)
        self.set_outputshape(256)#-1 because we remove id afterwards
    
    def __call__(self, numbers, phase: Phase):
        return self.roberta(numbers)
    
    @staticmethod
    def get_formatter(stage: Stage):
        if stage == stage.PRETRAIN:
            return dummy_formatter
        elif stage == stage.FINETUNING:
            return pair_based_ditto_formatter
            return numeric_prompt_formatter