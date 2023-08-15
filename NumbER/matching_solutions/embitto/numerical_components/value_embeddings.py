import numpy as np
import torch
from torch import nn
from transformers import RobertaModel, AutoTokenizer
from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.formatters import ditto_formatter, pair_based_ditto_formatter
import torch
import numpy as np
import sys


from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.numerical_components.base_component import BaseNumericComponent
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_numeric_formatter

class ValueTransformerEmbeddings(BaseNumericComponent):
    def __init__(self, train_data, val_data, test_data, dimensions, should_pretrain):
        super().__init__(train_data, val_data, test_data, dimensions, should_pretrain)
        #self.set_outputshape(2*(len(self.train_data.columns) -1))
        self.set_outputshape(dimensions)
        multiplicator = 1 if should_pretrain else 2
        self.base = Transformer(multiplicator*(len(self.train_data.columns) -1), dimensions)
        print("base", self.base)

    def __call__(self, batch, phase):
        res = []
        for numbers in batch:
            embed = []
            for number in numbers:
                if isinstance(number, tuple):
                    # print("num", number[0])
                    # print("daw", np.isnan(number[0]))
                    embed.append(number[0] if not np.isnan(number[0]) else -1)#!-1 nicht gut, da das auch einfach in den daten vorkommen 
                    embed.append(number[1] if not np.isnan(number[1]) else -1)
                else:
                    embed.append(number if not np.isnan(number) else -1)
            res.append(embed)
        #print("res",res)
        return self.base(torch.tensor(res, dtype=torch.float32).to("cuda"))
    
    def get_outputshape(self):
        return self.output_shape
    
    def set_outputshape(self, output_shape):
        self.output_shape = output_shape
        
    @staticmethod
    def get_formatter(stage: Stage):
        if stage == stage.PRETRAIN:
            return dummy_formatter
        elif stage == stage.FINETUNING:
            return pair_based_numeric_formatter

class ValueBaseEmbeddings(BaseNumericComponent):
    def __init__(self, train_data, val_data, test_data, dimensions, should_pretrain=True):
        super().__init__(train_data, val_data, test_data, dimensions, should_pretrain)
        #self.set_outputshape(2*(len(self.train_data.columns) -1))
        self.set_outputshape(dimensions)
        multiplicator = 1 if should_pretrain else 2
        self.base = Base(multiplicator * (len(self.train_data.columns) -1), dimensions)

    def __call__(self, batch, phase):
        res = []
        # if self.stage == Stage.PRETRAIN:
        #     return
        for numbers in batch:
            embed = []
            for number in numbers:
                if isinstance(number, tuple):
                    # print("num", number[0])
                    # print("daw", np.isnan(number[0]))
                    embed.append(number[0] if not np.isnan(number[0]) else -1)#!-1 nicht gut, da das auch einfach in den daten vorkommen 
                    embed.append(number[1] if not np.isnan(number[1]) else -1)
                else:
                    embed.append(number if not np.isnan(number) else -1)
            res.append(embed)
        #print("res",res)
        return self.base(torch.tensor(res, dtype=torch.float32).to("cuda"))
    
    def get_outputshape(self):
        return self.output_shape
    
    def set_outputshape(self, output_shape):
        self.output_shape = output_shape
        
    @staticmethod
    def get_formatter(stage: Stage):
        if stage == stage.PRETRAIN:
            return dummy_formatter
        elif stage == stage.FINETUNING:
            return pair_based_numeric_formatter
        
class ValueValueEmbeddings(BaseNumericComponent):
    def __init__(self, train_data, val_data, test_data, dimensions, should_pretrain=True):
        super().__init__(train_data, val_data, test_data, dimensions, should_pretrain)
        self.set_outputshape(2*(len(self.train_data.columns) -1))

    def __call__(self, batch, phase):
        res = []
        # if self.stage == Stage.PRETRAIN:
        #     return
        for numbers in batch:
            embed = []
            for number in numbers:
                if isinstance(number, tuple):
                    # print("num", number[0])
                    # print("daw", np.isnan(number[0]))
                    embed.append(number[0] if not np.isnan(number[0]) else -1)#!-1 nicht gut, da das auch einfach in den daten vorkommen 
                    embed.append(number[1] if not np.isnan(number[1]) else -1)
                else:
                    if number == 0:
                        embed.append(0 + np.finfo(np.float32).eps)
                    embed.append(number if not np.isnan(number) else -1)
            res.append(embed)
        #print("res",res)
        return torch.log(torch.tensor(res, dtype=torch.float32).to("cuda"))
    
    def get_outputshape(self):
        return self.output_shape
    
    def set_outputshape(self, output_shape):
        self.output_shape = output_shape
        
    @staticmethod
    def get_formatter(stage: Stage):
        if stage == stage.PRETRAIN:
            return dummy_formatter
        elif stage == stage.FINETUNING:
            return pair_based_numeric_formatter

class Base(nn.Module):
    def __init__(self, input_dimension, dimensions):
        super(Base, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dimension, device="cuda")
        self.embeddings = nn.Linear(input_dimension, dimensions, device="cuda")

    def forward(self, input_sequence):
        norm = self.layer_norm(input_sequence)
        return self.embeddings(norm)
        return self.embeddings(self.layer_norm(input_sequence))

class Transformer(nn.Module):
    def __init__(self, input_dimension, dimensions):
        super(Transformer, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dimension, device="cuda")
        self.linear = nn.Linear(input_dimension, 128, device="cuda")
        #self.embedding = nn.Embedding(input_dimension, dimensions, device="cuda")
        #self.transformer = nn.Transformer(input_dimension, dimensions, device="cuda")
        #self.transformer = nn.Transformer(dimensions, num_encoder_layers=6, device="cuda")
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, device="cuda")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=dimensions, nhead=8, device="cuda")
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    def forward(self, input_sequence):
        #return self.layer_norm(input_sequence)
        #print(x)
        x = self.layer_norm(input_sequence)
        x = self.linear(x)
        #print(len(x))
        #x = self.embedding(x.long())
        #tgt = x.clone()
        x = self.transformer_encoder(x)
        return x
        return self.transformer(self.embedding())
