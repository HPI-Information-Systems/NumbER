from torch import nn
from transformers import RobertaModel, AutoTokenizer
from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.formatters import ditto_formatter, pair_based_ditto_formatter, complete_prompt_formatter
import torch
import numpy as np


class BaseRoberta(nn.Module):
    def __init__(self, max_length: int = 40, embedding_size: int = 256, stage: Stage = Stage.PRETRAIN):
        super(BaseRoberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.to("cuda")
        self.embeddings = nn.Linear(self.roberta.config.hidden_size, embedding_size, device="cuda")
        self.dropout = nn.Dropout(0.1)
        self.max_length = max_length
        self.stage = stage
        self.embedding_size = embedding_size
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
    
    def get_outputshape(self):
        return self.embedding_size

    def forward(self, input_sequence):
        #if self.stage == Stage.PRETRAIN:
            #input_sequence = [torch.tensor(self.tokenizer.encode(input, max_length=self.max_length, truncation=True)) for input in input_sequence]
            #input_sequence = self.pad_tensors(input_sequence, self.max_length) #todo check if abstract
        outputs = self.roberta(
            input_sequence
        )
        #print("shape", np.shape(outputs[0][:, 0, :]))
        #return outputs[0][:, 0, :]
        embeddings = self.embeddings(outputs[0][:, 0, :])
        return embeddings
        #return pooled_output

    def get_formatter(self):
        if self.stage == Stage.PRETRAIN:
            return ditto_formatter
        elif self.stage == Stage.FINETUNING:
            return pair_based_ditto_formatter #! pair_based_ditto_formatter