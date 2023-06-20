from torch import nn
from transformers import RobertaModel, AutoTokenizer
from NumbER.matching_solutions.embitto.embitto import Stage
from NumbER.matching_solutions.embitto.formatters import ditto_formatter, pair_based_ditto_formatter
import torch
import numpy as np


class BaseRoberta(nn.Module):
    def __init__(self, max_length: int = 40, embedding_size: int = 256, stage: Stage = Stage.PRETRAIN):
        super(BaseRoberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.embeddings = nn.Linear(self.roberta.config.hidden_size, embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.max_length = max_length
        self.stage = stage
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_sequence):
        input_ids = []
        if self.stage == Stage.PRETRAIN:
            input_ids = [torch.tensor(self.tokenizer.encode(input, max_length=self.max_length, truncation=True)) for input in input_sequence]
        else:
            #input_sequence = np.reshape(input_sequence, (shape[1], shape[0]))
            input_ids = [self.tokenizer.encode(input[0], text_pair=input[1], max_length=self.max_length, truncation=True) for input in zip(input_sequence[0], input_sequence[1])]
            #input_ids = torch.tensor(input_ids).to('cuda')
        input_ids = self.pad_tensor_list(input_ids, self.max_length).long()
        #input_ids = torch.stack(input_ids).to('cuda')
        outputs = self.roberta(
            input_ids
        )
        return outputs[0][:, 0, :]
        embeddings = self.embeddings(outputs[0][:, 0, :])
        return embeddings
        #return pooled_output
        
    def pad_tensors(self, tensors, max_length):
        padded_tensors = []
        for tensor in tensors:
            tensor = tensor.to('cuda')
            padding_length = max_length - tensor.shape[0]
            if padding_length > 0:
                padded_tensor = torch.zeros(max_length, device='cuda', dtype=torch.long)
                padded_tensor[:tensor.shape[0]] = tensor
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        return torch.stack(padded_tensors).to('cuda')
    def pad_tensor_list(self, tensors, max_length):
        padded_tensors = []
        for tensor in tensors:
            padding_length = max_length - np.shape(tensor)[0]#.shape[0]
            if padding_length > 0:
                padded_tensor = np.zeros(max_length)
                padded_tensor[:np.shape(tensor)[0]] = tensor
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        return torch.tensor(padded_tensors).to('cuda')
        #
    def get_formatter(self):
        if self.stage == Stage.PRETRAIN:
            return ditto_formatter
        elif self.stage == Stage.FINETUNING:
            return pair_based_ditto_formatter