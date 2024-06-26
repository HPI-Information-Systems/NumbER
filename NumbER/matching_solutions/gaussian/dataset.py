from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import torch
    
class PairBasedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, numerical_feature_vector, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.df = df
        self.textual_pairs = textual_data
        self.groundtruth = groundtruth
        self.numerical_feature_vector = numerical_feature_vector
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        assert len(self.textual_pairs) == len(self.groundtruth)
        
    def __len__(self):
        return len(self.textual_pairs)
    
    def __getitem__(self, idx):
        if self.textual_pairs is None:
            textual_pairs = None
        else:
            try:
                #print("TEXTUAL PAIRS", np.shape(self.textual_pairs[idx]))
                if np.shape(self.textual_pairs[idx])[0] == 2:
                    textual_pairs = self.tokenizer.encode(text=self.textual_pairs[idx][0],
                                  text_pair=self.textual_pairs[idx][1],
                                  max_length=256,
                                  truncation=True)
                else:
                    textual_pairs = self.tokenizer.encode(text=self.textual_pairs[idx],
                                  max_length=256,
                                  truncation=True)
            except Exception as e:
                #print("ERROR", e)
                textual_pairs = self.tokenizer.encode(text=self.textual_pairs[idx],
                                  max_length=256,
                                  truncation=True)
        return textual_pairs, self.numerical_feature_vector[idx], self.groundtruth['prediction'][idx]

    #!CHange
    @staticmethod
    def pad(batch): #copied and adopted from ditto
        textual, numerical_feature_vector, y = zip(*batch)
        is_textual_none = all(v is None for v in textual)
        maxlen = max([len(x) for x in textual])
        textual = [xi + [0]*(maxlen - len(xi)) for xi in textual] if not is_textual_none else None
        textual = torch.LongTensor(textual) if not is_textual_none else None
        print(numerical_feature_vector)
        return textual, torch.LongTensor(numerical_feature_vector), torch.LongTensor(y)
        try:
            maxlen = max([len(x) for x in numerical]) if not is_numerical_none else 0
            numerical = torch.LongTensor([xi + [0]*(maxlen - len(xi)) for xi in numerical]) if not is_numerical_none else None
            return textual, numerical, torch.LongTensor(y)#!remove second long tensor
        except:
            return textual, numerical if not is_numerical_none else None, torch.LongTensor(y)
