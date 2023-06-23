from torch.utils.data import Dataset
import pandas as pd
from NumbER.matching_solutions.utils.transitive_closure import build_entity_ids
from transformers import AutoTokenizer
import numpy as np
import torch

class CompleteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.numerical_data = numerical_data
        self.textual_data = textual_data
        self.df = df
        self.groundtruth = groundtruth  
        entity_ids = build_entity_ids(groundtruth, df)
        entity_ids_df = pd.DataFrame(entity_ids, columns=['entity_id']).sort_values('entity_id')
        indices = np.argsort(entity_ids)
        self.entity_ids = [entity_ids[i] for i in indices]
        # print(self.numerical_data)
        # print(self.textual_data)
        self.numerical_data = [self.numerical_data[i] for i in indices]
        #self.numerical_data = self.numerical_data.reindex(entity_ids_df.index)
        self.textual_data = [self.textual_data[i] for i in indices]
        self.df = self.df.reindex(entity_ids_df.index)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.df.drop(columns=['entity_id'], inplace=True)
        assert len(self.numerical_data) == len(self.textual_data) == len(self.entity_ids)

    def __len__(self):
        return len(self.textual_data)

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.textual_data[idx], max_length=80, truncation=True)
        
        return x, self.numerical_data.loc[idx].array, self.entity_ids[idx]
    
    @staticmethod
    def pad(batch): #copied and adopted from ditto
        x12,wow, y = zip(*batch)
        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
        return torch.LongTensor(x12), wow, torch.LongTensor(y)
    
class PairBasedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.df = df
        self.numerical_pairs = numerical_data
        self.textual_pairs = textual_data
        self.groundtruth = groundtruth
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        assert len(self.numerical_pairs) == len(self.textual_pairs) == len(self.groundtruth)
        
    def __len__(self):
        return len(self.textual_pairs)
    
    def __getitem__(self, idx):
        #print(self.textual_pairs)
        
        x = self.tokenizer.encode(text=self.textual_pairs[idx][0],
                                  text_pair=self.textual_pairs[idx][1],
                                  max_length=256,
                                  truncation=True)
        return x, self.numerical_pairs.loc[idx].array, self.groundtruth['prediction'][idx]
    
    @staticmethod
    def pad(batch): #copied and adopted from ditto
        x12,wow, y = zip(*batch)
        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
        return torch.LongTensor(x12),wow, torch.LongTensor(y)
