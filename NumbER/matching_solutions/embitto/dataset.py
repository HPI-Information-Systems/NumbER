from torch.utils.data import Dataset
import pandas as pd
from NumbER.matching_solutions.utils.transitive_closure import build_entity_ids
from transformers import AutoTokenizer
import numpy as np

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
        self.df.drop(columns=['entity_id'], inplace=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        assert len(self.numerical_data) == len(self.textual_data) == len(self.entity_ids)

    def __len__(self):
        return len(self.textual_data)

    def __getitem__(self, idx):
        return self.textual_data[idx], self.numerical_data[idx], self.entity_ids[idx]
    
class PairBasedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.df = df
        print("GSGS", self.df)
        self.numerical_pairs = numerical_data
        self.textual_pairs = textual_data
        self.groundtruth = groundtruth
        assert len(self.numerical_pairs) == len(self.textual_pairs) == len(self.groundtruth)
        
    def __len__(self):
        return len(self.textual_pairs)
    
    def __getitem__(self, idx):
        return self.textual_pairs[idx], self.numerical_pairs[idx], self.groundtruth['prediction'][idx]