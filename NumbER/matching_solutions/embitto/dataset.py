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
        #self.numerical_data = [self.numerical_data[i] for i in indices]
        self.numerical_data = self.numerical_data.reindex(entity_ids_df.index) if self.numerical_data is not None else None
        self.textual_data = [self.textual_data[i] for i in indices] if self.textual_data is not None else None
        self.df = self.df.reindex(entity_ids_df.index)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.df.drop(columns=['entity_id', 'id'], inplace=True)
        self.numerical_data.drop(columns=['id'], inplace=True) if self.numerical_data is not None else None
        if self.textual_data is None:
            assert len(self.numerical_data) == len(self.entity_ids)
        elif self.numerical_data is None:
            assert len(self.textual_data) == len(self.entity_ids)
        else:
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
        if self.textual_pairs is None:
            print("Textual pairs is NONE. Please check...")
            assert len(self.numerical_pairs) == len(self.groundtruth)
        elif self.numerical_pairs is None or len(self.numerical_pairs) == 0:
            self.numerical_pairs = None
            assert len(self.textual_pairs) == len(self.groundtruth)
        else:
            assert len(self.numerical_pairs) == len(self.textual_pairs) == len(self.groundtruth)
        
    def __len__(self):
        return len(self.numerical_pairs) if self.textual_pairs is None else len(self.textual_pairs)
    
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
        if self.numerical_pairs is not None:
            try:
                numeric = self.tokenizer.encode(text=self.numerical_pairs[idx], max_length=256, truncation=True)
            except:
                numeric = self.numerical_pairs.loc[idx].array
        else:
            numeric = None
        #! CHANGE numeric (second argument) BACK to self.numerical_pairs.loc[idx].array
        return textual_pairs, numeric, self.groundtruth['prediction'][idx]

    #!CHange
    @staticmethod
    def pad(batch): #copied and adopted from ditto
        textual,numerical, y = zip(*batch)
        is_textual_none = all(v is None for v in textual)
        is_numerical_none = all(v is None for v in numerical)
        maxlen = max([len(x) for x in textual]) if not is_textual_none else 0
        textual = [xi + [0]*(maxlen - len(xi)) for xi in textual] if not is_textual_none else None
        textual = torch.LongTensor(textual) if not is_textual_none else None
        try:
            maxlen = max([len(x) for x in numerical]) if not is_numerical_none else 0
            numerical = torch.LongTensor([xi + [0]*(maxlen - len(xi)) for xi in numerical]) if not is_numerical_none else None
            return textual, numerical, torch.LongTensor(y)#!remove second long tensor
        except:
            return textual, numerical if not is_numerical_none else None, torch.LongTensor(y)
