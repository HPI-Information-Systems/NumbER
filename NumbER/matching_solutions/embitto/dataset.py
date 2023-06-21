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
        self.df.drop(columns=['entity_id'], inplace=True)
        
        assert len(self.numerical_data) == len(self.textual_data) == len(self.entity_ids)

    def __len__(self):
        return len(self.textual_data)

    def __getitem__(self, idx):
        return self.textual_data[idx], self.numerical_data[idx], self.entity_ids[idx]
    
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
        return x, self.numerical_pairs[idx], self.groundtruth['prediction'][idx]
    
    @staticmethod
    def pad(batch): #copied from ditto
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        # if len(batch[0]) == 3:
        #     x1, x2, y = zip(*batch)

        #     maxlen = max([len(x) for x in x1+x2])
        #     x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
        #     x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
        #     return torch.LongTensor(x1), \
        #            torch.LongTensor(x2), \
        #            torch.LongTensor(y)
        # else:
        x12,wow, y = zip(*batch)
        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
        return torch.LongTensor(x12),wow, torch.LongTensor(y)
