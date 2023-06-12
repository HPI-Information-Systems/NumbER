from torch.utils.data import Dataset
import pandas as pd
from NumbER.matching_solutions.utils.transitive_closure import calculate_clusters
from transformers import RobertaModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class CompleteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.numerical_data = numerical_data
        self.textual_data = textual_data
        self.df = df
        self.groundtruth = groundtruth
        entity_ids = self.build_entity_ids()
        self.entity_ids = entity_ids
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        assert len(self.numerical_data) == len(self.textual_data) == len(self.entity_ids)
        
    def build_entity_ids(self):
        clusters = calculate_clusters(self.groundtruth)
        entity_id = 0
        for cluster in clusters:
            for id in cluster:
                self.df.loc[self.df['id'] == id, 'entity_id'] = entity_id
            entity_id += 1
        entity_ids = self.df['entity_id'].values
        self.df.drop(columns=['entity_id'], inplace=True)
        return entity_ids

    def __len__(self):
        return len(self.textual_data)

    def __getitem__(self, idx):
        #print("to encode: ", self.textual_data[idx])
        
        #print("Textual data for idx", textual_data, idx)
        #print(len(textual_data), self.entity_ids[idx])
        return self.textual_data[idx], self.entity_ids[idx]#self.numerical_data[idx], self.entity_ids[idx]
    
class PairBasedDataset(Dataset):
    def __init__(self, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.numerical_pairs = numerical_data
        self.textual_pairs = textual_data
        self.groundtruth = groundtruth
        assert len(self.numerical_data) == len(self.textual_data) == len(self.groundtruth)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.textual_pairs[idx], self.numerical_pairs[idx], self.groundtruth[idx]