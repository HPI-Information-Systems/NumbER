from torch.utils.data import Dataset
import pandas as pd
from NumbER.matching_solutions.utils.transitive_closure import calculate_clusters

class CompleteDataset(Dataset):
    def __init__(self, numerical_data: pd.DataFrame, textual_data: pd.DataFrame, groundtruth: pd.DataFrame):
        self.numerical_data = numerical_data
        self.textual_data = textual_data
        self.groundtruth = groundtruth
        entity_ids = self.build_entity_ids()
        self.entity_ids = entity_ids
        assert len(self.numerical_data) == len(self.textual_data) == len(self.entity_ids)
        
    def build_entity_ids(self):
        clusters = calculate_clusters(self.groundtruth)
        entity_id = 0
        for cluster in clusters:
            for id in cluster:
                self.data.loc[self.data['id'] == id, 'entity_id'] = entity_id
            entity_id += 1
        entity_ids = self.data['entity_id'].values
        self.data.drop(columns=['entity_id'], inplace=True)
        return entity_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.textual_pairs[idx], self.numerical_pairs[idx], self.entity_ids[idx]
    
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