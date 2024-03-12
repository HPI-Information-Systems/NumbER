from NumbER.matching_solutions.combiner.dataset import PairBasedDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
#from NumbER.matching_solutions.embitto.enums import Stage

    
class CombinerDataModule(pl.LightningDataModule):
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, predict_data: pd.DataFrame, batch_size: int = 50):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.predict_data = predict_data
        self.batch_size = batch_size
        self.dataset_class = PairBasedDataset
        self.train_dataset = self.dataset_class(self.train_data['all_data'], self.train_data['statistical_output'], self.train_data['textual_data'], self.train_data['matches'])
        self.valid_dataset = self.dataset_class(self.valid_data['all_data'], self.valid_data['statistical_output'], self.valid_data['textual_data'], self.valid_data['matches'])
        self.test_dataset = self.dataset_class(self.test_data['all_data'], self.test_data['statistical_output'], self.test_data['textual_data'], self.test_data['matches'])
        self.predict_dataset = self.dataset_class(self.predict_data['all_data'], self.predict_data['statistical_output'], self.predict_data['textual_data'], self.predict_data['matches'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.train_dataset.pad, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size*4, collate_fn=self.train_dataset.pad, shuffle=True)#, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size*4, collate_fn=self.train_dataset.pad, shuffle=True)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size*4, collate_fn=self.train_dataset.pad)