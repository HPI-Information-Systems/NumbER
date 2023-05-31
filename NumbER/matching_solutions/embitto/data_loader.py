from NumbER.matching_solutions.embitto.dataset import CompleteDataset, PairBasedDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from enum import Enum
import pandas as pd
from NumbER.matching_solutions.embitto.embitto import Stage

    
class EmbittoDataModule(pl.LightningDataModule):
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, stage: Stage, batch_size: int = 32):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.dataset_class = CompleteDataset if stage == Stage.PRETRAIN else PairBasedDataset

    def setup(self):
        self.train_dataset = self.dataset_class(self.train_data['numerical_data'], self.train_data['textual_data'], self.train_data['matches'])
        self.valid_dataset = self.dataset_class(self.valid_data['numerical_data'], self.valid_data['textual_data'], self.valid_data['matches'])
        self.test_dataset = self.dataset_class(self.test_data['numerical_data'], self.test_data['textual_data'], self.test_data['matches'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)