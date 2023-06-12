from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
from enum import Enum
from torch.nn import functional as F
from torch import nn
from NumbER.matching_solutions.embitto.utils.contrastive_loss import contrastive_loss
import numpy as np
class Stage(Enum):
    PRETRAIN = 1
    FINETUNING = 2

class Embitto(pl.LightningModule):
    def __init__(self, stage: Stage, numerical_component, textual_component, fusion_component=None, finetuning_step=None):
        super(Embitto, self).__init__()
        self.nn_layers = nn.ModuleList()
        #self.numerical_component = numerical_component
        print(textual_component)
        self.textual_component = textual_component(256, stage)
        #self.textual_component = RobertaClassifier(256)
        #print("HDKWHDW DWH", textual_component.parameters())
        #self.fusion_component = fusion_component
        #self.finetuning_step = finetuning_step
        self.stage = stage
        self.criterion = contrastive_loss if self.stage == Stage.PRETRAIN else F.cross_entropy

    def forward(self, input):
        #input_numerical = input[0]
        input_textual = input[1]
        #print("input", input.shape)
        #output_numerical = self.numerical_component(input_numerical)
        output_textual = self.textual_component(input)
        return output_textual
        #embeddings = self.fusion_component(output_numerical, output_textual)
        #self.finetuning_step(embeddings)

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]#eigtl 2
        predictions = self(data)
        loss = self.criterion(predictions, labels)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]#eigtl 2
        predictions = self(data)
        print("DOING THE VALIDATIOn")
        loss = self.criterion(predictions, labels)
        print("Validarion loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        pass