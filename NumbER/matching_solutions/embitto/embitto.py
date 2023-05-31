from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
from torch.nn import functional as F

class Stage(Enum):
    PRETRAIN = 1
    FINETUNING = 2

class Embitto(pl.LightningModule):
    def __init__(self, stage: Stage, numerical_component, textual_component, fusion_component, finetuning_step):
        self.numerical_component = numerical_component
        self.textual_component = textual_component
        self.fusion_component = fusion_component
        self.finetuning_step = finetuning_step
        self.stage = stage
        self.criterion = "triplet_loss" if self.stage == Stage.PRETRAIN else F.cross_entropy
    
    def forward(self, input):
        input_numerical = input[0]
        input_textual = input[1]
        output_numerical = self.numerical_component(input_numerical)
        output_textual = self.textual_component(input_textual)
        embeddings = self.fusion_component(output_numerical, output_textual)
        self.finetuning_step(embeddings)

    def training_step(self, batch, batch_idx):
            
        predictions = self.forward(num_data, text_data)
        loss = self.criterion(predictions, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())