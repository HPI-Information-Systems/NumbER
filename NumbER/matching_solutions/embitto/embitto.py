from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
from enum import Enum
from torch.nn import functional as F
from transformers import AdamW
from torch import nn
from NumbER.matching_solutions.embitto.utils.contrastive_loss import contrastive_loss, ContrastiveLoss, calculatue_tuples
from NumbER.matching_solutions.embitto.enums import Stage, Phase
import numpy as np

class Embitto(pl.LightningModule):
    def __init__(self, stage: Stage, numerical_component, textual_component, fusion_component=None, finetuning_step=None, learning_rate: int = 0.00003):
        super(Embitto, self).__init__()
        #self.save_hyperparameters()
        self.stage = stage
        self.learning_rate = learning_rate
        self.textual_component = textual_component
        self.numerical_component = numerical_component
        self.fusion_component = fusion_component
        self.finetuning_step = nn.Linear(480, 2) #!input war 256
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5) if self.stage == Stage.PRETRAIN else nn.CrossEntropyLoss()

    def forward(self, input_textual, input_numerical, phase: Phase):
        #output_numerical = self.numerical_component(input_numerical)
        #print(input_textual)
        output_textual = self.textual_component(input_textual) if self.textual_component is not None else None
        if self.numerical_component is not None and self.textual_component is not None:
            output_numerical = self.numerical_component(input_numerical, phase)
            output = self.fusion_component(output_textual, output_numerical)
        elif self.numerical_component is None and self.textual_component is not None:
            output = output_textual
        elif self.numerical_component is not None and self.textual_component is None:
            output = self.numerical_component(input_numerical, phase).type(torch.float32)
        if self.stage == Stage.FINETUNING:
            #output = self.dropout(output)
            output = self.finetuning_step(output)
        #embeddings = self.fusion_component(output_numerical, output_textual)
        #self.finetuning_step(embeddings)
        return output
    
    def set_stage(self, stage: Stage):
        self.stage = stage
        if self.textual_component is not None:
            self.textual_component.stage = stage
            self.textual_component.max_length = 60 if stage == Stage.PRETRAIN else 256
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5) if self.stage == Stage.PRETRAIN else nn.CrossEntropyLoss()
        if self.numerical_component is not None:
            self.numerical_component.stage = stage
        
    def training_step(self, batch, batch_idx):
        textual_data, numerical_data, labels = batch
        predictions = self(textual_data, numerical_data, Phase.TRAIN)
        loss = self.calculate_embedding_loss(predictions, labels, phase="train") if self.stage == Stage.PRETRAIN else self.criterion(input=predictions,target=labels)
        self.log("train_loss", loss)
        print("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00003) if self.stage == Stage.PRETRAIN else AdamW(self.parameters(), lr=3e-5,weight_decay=1e-5)
    
    def validation_step(self, batch, batch_idx):
        textual_data, numerical_data, labels = batch
        predictions = self(textual_data, numerical_data, Phase.VALID)
        loss = self.calculate_embedding_loss(predictions, labels, phase="val") if self.stage == Stage.PRETRAIN else self.criterion(input=predictions,target=labels)
        print("Loss", loss)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        print("predict ste", batch)
        textual_data, numerical_data, labels = batch
        result = self(textual_data, numerical_data, Phase.TEST)
        return result
    
    def calculate_embedding_loss(self, predictions, labels, phase="train"):
        positive_firsts, positive_seconds, negative_firsts, negative_seconds = calculatue_tuples(predictions, labels)
        positive_firsts = torch.stack(positive_firsts).to('cuda')
        positive_seconds = torch.stack(positive_seconds).to('cuda')
        negative_firsts = torch.stack(negative_firsts).to('cuda')
        negative_seconds = torch.stack(negative_seconds).to('cuda')
        sum_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).to('cuda')
        pos_sim = []
        pos_loss = []
        neg_loss = []
        neg_sim = []
        sum_pos_loss = 0
        sum_neg_loss = 0
        for pos1, pos2 in zip(positive_firsts, positive_seconds):
            loss = self.criterion(pos1,pos2, target=torch.tensor(1))
            sum_loss += loss
            sum_pos_loss += loss
            pos_loss.append(loss)
            pos_sim.append(torch.cosine_similarity(pos1, pos2, dim=0))
        for neg1, neg2 in zip(negative_firsts, negative_seconds):
            loss = self.criterion(neg1, neg2, torch.tensor(-1))
            sum_loss += loss
            neg_loss.append(loss)
            sum_neg_loss += loss
            neg_sim.append(torch.cosine_similarity(neg1, neg2,dim=0))
        if phase == "val":
            print("Positive lossmean", torch.mean(torch.tensor(pos_loss)))
            print("Negative lossmean", torch.mean(torch.tensor(neg_loss)))
            print("Positive loss sum", torch.sum(torch.tensor(pos_loss)))
            print("Negative loss sum", torch.sum(torch.tensor(neg_loss)))
            print("Positive similarity", torch.mean(torch.tensor(pos_sim)))
            print("Negative similarity", torch.mean(torch.tensor(neg_sim)))
        sum_loss = sum_loss / (len(positive_firsts) + len(negative_firsts))
        sum_loss = (sum_pos_loss / len(positive_firsts) + sum_neg_loss / len(negative_firsts)) / 2
        print("Validarion loss", sum_loss) if phase == "val" else print("Training loss", sum_loss)
        return sum_loss
    
    def test_step(self, batch, batch_idx):
        pass