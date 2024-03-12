import torch
import pytorch_lightning as pl
from transformers import AdamW
from torch import nn

class Combiner(pl.LightningModule):
    def __init__(self, textual_component, learning_rate: int = 0.00003):
        super(Combiner, self).__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.textual_component = textual_component
        output_size = textual_component.get_outputshape()
        self.finetuning_step = nn.Linear(output_size, 2)#!output_size
        self.intermediate_step = nn.Linear(2+2, 128)
        self.output_step = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, input_textual, statistical_output):
        output = self.finetuning_step(self.textual_component(input_textual))
        #output = self.textual_component(input_textual)
        combined_input = torch.cat((output, statistical_output), dim=1)
        return self.output_step(self.intermediate_step(combined_input))
        #return self.finetuning_step(output)
        
    def training_step(self, batch, batch_idx):
        textual_data, statistical_output, labels = batch
        predictions = self(textual_data, statistical_output)
        loss = self.criterion(input=predictions,target=labels)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate,weight_decay=1e-5)
    
    def validation_step(self, batch, batch_idx):
        textual_data, statistical_output, labels = batch
        predictions = self(textual_data, statistical_output)
        loss = self.criterion(input=predictions,target=labels)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        textual_data, statistical_output, labels = batch
        result = self(textual_data, statistical_output)
        return result
    
    def test_step(self, batch, batch_idx):
        pass