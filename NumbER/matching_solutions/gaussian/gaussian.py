import torch
import pytorch_lightning as pl
from transformers import AdamW
from torch import nn
import torch
import gpytorch
import pytorch_lightning as pl
from torch import nn
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
        
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class Gaussian(pl.LightningModule):
    def __init__(self, textual_component, learning_rate: int = 0.00003):
        super(Gaussian, self).__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.textual_component = textual_component
        output_size = textual_component.get_outputshape()
        self.finetuning_step = nn.Linear(output_size, 1)#!output_size
        self.likelihood1 = GaussianLikelihood()
        self.gp1 = GPModel(None, None, self.likelihood1)
        self.likelihood2 = GaussianLikelihood()
        self.gp2 = GPModel(None, None, self.likelihood2)
        #self.intermediate_step = nn.Linear(2+2, 128)
        #self.output_step = nn.Linear(128, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, input_textual, features_numerical):
        textual_output = self.finetuning_step(self.textual_component(input_textual))
        stat_output = self.gp1(features_numerical).mean
        inputs_for_final_gp = torch.stack((textual_output.squeeze(), stat_output), dim=1)
        res = self.gp2(inputs_for_final_gp).mean
        #print("RES", res)
        return res

        #return self.output_step(self.intermediate_step(combined_input))
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
        print(predictions)
        print(labels.type(torch.FloatTensor).to("cuda"))
        loss = self.criterion(input=predictions,target=labels.type(torch.FloatTensor).to("cuda"))
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        textual_data, statistical_output, labels = batch
        result = self(textual_data, statistical_output)
        return result
    
    def test_step(self, batch, batch_idx):
        pass