import torch
import gpytorch
import pytorch_lightning as pl
from torch import nn
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class CompositeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.nn = SimpleNN()
        self.likelihood1 = GaussianLikelihood()
        self.gp1 = GPModel(None, None, self.likelihood1)
        self.likelihood2 = GaussianLikelihood()
        self.gp2 = GPModel(None, None, self.likelihood2)

    def forward(self, x):
        nn_output = self.nn(x)
        gp1_output = self.gp1(nn_output).mean
        inputs_for_final_gp = torch.cat([nn_output, gp1_output.unsqueeze(-1)], dim=-1)
        final_output = self.gp2(inputs_for_final_gp).mean
        return final_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = torch.mean((preds - y) ** 2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer