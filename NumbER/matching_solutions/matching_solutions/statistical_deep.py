import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from sklearn.metrics import f1_score, roc_curve
import xgboost as xgb
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from NumbER.matching_solutions.matching_solutions.embitto import EmbittoMatchingSolution
from NumbER.matching_solutions.matching_solutions.lightgbm import LightGBMMatchingSolution
from NumbER.matching_solutions.matching_solutions.xgboost import XGBoostMatchingSolution
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class EnsembleLearnerMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.file_format = 'embitto'
  
	def model_train(self, l_params, l_epochs, x_epochs, x_params,  numerical_config, wandb_id, include_numerical_features_in_textual, seed, textual_config, pretrain_batch_size, num_pretrain_epochs, finetune_batch_size, num_finetune_epochs, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, use_statistical_model, use_as_feature,use_as_decider,  output_embedding_size=256, lr=3e-5, should_pretrain=False, should_finetune=True):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False

        #! TODO: Select the correct dataset paths
		iteration = self.train_path[-5]
		lgbm_train_path = f"{Path(self.train_path).parent}/deep_matcher_train_{iteration}.csv"
		lgbm_valid_path = f"{Path(self.valid_path).parent}/deep_matcher_valid_{iteration}.csv"
		lgbm_test_path = f"{Path(self.test_path).parent}/deep_matcher_test_{iteration}.csv"
		self.embitto = EmbittoMatchingSolution(self.dataset_name, self.train_path, self.valid_path, self.test_path)
		number_numeric_cols = len(list(pd.read_csv(lgbm_train_path).drop(columns=['prediction', 'id'], errors="ignore").select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns))
		number_other_cols = len(list(pd.read_csv(lgbm_train_path))) - number_numeric_cols
		self.lightgbm = LightGBMMatchingSolution(self.dataset_name, lgbm_train_path, lgbm_valid_path, lgbm_test_path)
		self.xgboost = XGBoostMatchingSolution(self.dataset_name, lgbm_train_path, lgbm_valid_path, lgbm_test_path)
		
		#self.aggregation_model = LightGBMMatchingSolution(self.dataset_name, lgbm_train_path, lgbm_valid_path, lgbm_test_path)
		_, embitto_model, _, _ = self.embitto.model_train(numerical_config, wandb_id, include_numerical_features_in_textual, seed, textual_config, pretrain_batch_size, num_pretrain_epochs, finetune_batch_size, num_finetune_epochs, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, use_statistical_model, use_as_feature,use_as_decider)
		if number_numeric_cols == 0:
			return _, {'embitto': embitto_model}, _, _
            
		
		_, lightgbm_model, _, _ = self.lightgbm.model_train(l_params, l_epochs, wandb_id, seed, i)
		_, xgboost_model, _, _ = self.xgboost.model_train(x_params, x_epochs, wandb_id, seed, i)
		embitto_valid = self.embitto.model_predict_valid(embitto_model)
		lightgbm_valid = self.lightgbm.model_predict_valid(lightgbm_model)
		xgboost_valid = self.xgboost.model_predict_valid(xgboost_model)
		df_valid = pd.DataFrame({'embitto': embitto_valid, 'lightgbm': lightgbm_valid, 'xgboost': xgboost_valid, 'prediction': pd.read_csv(valid_goldstandard_path)['prediction']})
		valid_label = df_valid['prediction'].to_numpy()
		embitto_train = self.embitto.model_predict_train(embitto_model)
		lightgbm_train = self.lightgbm.model_predict_train(lightgbm_model)
		xgboost_train = self.xgboost.model_predict_train(xgboost_model)
		df_train = pd.DataFrame({'embitto': embitto_train, 'lightgbm': lightgbm_train, 'xgboost': xgboost_train, 'prediction': pd.read_csv(train_goldstandard_path)['prediction']})
		valid_label = df_valid['prediction'].to_numpy()
		train_label = df_train['prediction'].to_numpy()
		valid_data = xgb.DMatrix(df_valid.drop(columns=['prediction'], errors="ignore").to_numpy(), label=valid_label)
		train_data = xgb.DMatrix(df_train.drop(columns=['prediction'], errors="ignore").to_numpy(), label=train_label)
  
		train_tensor = TensorDataset(torch.Tensor(df_train.drop(columns=['prediction'], errors="ignore").to_numpy()), torch.LongTensor(train_label))
		train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
		dl_aggregator_model = SimpleNN()
		self.agg_trainer = Trainer(max_epochs=10)
		self.agg_trainer.fit(dl_aggregator_model, train_loader)
		aggregator_model = xgb.train(x_params, train_data, x_epochs)
		val_tensor = torch.Tensor(df_valid.drop(columns=['prediction'], errors="ignore").to_numpy())
		valid_tensor = TensorDataset(val_tensor)
		valid_loader = DataLoader(valid_tensor, batch_size=32)
		valid_pred = torch.cat(self.agg_trainer.predict(dl_aggregator_model, valid_loader), dim=0).flatten().numpy()
		#valid_pred = torch.cat(self.agg_trainer.predict(dl_aggregator_model, valid_loader), dim=0).float().softmax(dim=1)[:, 1]
		#valid_pred = aggregator_model.predict(xgb.DMatrix(df_valid.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()))
		
		# train_pred = aggregator_model.predict(xgb.DMatrix(df_train.drop(columns=['prediction'], errors="ignore").to_numpy()))

		_, _, thresholds = roc_curve(valid_label, valid_pred)
		optimal_idx = np.argmax([f1_score(valid_label, valid_pred >= thresh) for thresh in thresholds])
		optimal_threshold = thresholds[optimal_idx]
		return _, {'embitto': embitto_model, 'dl_aggregator_model': dl_aggregator_model, 'lightgbm': lightgbm_model, 'xgboost': xgboost_model, 'aggregator_model': aggregator_model, 'thresholds': optimal_threshold}, _, _
		

	def model_predict(self, model, cluster, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path, use_statistical_model, use_as_feature, use_as_decider, threshold=None):
		embitto_prediction = self.embitto.model_predict(model['embitto'], cluster, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path, use_statistical_model, use_as_feature, use_as_decider)
		iteration = self.train_path[-5]
		lgbm_train_path = f"{Path(self.train_path).parent}/deep_matcher_train_{iteration}.csv"
		if 'lightgbm' not in model.keys():
			return embitto_prediction
		lightgbm_prediction = self.lightgbm.model_predict(model['lightgbm'])
		xgboost_prediction = self.xgboost.model_predict(model['xgboost'])
		df = pd.DataFrame({'embitto': embitto_prediction['predict'][0]['score'],'lightgbm': lightgbm_prediction['predict'][0]['score'], 'xgboost': xgboost_prediction['predict'][0]['score']})
		test_tensor = TensorDataset(torch.Tensor(df.drop(columns=['prediction'], errors="ignore").to_numpy()))
		test_loader = DataLoader(test_tensor, batch_size=32)

		#ypred = model['aggregator_model'].predict(xgb.DMatrix(df.to_numpy()))
		#pred = self.agg_trainer.predict(model['dl_aggregator_model'], test_loader)
		ypred = torch.cat(self.agg_trainer.predict(model['dl_aggregator_model'], test_loader), dim=0).flatten().numpy()
		#ypred = torch.cat(pred, dim=0).flatten().numpy()
		result = pd.DataFrame({'score': ypred})
		result['prediction'] = 0
		result.loc[result['score']>model['thresholds'], 'prediction'] = 1
		return {'predict': [result], 'evaluate': None}



class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer_1 = nn.Linear(3, 128)  # Input layer to hidden layer
        self.layer_2 = nn.Linear(128, 1)  # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = torch.relu(self.layer_1(x))
        #return self.layer_2(x)
        x = self.sigmoid(self.layer_2(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        #loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = nn.BCELoss()(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.BCELoss()(y_hat, y.float())
        #val_loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', val_loss)
