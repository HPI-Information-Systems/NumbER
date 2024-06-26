from NumbER.matching_solutions.gaussian.gaussian import Gaussian
from NumbER.matching_solutions.gaussian.data_loader import GaussianDataModule
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_curve
from pytorch_lightning.loggers import WandbLogger
import lightgbm as lgb
import wandb
import os
from sklearn import metrics
import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import random
from NumbER.matching_solutions.matching_solutions.lightgbm import LightGBMMatchingSolution

class GaussianMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.file_format = 'embitto'
  
	def model_train(self, l_epochs, l_params, x_epochs, x_params, wandb_id, include_numerical_features_in_textual, seed, textual_config, finetune_batch_size, num_finetune_epochs, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, lr=3e-5):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		train_record_data = pd.read_csv(self.train_path)
		valid_record_data = pd.read_csv(self.valid_path)
		test_record_data = pd.read_csv(self.test_path)
		train_goldstandard_data = pd.read_csv(train_goldstandard_path)
		valid_goldstandard_data = pd.read_csv(valid_goldstandard_path)
		test_goldstandard_data = pd.read_csv(test_goldstandard_path)
		textual_component = textual_config['model'](textual_config['max_length'], textual_config['embedding_size'])
		textual_finetune_formatter = textual_config['finetune_formatter']
		iteration = self.train_path[-5]
		lgbm_train_path = f"{Path(self.train_path).parent}/deep_matcher_train_{iteration}.csv"
		lgbm_valid_path = f"{Path(self.valid_path).parent}/deep_matcher_valid_{iteration}.csv"
		lgbm_test_path = f"{Path(self.test_path).parent}/deep_matcher_test_{iteration}.csv"
		number_numeric_cols = len(list(pd.read_csv(lgbm_train_path).drop(columns=['prediction', 'id'], errors="ignore").select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns))

		#just using it to build training data. Refactor by auslagern die funktionen
		self.lightgbm = LightGBMMatchingSolution(self.dataset_name, lgbm_train_path, lgbm_valid_path, lgbm_test_path)
		lgbm_train_data = self.lightgbm.calculate_feature_vector(pd.read_csv(lgbm_train_path)).fillna(0).drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		lgbm_valid_data = self.lightgbm.calculate_feature_vector(pd.read_csv(lgbm_valid_path)).fillna(0).drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		lgbm_test_data = self.lightgbm.calculate_feature_vector(pd.read_csv(lgbm_test_path)).fillna(0).drop(columns=['prediction', 'id'], errors="ignore").to_numpy()

		gaussian = Gaussian(
         	textual_component=textual_component,
			learning_rate=lr,
        )		
		self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_finetune_formatter, lgbm_train_data, include_numerical_features_in_textual, train_record_data)
		self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_finetune_formatter, lgbm_valid_data, include_numerical_features_in_textual, train_record_data)
		self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_finetune_formatter, lgbm_test_data, include_numerical_features_in_textual, train_record_data)
		self.data = GaussianDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, batch_size=finetune_batch_size)
		checkpoint_callback_1 = ModelCheckpoint(
					monitor="val_loss",
					mode="min",
					save_top_k=1,
					dirpath="/hpi/fs00/share/fg-naumann/lukas.laskowski/saved_models/gaussian/",
					filename=f"best_model_correct_{wandb_id}",
				)
		trainer = pl.Trainer(accelerator="gpu",precision="16-mixed", devices=1, logger=WandbLogger(), max_epochs=num_finetune_epochs,callbacks=[checkpoint_callback_1])#,logger=logger, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
		trainer.fit(gaussian, self.data)
		#add xgboost and lightgbm
		gaussian = gaussian.load_from_checkpoint(checkpoint_callback_1.best_model_path,learning_rate=lr, textual_component=textual_component)
		return None, gaussian, None, None

	def model_predict(self, model, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path):
		trainer = pl.Trainer()
		valid_gold = pd.read_csv(valid_goldstandard_path)['prediction']
		#eval = trainer.test(model, self.data)
		#print("Eval: ", eval)
		#predict_data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN if cluster else Stage.FINETUNING)
		valid = GaussianDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.valid_data)
		valid_predictions = torch.cat(trainer.predict(model, valid), dim=0).float().softmax(dim=1)[:, 1]
		predict_data = self.data
		test_predictions = torch.cat(trainer.predict(model, predict_data), dim=0).float().softmax(dim=1)[:, 1]
		_, _, thresholds = roc_curve(valid_gold, valid_predictions)
		optimal_idx = np.argmax([f1_score(valid_gold, valid_predictions >= thresh) for thresh in thresholds])
		optimal_threshold = thresholds[optimal_idx]
		pairs = self.data.test_dataset.groundtruth
		pairs['score'] = test_predictions
		pairs["prediction"] = [1 if p >= optimal_threshold else 0 for p in test_predictions]
		print("Wandb id: ", wandb.run.id)
		path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/saved_models/gaussian"
		if os.path.exists(f"{path}/best_model_correct_{wandb.run.id}.ckpt"):
			os.remove(f"{path}/best_model_correct_{wandb.run.id}.ckpt")
		return {'predict': [pairs], 'evaluate': None}	

	@staticmethod
	def get_numeric_columns(df):
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		cols = list(df.select_dtypes(include=numerics).columns)
		cols.append("id") if "id" not in cols else None
		print("NUMERIC COLS", cols)
		#return df.columns #!change this
		if len(cols) == 1:
			if cols[0] != "id":
				print("ALAAARM", cols)
				return None
			return None
		return cols
	
	@staticmethod
	def get_textual_columns(df, include_numerical_features_in_textual):
		data = df.select_dtypes(include=['object'])#, 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
		cols = list(data.columns)
		cols.append("id") if "id" not in cols else None
		print("TEXTUAL COLS", cols)
		if include_numerical_features_in_textual:
			return df.columns
		else:
			if len(cols) == 1:
				if cols[0] != "id":
					print("ALAAARM", cols)
					return None
				return None
			return cols

	def process_dataframe(self, df: pd.DataFrame, matches: pd.DataFrame, textual_formatter, numerical_feature_vector, include_numerical_features_in_textual=True, train_data=None):
		textual_columns = self.get_textual_columns(df, include_numerical_features_in_textual)
		textual_data = df[textual_columns]
		textual_data = self.build_pairs(textual_data, matches)
		if textual_formatter.__name__ in ["complete_prompt_formatter", "complete_prompt_formatter_min_max_scaled", "text_sim_formatter", "textual_min_max_scaled", "num_text_sim_formatter"]:
			textual_data = textual_formatter(data=textual_data, train_data=train_data) if textual_data is not None else None
		else:
			textual_data = textual_formatter(textual_data)
		print("textualformatter", textual_formatter)
		return {'all_data': df, 'textual_data': textual_data, 'numerical_feature_vector': numerical_feature_vector, 'matches': matches}

	def build_pairs(self, df: pd.DataFrame, matches: pd.DataFrame):
		pairs = []
		df.drop(columns=['entity_id'], inplace=True) if 'entity_id' in df.columns else None
		columns = df.columns
		left_columns = ["left_" + column for column in columns]	
		right_columns = ["right_" + column for column in columns]
		for _, match in matches.iterrows():
			record_1 = df[df["id"] == match['p1']]
			record_2 = df[df["id"] == match['p2']]
			assert len(record_1) == 1
			assert len(record_2) == 1
			pairs.append((*record_1.values[0], *record_2.values[0]))
		return pd.DataFrame(pairs, columns=left_columns + right_columns)