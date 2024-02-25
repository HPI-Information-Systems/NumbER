from NumbER.matching_solutions.embitto.embitto import Embitto
from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.data_loader import EmbittoDataModule
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from NumbER.matching_solutions.embitto.aggregators.aggregators.concatenation import ConcatenationAggregator
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter, numeric_prompt_formatter
from NumbER.matching_solutions.embitto.aggregators.deep_learning import EmbeddimgFusion
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import DBSCAN
import wandb
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.neighbors import KNeighborsClassifier
from transformers import AdamW
import numpy as np
from NumbER.matching_solutions.embitto.numerical_components.base_component import BaseNumericComponent
import torch
from NumbER.matching_solutions.utils.transitive_closure import convert_to_pairs, calculate_from_entity_ids
import pytorch_lightning as pl
import pandas as pd
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from pytorch_lightning.callbacks import ModelCheckpoint
import random

class EmbittoMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.file_format = 'embitto'
  
	def model_train(self, numerical_config, wandb_id, include_numerical_features_in_textual, seed, textual_config, pretrain_batch_size, num_pretrain_epochs, finetune_batch_size, num_finetune_epochs, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, output_embedding_size=256, lr=3e-5,should_pretrain=False, should_finetune=True):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		stage = Stage.PRETRAIN
		train_record_data = pd.read_csv(self.train_path)
		print("train", train_record_data)
		print("train_matches", train_goldstandard_path)
		valid_record_data = pd.read_csv(self.valid_path)
		test_record_data = pd.read_csv(self.test_path)
		train_goldstandard_data = pd.read_csv(train_goldstandard_path)
		valid_goldstandard_data = pd.read_csv(valid_goldstandard_path)
		test_goldstandard_data = pd.read_csv(test_goldstandard_path)
		numerical_embedding_size = 40
		textual_component = textual_config['model'](textual_config['max_length'], textual_config['embedding_size'], Stage.PRETRAIN)

		textual_pretrain_formatter = textual_config['pretrain_formatter']
		numerical_pretrain_formatter = numerical_config['pretrain_formatter']
		textual_finetune_formatter = textual_config['finetune_formatter']
		numerical_finetune_formatter = numerical_config['finetune_formatter']
		self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_pretrain_formatter, numerical_pretrain_formatter, include_numerical_features_in_textual, stage=Stage.PRETRAIN)
		numeric_order = self.train_data["numerical_data"].columns if self.train_data["numerical_data"] is not None else None
		self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_pretrain_formatter, numerical_pretrain_formatter,include_numerical_features_in_textual, stage=Stage.PRETRAIN)
		self.valid_data["numerical_data"] = self.valid_data["numerical_data"][numeric_order] if self.valid_data["numerical_data"] is not None else None
		self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_pretrain_formatter, numerical_pretrain_formatter,include_numerical_features_in_textual, stage=Stage.PRETRAIN)
		if numeric_order is not None:
			self.test_data["numerical_data"] = self.test_data["numerical_data"][numeric_order] if self.test_data["numerical_data"] is not None else None
		else:
			self.test_data["numerical_data"] = None
		if numerical_config['model'] is not None and self.train_data["numerical_data"] is not None:
			numerical_component = numerical_config['model'](self.train_data["numerical_data"], self.valid_data["numerical_data"], self.test_data["numerical_data"], numerical_config['embedding_size'], should_pretrain=should_pretrain)
			fusion_component = EmbeddimgFusion(
				embedding_combinator=ConcatenationAggregator,
				textual_input_embedding_size = textual_config['embedding_size'] if self.train_data["textual_data"] is not None else 0,
				numerical_input_embedding_size = numerical_component.get_outputshape(),#*2,
				output_embedding_size = output_embedding_size
			)
		else:
			numerical_component = None
			fusion_component = None
		# numerical_component = None
		# fusion_component = None
		#textual_component = None
		print("NUMERICAL", numerical_component)
		print("TEXTUAL", textual_component)
		print("FUSION", fusion_component)
		embitto = Embitto(
      		stage=Stage.PRETRAIN,
        	numerical_component=numerical_component,
         	textual_component=textual_component,
          	fusion_component=fusion_component,
			learning_rate=lr,
			should_pretrain=should_pretrain,
        )
		
		self.data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN, batch_size=pretrain_batch_size)
		logger = pl_loggers.CSVLogger('logs/', name='embitto')
		#embitto = embitto.load_from_checkpoint(checkpoint_path, stage=Stage.FINETUNING, numerical_component=numerical_component, textual_component=textual_component, fusion_component=fusion_component)
		trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=num_pretrain_epochs)#, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
		if should_pretrain:
			print("HALLO")
			trainer.fit(embitto, self.data)
		
		print("STAGE", should_finetune)	
		stage = Stage.FINETUNING
		if should_finetune:
			embitto.set_stage(stage)
			self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_finetune_formatter, numerical_finetune_formatter, include_numerical_features_in_textual,stage, train_record_data)
			self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_finetune_formatter, numerical_finetune_formatter,include_numerical_features_in_textual, stage, train_record_data)
			self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_finetune_formatter, numerical_finetune_formatter,include_numerical_features_in_textual, stage, train_record_data)
			self.data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.FINETUNING, batch_size=finetune_batch_size)
			checkpoint_callback_1 = ModelCheckpoint(
						monitor="val_loss",
						mode="min",
						save_top_k=1,
						dirpath="/hpi/fs00/share/fg-naumann/lukas.laskowski/saved_models/",
						filename=f"best_model_correct_{wandb_id}",
					)
			trainer = pl.Trainer(accelerator="gpu",precision="16-mixed", devices=1, logger=WandbLogger(), max_epochs=num_finetune_epochs,callbacks=[checkpoint_callback_1])#,logger=logger, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
			trainer.fit(embitto, self.data)
			embitto = embitto.load_from_checkpoint(checkpoint_callback_1.best_model_path, stage=Stage.FINETUNING, numerical_component=numerical_component,learning_rate=lr, textual_component=textual_component, fusion_component=fusion_component, should_pretrain=should_pretrain)
		return None, embitto, None, None

	def model_predict(self, model, cluster, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path):
		trainer = pl.Trainer()
		#eval = trainer.test(model, self.data)
		#print("Eval: ", eval)
		#predict_data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN if cluster else Stage.FINETUNING)
		valid = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.valid_data, stage=Stage.FINETUNING)
		valid_predictions = trainer.predict(model, valid)
		valid_predictions = torch.cat(valid_predictions, dim=0)
		predict_data = self.data
		test_predictions = trainer.predict(model, predict_data)
		test_predictions = torch.cat(test_predictions, dim=0)
		if cluster:
			neigh = DBSCAN(eps=0.9, min_samples=1)
			entity_ids = neigh.fit_predict(test_predictions)
			self.data.test_dataset.df['pred_entity_ids'] = entity_ids
			self.data.test_dataset.df.to_csv("test_data.csv")
			clusters = calculate_from_entity_ids(self.data.test_dataset.df)
			pairs = convert_to_pairs(clusters)
			pairs["score"] = 1.0
		else:
			test_predictions = test_predictions.softmax(dim=1)[:, 1]
			valid_predictions = valid_predictions.softmax(dim=1)[:, 1]
			best_th = 0.5
			f1 = 0.0 # metrics.f1_score(all_y, all_p)

			for th in np.arange(0.0, 1.0, 0.05):
				pred = [1 if p > th else 0 for p in valid_predictions]
				new_f1 = metrics.f1_score(self.valid_data["matches"]["prediction"].values, pred)
				if new_f1 > f1:
					f1 = new_f1
					best_th = th
			pairs = self.data.test_dataset.groundtruth
			pairs['score'] = test_predictions
			pairs["prediction"] = [1 if p > best_th else 0 for p in test_predictions]
		print("Wandb id: ", wandb.run.id)
		path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/saved_models"
		if os.path.exists(f"{path}/best_model_correct_{wandb.run.id}.ckpt"):
			os.remove(f"{path}/best_model_correct_{wandb.run.id}.ckpt")
		return {'predict': [pairs], 'evaluate': None}

		#entity_ids = neigh.predict(test_predictions)#returned die entity ids
		#mapping zu test_data machen
		#model(data=self.data, stage=Stage.FINETUNE)
		
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

	def process_dataframe(self, df: pd.DataFrame, matches: pd.DataFrame, textual_formatter, numerical_formatter = dummy_formatter,include_numerical_features_in_textual=True, stage = Stage.PRETRAIN, train_data=None):
		textual_columns = self.get_textual_columns(df, include_numerical_features_in_textual)
		numerical_columns = self.get_numeric_columns(df)
		print(numerical_columns)
		print("df", df)

		textual_data = df[textual_columns] if textual_columns is not None else None
		numerical_data = df[numerical_columns] if numerical_columns is not None else None
		if stage == Stage.FINETUNING:
			textual_data = self.build_pairs(textual_data, matches) if textual_data is not None else None
			numerical_data = self.build_pairs(numerical_data, matches) if numerical_data is not None else None
		if textual_formatter.__name__ in ["complete_prompt_formatter", "complete_prompt_formatter_min_max_scaled", "text_sim_formatter", "textual_min_max_scaled", "num_text_sim_formatter"]:
			textual_data = textual_formatter(data=textual_data, train_data=train_data) if textual_data is not None else None
		else:
			textual_data = textual_formatter(textual_data) if textual_data is not None else None
		numerical_data = numerical_formatter(numerical_data) if numerical_data is not None else None
		print("textualformatter", textual_formatter)
		return {'all_data': df, 'numerical_data': numerical_data, 'textual_data': textual_data, 'matches': matches}

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