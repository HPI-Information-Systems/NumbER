from NumbER.matching_solutions.embitto.embitto import Embitto
from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.data_loader import EmbittoDataModule
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from sklearn.cluster import DBSCAN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.neighbors import KNeighborsClassifier
from transformers import AdamW
import numpy as np
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddings, DICEEmbeddingAggregator
import torch
from NumbER.matching_solutions.utils.transitive_closure import convert_to_pairs, calculate_from_entity_ids
import pytorch_lightning as pl
import pandas as pd
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from pytorch_lightning.callbacks import ModelCheckpoint

class EmbittoMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.file_format = 'embitto'
  
	def model_train(self, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, should_finetune=False):
		train_record_data = pd.read_csv(self.train_path)
		valid_record_data = pd.read_csv(self.valid_path)
		test_record_data = pd.read_csv(self.test_path)
		train_goldstandard_data = pd.read_csv(train_goldstandard_path)
		valid_goldstandard_data = pd.read_csv(valid_goldstandard_path)
		test_goldstandard_data = pd.read_csv(test_goldstandard_path)
		textual_embedding_size = 256
		numerical_embedding_size = 256
		textual_max_length = 256
		textual_component = BaseRoberta(textual_max_length, textual_embedding_size, Stage.PRETRAIN)
		textual_formatter = textual_component.get_formatter()
		
		self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_formatter)
		numeric_order = self.train_data["numerical_data"].columns
		self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_formatter)
		self.valid_data["numerical_data"] = self.valid_data["numerical_data"][numeric_order]
		self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_formatter)
		self.test_data["numerical_data"] = self.test_data["numerical_data"][numeric_order]
		train_dice_config = self.prepare_dice_embeddings(self.train_data["numerical_data"])
		valid_dice_config = self.prepare_dice_embeddings(self.valid_data["numerical_data"])
		test_dice_config = self.prepare_dice_embeddings(self.test_data["numerical_data"])
		numerical_component = DICEEmbeddingAggregator(
			train_dice=DICEEmbeddings(train_dice_config, numerical_embedding_size),
			valid_dice=DICEEmbeddings(valid_dice_config, numerical_embedding_size),
			test_dice=DICEEmbeddings(test_dice_config, numerical_embedding_size),
		)
		fusion_component = None
		embitto = Embitto(
      		stage=Stage.PRETRAIN,
        	numerical_component=numerical_component,
         	textual_component=textual_component,
          	fusion_component=fusion_component
        )
		# embitto.optimizer=AdamW(embitto.parameters(), lr=3e-5)
		# embitto, optimizer = amp.initialize(embitto, embitto.optimizer, opt_level='O2')
		# embitto.optimizer = optimizer
		checkpoint_callback_1 = ModelCheckpoint(
			monitor="val_loss",
			mode="min",
			save_top_k=1,
			dirpath="saved_models/",
			filename="best_model_pretran_x3_numeric",
		)
		checkpoint_callback_2 = ModelCheckpoint(
			monitor="val_loss",
			mode="min",
			save_top_k=1,
			dirpath="saved_models/",
			filename="best_model_finetune_x3_numeric",
		)
		self.data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN)
		logger = pl_loggers.CSVLogger('logs/', name='embitto')
		checkpoint_path = "saved_models/best_model_finetune-v1.ckpt"  # Path to the saved model checkpoint
		#embitto = embitto.load_from_checkpoint(checkpoint_path, stage=Stage.FINETUNING, numerical_component=numerical_component, textual_component=textual_component, fusion_component=fusion_component)
		trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger, max_epochs=70, callbacks=[checkpoint_callback_1])#, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
		trainer.fit(embitto, self.data)
		stage = Stage.FINETUNING
		if should_finetune:
			embitto.set_stage(stage)
			textual_formatter = textual_component.get_formatter()
			self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_formatter, stage)
			self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_formatter, stage)
			self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_formatter, stage)
			self.data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.FINETUNING)
			trainer = pl.Trainer(accelerator="gpu",precision=16, devices=1, max_epochs=32, callbacks=[checkpoint_callback_2])#,logger=logger, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
			trainer.fit(embitto, self.data)
		return None, embitto, None, None

	def model_predict(self, model, cluster, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path):
		trainer = pl.Trainer()
		#eval = trainer.test(model, self.data)
		#print("Eval: ", eval)
		predict_data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN if cluster else Stage.FINETUNING)
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
			pairs = self.data.test_dataset.groundtruth
			pairs['score'] = test_predictions
			pairs["prediction"] = pairs["score"] > 0.5
		pairs.to_csv("pairs.csv")
		return {'predict': [pairs], 'evaluate': None}

		#entity_ids = neigh.predict(test_predictions)#returned die entity ids
		#mapping zu test_data machen
		#model(data=self.data, stage=Stage.FINETUNE)
		
	def prepare_dice_embeddings(self, data: pd.DataFrame):
		dice_config = []
		for column in data.columns:
			lower_bound = data[column].quantile(.2)
			upper_bound = data[column].quantile(.8)
			dice_config.append({
				'embedding_dim': 10,
				'lower_bound': lower_bound,
				'upper_bound': upper_bound
			})
		return dice_config

	def get_numeric_columns(self,df):
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		return df.select_dtypes(include=numerics).columns

	def get_textual_columns(self, df):
		data = df.select_dtypes(include=['object', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
		return df.columns
		return data.columns

	def process_dataframe(self, df: pd.DataFrame, matches: pd.DataFrame, textual_formatter, stage = Stage.PRETRAIN):
		textual_columns = self.get_textual_columns(df)
		numerical_columns = self.get_numeric_columns(df)
		textual_data = df[textual_columns]
		numerical_data = df[numerical_columns]
		if stage == Stage.FINETUNING:
			textual_data = self.build_pairs(textual_data, matches)
			numerical_data = self.build_pairs(numerical_data, matches)
		textual_data = textual_formatter(textual_data)
		#numerical_data = textual_formatter(numerical_data) #todo warum war das drin
		# print("Textual data: ", textual_data)
		# print("Numerical data: ", numerical_data)
		# numerical_data = self.get_values(df, numerical_columns)
		return {'all_data': df, 'numerical_data': numerical_data, 'textual_data': textual_data, 'matches': matches}

	def build_pairs(self, df: pd.DataFrame, matches: pd.DataFrame):
		pairs = []
		df.drop(columns=['entity_id'], inplace=True)
		columns = df.columns
		left_columns = ["left_" + column for column in columns]	
		right_columns = ["right_" + column for column in columns]
		for _, match in matches.iterrows():
			record_1 = df[df["id"] == match['p1']]
			record_2 = df[df["id"] == match['p2']]
			pairs.append((*record_1.values[0], *record_2.values[0]))
		return pd.DataFrame(pairs, columns=left_columns + right_columns)
   
