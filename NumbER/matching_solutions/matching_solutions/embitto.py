from NumbER.matching_solutions.embitto.embitto import Embitto, Stage
from NumbER.matching_solutions.embitto.data_loader import EmbittoDataModule
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from sklearn.cluster import DBSCAN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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
  
	def model_train(self, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path,i, should_finetune=True):
		train_record_data = pd.read_csv(self.train_path)
		valid_record_data = pd.read_csv(self.valid_path)
		test_record_data = pd.read_csv(self.test_path)
		train_goldstandard_data = pd.read_csv(train_goldstandard_path)
		valid_goldstandard_data = pd.read_csv(valid_goldstandard_path)
		test_goldstandard_data = pd.read_csv(test_goldstandard_path)
		textual_embedding_size = 256
		numerical_embedding_size = 256
		textual_max_length = 150
		textual_component = BaseRoberta(textual_max_length, textual_embedding_size, Stage.PRETRAIN)
		textual_formatter = textual_component.get_formatter()
		self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_formatter)
		self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_formatter)
		self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_formatter)
		numerical_component = None
		fusion_component = None
		embitto = Embitto(
      		stage=Stage.PRETRAIN,
        	numerical_component=numerical_component,
         	textual_component=textual_component,
          	fusion_component=fusion_component
        )
		checkpoint_callback = ModelCheckpoint(
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
		# trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger, max_epochs=70, callbacks=[checkpoint_callback])#, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
		# trainer.fit(embitto, self.data)
		stage = Stage.FINETUNING
		if should_finetune:
			embitto.set_stage(stage)
			textual_formatter = textual_component.get_formatter()
			self.train_data = self.process_dataframe(train_record_data, train_goldstandard_data, textual_formatter, stage)
			self.test_data = self.process_dataframe(test_record_data, test_goldstandard_data, textual_formatter, stage)
			self.valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data, textual_formatter, stage)
			self.data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.FINETUNING)
			trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=40, callbacks=[checkpoint_callback])#,logger=logger, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
			trainer.fit(embitto, self.data)
		return None, embitto, None, None

	def model_predict(self, model, cluster, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path):
		trainer = pl.Trainer()
		#eval = trainer.test(model, self.data)
		#print("Eval: ", eval)
		predict_data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.PRETRAIN if cluster else Stage.FINETUNING)
		predict_data = self.data
		print("Predict data: ", predict_data.test_dataset.groundtruth)
		print("Predict data: ", len(predict_data.test_dataset.textual_pairs))
		test_predictions = trainer.predict(model, predict_data)
		print("Test predictions: ", test_predictions)
		test_predictions = torch.cat(test_predictions, dim=0)
		print("Test predictions2: ", test_predictions)
		test_predictions = test_predictions.softmax(dim=1)[:, 1]
		print(test_predictions)
		print("Test predictions3: ", test_predictions)
		print("Cluster: ", cluster)
		if cluster:
			neigh = DBSCAN(eps=0.9, min_samples=1)
			entity_ids = neigh.fit_predict(test_predictions)
			self.data.test_dataset.df['pred_entity_ids'] = entity_ids
			self.data.test_dataset.df.to_csv("test_data.csv")
			clusters = calculate_from_entity_ids(self.data.test_dataset.df)
			pairs = convert_to_pairs(clusters)
		else:
			#predict_data = EmbittoDataModule(train_data=self.train_data, valid_data=self.valid_data, test_data=self.test_data, predict_data=self.test_data, stage=Stage.FINETUNING)
			print(self.data.test_dataset.groundtruth)
			#self.data.test_dataset.groundtruth['prediction'] = 0
			pairs = self.data.test_dataset.groundtruth
			pairs['score'] = test_predictions
			pairs["prediction"] = pairs["score"] > 0.5
		pairs.to_csv("pairs.csv")
		#pairs['score'] = 1.0
		return {'predict': [pairs], 'evaluate': None}

		#entity_ids = neigh.predict(test_predictions)#returned die entity ids
		#mapping zu test_data machen
		#model(data=self.data, stage=Stage.FINETUNE)
		
		
	def get_numeric_columns(self,df):
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		return df.select_dtypes(include=numerics).columns

	def get_textual_columns(self, df):
		data = df.select_dtypes(include=['object', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
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
		numerical_data = textual_formatter(numerical_data)
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
   
