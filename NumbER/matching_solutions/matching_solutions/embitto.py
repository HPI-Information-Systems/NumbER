from NumbER.matching_solutions.embitto.embitto import Embitto, Stage
from NumbER.matching_solutions.embitto.data_loader import EmbittoDataModule
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

import pytorch_lightning as pl
import pandas as pd
from NumbER.matching_solutions.embitto.textual_components.base_roberta import RobertaClassifier

class EmbittoMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.file_format = 'embitto'
  
	def model_train(self, train_goldstandard_path, valid_goldstandard_path, test_goldstandard_path, i):
		train_record_data = pd.read_csv(self.train_path)
		valid_record_data = pd.read_csv(self.valid_path)
		test_record_data = pd.read_csv(self.test_path)
		train_goldstandard_data = pd.read_csv(train_goldstandard_path)
		valid_goldstandard_data = pd.read_csv(valid_goldstandard_path)
		test_goldstandard_data = pd.read_csv(test_goldstandard_path)
		train_data = self.process_dataframe(train_record_data, train_goldstandard_data)
		valid_data = self.process_dataframe(valid_record_data, valid_goldstandard_data)
		test_data = self.process_dataframe(test_record_data, test_goldstandard_data)
		embitto = Embitto(stage=Stage.PRETRAIN, numerical_component=RobertaClassifier, textual_component=RobertaClassifier)
		self.data = EmbittoDataModule(train_data=train_data, valid_data=valid_data, test_data=test_data, phase=Stage.PRETRAIN)
		trainer = pl.Trainer(accelerator="gpu", devices=1)
		print("HAL")
		trainer.fit(embitto, self.data)
		print("HAL2")
		return None, embitto, None, None

	def model_predict(self, model):
		trainer = pl.Trainer()
		eval = trainer.test(model, self.data)
		print("Eval: ", eval)
		predictions = trainer.predict(model, self.data)
		print("Predictions: ", predictions)
		
	def get_numeric_columns(self,df):
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		return df.select_dtypes(include=numerics).columns

	def get_values(self, df, columns):
		data = df[columns]
		#result = data.copy()
		result = []
		for idx, record in data.iterrows():
			temp = ""
			for col, val in record.items():
				temp += f"COL {col} VAL {val} "
			#result.loc[idx] = temp
			result.append(temp)
		return result
		#input: die record liste.

	def get_textual_columns(self, df):
		data = df.select_dtypes(include=['object'])
		return data.columns

	def process_dataframe(self, df: pd.DataFrame, matches: pd.DataFrame):
		textual_columns = self.get_textual_columns(df)
		numerical_columns = self.get_numeric_columns(df)
		textual_data = self.get_values(df, textual_columns)
		numerical_data = self.get_values(df, numerical_columns)
		#for finetuning adapten
		return {'all_data': df,'numerical_data': numerical_data, 'textual_data': textual_data, 'matches': matches}