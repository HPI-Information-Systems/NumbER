from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
from transformers import RobertaModel, RobertaTokenizer
from NumbER.matching_solutions.embitto.enums import Stage, Phase
import numpy as np
import torch
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.metrics import f1_score, roc_curve
import random
import lightgbm as lgb


class LightGBMMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path, llm_train_predictions=None, llm_valid_predictions=None, llm=None):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
		self.llm_train_predictions = llm_train_predictions 
		self.llm_valid_predictions = llm_valid_predictions
		# word2vec_path = "/hpi/fs00/home/lukas.laskowski/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"
		# self.word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		model_name = 'roberta-base'
		# if llm == None:
		self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
		self.model = RobertaModel.from_pretrained(model_name).to("cuda")
		self.model.eval()
		# else:
		# self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
		# self.model = llm
		#self.model.eval()

	def model_train(self, params, epochs, wandb_id, seed, i):
		print("PARAMS", params)
		random.seed(seed)
		np.random.seed(seed)
		def build_data(path, additional=None):
			data = pd.read_csv(path)
			label = data["prediction"].to_numpy()
			data = self.calculate_feature_vector(data)
			if additional != None:
				data['llm'] = additional
			return lgb.Dataset(data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy(), label=label)
		train_data = build_data(self.train_path, self.llm_train_predictions)
		print("Train", train_data)
		valid_data = build_data(self.valid_path, self.llm_valid_predictions)
		print(params)
		print(train_data)
		print(epochs)
		print(valid_data)
		bst = lgb.train(params, train_data, epochs, valid_sets=[valid_data])
		return None, bst, None, None

	def model_predict(self, model, mode="test", llm_train_data=None, llm_valid_data=None, llm_test_data=None):
		test_data = pd.read_csv(self.test_path)
		valid_data = pd.read_csv(self.valid_path)
		train_data = pd.read_csv(self.train_path)
		valid_gold = valid_data['prediction']
		valid_data = self.calculate_feature_vector(valid_data)
		if llm_valid_data != None:
			valid_data['llm'] = llm_valid_data
		train_data = self.calculate_feature_vector(train_data)
		if llm_train_data != None:
			train_data['llm'] = llm_train_data
		#print("Traindata", train_data.columns)
		#print("Valdidata", valid_data.columns)
		valid_data = valid_data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		train_data = train_data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		valid_pred = model.predict(valid_data)
		train_data = model.predict(train_data)
		if mode == "valid":
			return {'scores': valid_pred}
		if mode == "train":
			return {'scores': train_data}
		elif mode == "test":
			_, _, thresholds = roc_curve(valid_gold, valid_pred)
			optimal_idx = np.argmax([f1_score(valid_gold, valid_pred >= thresh) for thresh in thresholds])
			optimal_threshold = thresholds[optimal_idx]

			test_data = self.calculate_feature_vector(test_data)
			if llm_test_data != None:
				test_data['llm'] = llm_test_data
			test_data = test_data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
			ypred = model.predict(test_data)
			print("ypred", ypred)
			result = pd.DataFrame({'score': ypred})
			result['prediction'] = 0
			result.loc[result['score']>optimal_threshold, 'prediction'] = 1
			print(optimal_threshold)
			return {'predict': [result], 'evaluate': None}
		else:
			raise Exception("Mode can only have values: train/test/valid")

	def model_predict_valid(self, model):
		valid_data = pd.read_csv(self.valid_path)
		valid_data = self.calculate_feature_vector(valid_data)
		valid_data = valid_data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		return model.predict(valid_data)
	def model_predict_train(self, model):
		train_data = pd.read_csv(self.train_path)
		train_data = self.calculate_feature_vector(train_data)
		train_data = train_data.drop(columns=['prediction', 'id'], errors="ignore").to_numpy()
		return model.predict(train_data)

	def calculate_feature_vector(self, data: pd.DataFrame):
		final_df = pd.DataFrame()
		numeric_data = data[self.get_numeric_columns(data)]
		if len(data.columns) != len(self.get_numeric_columns(data)):
			print("numeric cl", self.get_numeric_columns(data))
			print("all cl", data.columns)
			textual_data = data[data.columns.difference(list(self.get_numeric_columns(data)))]
			print("textu", textual_data)
			for col in filter(lambda x: x.startswith("left"), textual_data.columns):
				col_1 = textual_data[col]
				col_2 = textual_data[f"right_{col[5:]}"]
				final_df[col[5:]] = self.calculate_similarities_for_column_pair(col_1, col_2)
		for col in filter(lambda x: x.startswith("left"), numeric_data.columns):
			final_df[col[5:]] = numeric_data[col] - numeric_data[f"right_{col[5:]}"]
		return final_df
	
	def generate_batch_embeddings(self, texts):
		inputs = self.tokenizer(texts.astype(str).values.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to("cuda")
		self.model.to("cuda")
		with torch.no_grad():
			outputs = self.model(**inputs)
			#outputs = self.model(inputs['input_ids'], None, Phase.TEST, get_hidden_stage=True)
		embeddings = outputs.cpu()  # Mean pooling across the token embeddings
		return embeddings
	
	def calculate_similarities_for_column_pair(self,col_1, col_2, batch_size=40):
		similarities = []
		for i in range(0, len(col_1), batch_size):
			batch_1 = col_1[i:i+batch_size]
			batch_2 = col_2[i:i+batch_size]
			embeddings1 = self.generate_batch_embeddings(batch_1)
			embeddings2 = self.generate_batch_embeddings(batch_2)
			batch_similarity_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).numpy()
			similarities.extend(batch_similarity_scores)
		return similarities
	
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
	
	def get_average_embedding(self, words, model):
		valid_words = [word for word in words if word in model.vocab]
		if not valid_words:
			# If no words are valid, return a zero vector
			return np.zeros(model.vector_size)
		embeddings = [model[word] for word in valid_words]
		average_embedding = np.mean(embeddings, axis=0)
		return average_embedding
