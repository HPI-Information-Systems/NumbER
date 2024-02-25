from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve
import random
import lightgbm as lgb


class LightGBMMatchingSolution(MatchingSolution):
	def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
		super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
  
	def model_train(self, params, epochs, wandb_id, seed, i):
		print("PARAMS", params)
		random.seed(seed)
		np.random.seed(seed)
		def build_data(path):
			data = pd.read_csv(path)
			label = data["prediction"].to_numpy()
			return lgb.Dataset(data.drop(columns=['prediction', 'id']).to_numpy(), label=label)
		train_data = build_data(self.train_path)
		print("Train", train_data)
		valid_data = build_data(self.valid_path)
		bst = lgb.train(params, train_data, epochs, valid_sets=[valid_data])
		return None, bst, None, None

	def model_predict(self, model):
		test_data = pd.read_csv(self.test_path)
		valid_data = pd.read_csv(self.valid_path)
		valid_gold = valid_data['prediction']
		valid_data = valid_data.drop(columns=['prediction', 'id']).to_numpy()
		valid_pred = model.predict(valid_data)
		_, _, thresholds = roc_curve(valid_gold, valid_pred)
		optimal_idx = np.argmax([f1_score(valid_gold, valid_pred >= thresh) for thresh in thresholds])
		optimal_threshold = thresholds[optimal_idx]
		test_data = test_data.drop(columns=['prediction', 'id']).to_numpy()
		ypred = model.predict(test_data)
		print("ypred", ypred)
		result = pd.DataFrame({'score': ypred})
		result['prediction'] = 0
		result.loc[result['score']>optimal_threshold, 'prediction'] = 1
		print(optimal_threshold)
		return {'predict': [result], 'evaluate': None}