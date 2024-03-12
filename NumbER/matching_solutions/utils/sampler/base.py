import pandas as pd
from abc import ABC, abstractmethod
from NumbER.matching_solutions.utils.output_formats import MD2MFormat, DittoFormat, DeepMatcherFormat, EmbittoFormat
from NumbER.matching_solutions.utils.output_formats import DummyFormat
import networkx as nx
import os
import pathlib

class BaseSampler(ABC):
	def __init__(self, records_path, goldstandard_path):
		self.records = pd.read_csv(records_path, index_col=None)
		self.records = self.records.replace({'\t': ''}, regex=True)
		self.records['id'] = self.records['id'].astype(float)
		self.records_path = records_path
		try:
			self.goldstandard = pd.read_csv(goldstandard_path, index_col=None)
			self.goldstandard_path = goldstandard_path
			if "prediction" in self.goldstandard.columns:
				print("Prediction is inside of it. Filtering only the matching pairs")
				self.goldstandard = self.goldstandard[self.goldstandard['prediction'] == 1]
			print("Goldstandard dwad", self.goldstandard)
		except:
			print("NO GROUNDTRUTH!")
			self.goldstandard = None

	def create_clusters(self, groundtruth):
		print("Creating clusters")
		groundtruth = groundtruth[['p1', 'p2']].to_numpy()
		print("Groundtruth", groundtruth)
		G = nx.Graph()
		G.add_edges_from(groundtruth)
		clusters = []
		for connected_component in nx.connected_components(G):
			clusters.append(list(connected_component))
		print("Clusters", clusters)
		return clusters

	def create_format(self, output_format, config):
		print("Creating format")
		train_data, valid_data, test_data = self.sample(config)
		print("Lengths", len(train_data), len(valid_data), len(test_data))
		print("Train_data: ", train_data)
		print("Valid_data: ", valid_data)
		print("Test_data: ", test_data)
		if config['constant_based']:
			print("Constant based")
			print(self.records.columns)
			train_formatter = DummyFormat(train_data, self.records, output_format, os.path.join(pathlib.Path(self.records_path).parent, 'samples', self.name))
			valid_formatter = DummyFormat(valid_data, self.records, output_format, os.path.join(pathlib.Path(self.records_path).parent, 'samples', self.name))
			test_formatter = DummyFormat(test_data, self.records, output_format, os.path.join(pathlib.Path(self.records_path).parent, 'samples', self.name))
			return train_formatter, valid_formatter, test_formatter
		print("SAMPLED", train_data)
		print(valid_data)
		print(test_data)
		print("OUTPUTFORMATS", output_format)
		if output_format == 'deep_matcher' or output_format == "xgboost":
			Formatter = DeepMatcherFormat
		elif output_format == 'ditto':
			Formatter = DittoFormat
		elif output_format == 'md2m':
			Formatter = MD2MFormat
		elif output_format == 'embitto':
			Formatter = EmbittoFormat
		if self.name != "deep_matcher_datasets":
			self.check_no_leakage(train_data, valid_data, test_data)
		for data in [train_data, valid_data, test_data]:
			print("CHECKING CORRECTNESS")
			if self.name != "deep_matcher_datasets":
				assert self.check_correctnesss(data)
			assert len(data.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1).unique()) == len(data)
			print("Amount of matching pairs", len(data[data['prediction'] == 1]))
			print("Amount of non-matching pairs", len(data[data['prediction'] == 0]))
		return Formatter(train_data, self.records), Formatter(valid_data, self.records), Formatter(test_data, self.records)

	def check_correctnesss(self, data):
		goldstandard = self.goldstandard[['p1', 'p2', 'prediction']]
		data['key'] = data.apply(lambda row: tuple(sorted([row['p1'], row['p2']])), axis=1)
		goldstandard['key'] = goldstandard.apply(lambda row: tuple(sorted([row['p1'], row['p2']])), axis=1)
		merged = pd.merge(data, goldstandard, on='key', how='inner')
		return all(merged['prediction_x'] == merged['prediction_y'])

	def check_no_leakage(self, train_data, valid_data, test_data):
		train_ids = set(train_data['p1'].tolist() + train_data['p2'].tolist())
		valid_ids = set(valid_data['p1'].tolist() + valid_data['p2'].tolist())
		test_ids = set(test_data['p1'].tolist() + test_data['p2'].tolist())
		train_data.to_csv('train_data.csv', index=False)
		valid_data.to_csv('valid_data.csv', index=False)
		test_data.to_csv('test_data.csv', index=False)
		assert len(train_ids.intersection(valid_ids)) == 0
		assert len(train_ids.intersection(test_ids)) == 0
		assert len(valid_ids.intersection(test_ids)) == 0
	
	@abstractmethod
	def sample(self, config):
		raise NotImplementedError

	def convert_to_pairs(self, cluster_groups):
		pairs = []
		for cluster in cluster_groups:
			for i in range(len(cluster)):
				for j in range(i+1, len(cluster)):
					pairs.append((cluster[i], cluster[j]))
		return pd.DataFrame(pairs, columns=['p1', 'p2'])