from NumbER.matching_solutions.embitto.aggregators.aggregators.base_aggregator import BaseAggregator

class AverageAggregator(BaseAggregator):
	def aggregate(self, input_embedding_1, input_embedding_2):
		assert input_embedding_1.shape == input_embedding_2.shape
		return (input_embedding_1 + input_embedding_2) / 2
	
	def get_size(self, input_embedding_size_1: int, input_embedding_size_2: int) -> int:
		assert input_embedding_size_1 == input_embedding_size_2
		return input_embedding_size_1