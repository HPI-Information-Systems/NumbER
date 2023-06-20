from NumbER.matching_solutions.embitto.aggregators.aggregators.base_aggregator import BaseAggregator
import torch

class ConcatenationAggregator(BaseAggregator):
    def aggregate(self, input_embedding_1, input_embedding_2):
        return torch.cat((input_embedding_1, input_embedding_2), dim=1)
    
    def get_size(self, input_embedding_size_1: int, input_embedding_size_2: int) -> int:
        return input_embedding_size_1 + input_embedding_size_2