from abc import ABC, abstractmethod


class BaseAggregator(ABC):
	def __init__(self, name: str):
		self.name = name

	@abstractmethod
	def aggregate(self, input_embedding_1, input_embedding_2):
		pass

	@abstractmethod
	def get_size(self, input_embedding_size_1: int, input_embedding_size_2: int) -> int:
		pass