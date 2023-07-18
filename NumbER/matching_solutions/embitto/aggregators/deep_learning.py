from torch import nn
import torch
from NumbER.matching_solutions.embitto.enums import Stage
from NumbER.matching_solutions.embitto.aggregators.aggregators.base_aggregator import BaseAggregator


class EmbeddimgFusion(nn.Module):
    def __init__(self, embedding_combinator: BaseAggregator, textual_input_embedding_size: int = 256, numerical_input_embedding_size: int = 256, output_embedding_size: int = 256):
        super(EmbeddimgFusion, self).__init__()
        self.embedding_aggregator = embedding_combinator()
        input_embeddings_size = self.embedding_aggregator.get_size(textual_input_embedding_size, numerical_input_embedding_size)
        self.output_embedding_size = output_embedding_size
        self.embedding_output = nn.Linear(input_embeddings_size, output_embedding_size)
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, textual_embeddings, numerical_embeddings):
        numerical_embeddings = numerical_embeddings.to(torch.float32)
        textual_embeddings = textual_embeddings.to(torch.float32) if textual_embeddings is not None else None
        embeddings = self.embedding_aggregator.aggregate(textual_embeddings, numerical_embeddings) if textual_embeddings is not None else numerical_embeddings
        embeddings = self.embedding_output(embeddings)
        return embeddings