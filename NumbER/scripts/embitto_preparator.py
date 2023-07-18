from NumbER.matching_solutions.utils.sampler.similarity_based import SimilarityBasedSampler
from NumbER.matching_solutions.utils.sampler.sorted_neighbourhood import SortedNeighbourhoodSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddingAggregator
from NumbER.matching_solutions.embitto.numerical_components.value_embeddings import ValueTransformerEmbeddings, ValueBaseEmbeddings
from NumbER.matching_solutions.embitto.numerical_components.numeric_roberta import NumericRoberta
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter,textual_prompt_formatter, complete_prompt_formatter, ditto_formatter, numeric_prompt_formatter, pair_based_numeric_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific

embitto_only_textual = {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": [None, ValueBaseEmbeddings, ValueTransformerEmbeddings, DICEEmbeddingAggregator],
							"pretrain_formatter": dummy_formatter,
       						"finetune_formatter": pair_based_numeric_formatter
						},
					"textual_config":
						{
							"model": BaseRoberta,
							"max_length": 256,
							"embedding_size": 128,
							"pretrain_formatter": ditto_formatter,
							"finetune_formatter": [pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter]
						},
					"num_pretrain_epochs": 50,
					"pretrain_batch_size": 50,
					"finetune_batch_size": 50,
					"num_finetune_epochs": 40,#30
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True,
					"include_numerical_features_in_textual": [True, False],
				},
				"test": {
					"cluster": False
				}
			}