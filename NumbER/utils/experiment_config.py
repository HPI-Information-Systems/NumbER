from NumbER.matching_solutions.utils.sampler.similarity_based import SimilarityBasedSampler
from NumbER.matching_solutions.utils.sampler.sorted_neighbourhood import SortedNeighbourhoodSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddingAggregator
from NumbER.matching_solutions.embitto.numerical_components.value_embeddings import ValueTransformerEmbeddings, ValueBaseEmbeddings
from NumbER.matching_solutions.embitto.numerical_components.numeric_roberta import NumericRoberta
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter,textual_prompt_formatter, complete_prompt_formatter, ditto_formatter, numeric_prompt_formatter, pair_based_numeric_formatter, complete_prompt_formatter_min_max_scaled,complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific
embitto_only_textual = {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": None,
							"pretrain_formatter": dummy_formatter,
       						"finetune_formatter": pair_based_numeric_formatter
						},
					"textual_config":
						{
							"model": BaseRoberta,
							"max_length": 256,
							"embedding_size": 128,
							"pretrain_formatter": ditto_formatter,
							"finetune_formatter": complete_prompt_formatter
						},
					"num_pretrain_epochs": 50,
					"pretrain_batch_size": 50,
					"finetune_batch_size": 50,
					"num_finetune_epochs": 40,#30
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True,
     				"include_numerical_features_in_textual": True
				},
				"test": {
					"cluster": False
				}
			}
finetune_formatters = [textual_prompt_formatter, pair_based_ditto_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific]
combinations = [
	{
		"numerical_model": None,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 256,
	},
 	{
		"numerical_model": ValueBaseEmbeddings,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
	{
		"numerical_model": ValueTransformerEmbeddings,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
 	{
		"numerical_model": DICEEmbeddingAggregator,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
  {
		"numerical_model": ValueBaseEmbeddings,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
	{
		"numerical_model": ValueTransformerEmbeddings,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
 	{
		"numerical_model": DICEEmbeddingAggregator,
  		"finetune_formatter": [textual_prompt_formatter],
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
]



experiment_configs = {
'numerical_datasets': {  
    "earthquakes": {
		"config": {
			"embitto": embitto_only_textual,
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	},  
	"vsx": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 "protein": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 	"vsx_combined": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
    },

'combined_1': {  
    "baby_products_combined": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	},  
	"book3_all": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 "books3_all_no_isbn": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 	"books3_combined": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
	"books3_combined_no_isbn": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
    },
	'combined_2': {  
    "x2_all": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	},  
	"x2_combined": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 "x3_all": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 	"x3_combined": {
		"config": {
			"embitto": embitto_only_textual,
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
   			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 40,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": True,
				},
				"test": {
					"batch_size": 32,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 50,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
	
    },

}
