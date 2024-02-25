from NumbER.matching_solutions.utils.sampler.similarity_based import SimilarityBasedSampler
from NumbER.matching_solutions.utils.sampler.sorted_neighbourhood import SortedNeighbourhoodSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddingAggregator
from NumbER.matching_solutions.embitto.numerical_components.value_embeddings import ValueTransformerEmbeddings, ValueBaseEmbeddings, ValueValueEmbeddings
from NumbER.matching_solutions.embitto.numerical_components.numeric_roberta import NumericRoberta
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter,textual_prompt_formatter, complete_prompt_formatter, ditto_formatter, numeric_prompt_formatter, pair_based_numeric_formatter, complete_prompt_formatter_min_max_scaled,complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific, text_sim_formatter, textual_scientific, textual_min_max_scaled, num_text_sim_formatter
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
							"finetune_formatter": textual_prompt_formatter
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
finetune_formatters = [
#     complete_prompt_formatter_scientific,
# pair_based_ditto_formatter_scientific,
# complete_prompt_formatter_min_max_scaled,
# textual_min_max_scaled
	# textual_prompt_formatter
	# text_sim_formatter,
	# complete_prompt_formatter_min_max_scaled,
	# textual_min_max_scaled,
 
    num_text_sim_formatter, 
    # textual_prompt_formatter,
    # pair_based_ditto_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific,
    # text_sim_formatter, complete_prompt_formatter_min_max_scaled,
    # textual_min_max_scaled, textual_scientific
    
    ]
combinations = [
 	#  {
	#  	"numerical_model": DICEEmbeddingAggregator,
  	#  	"finetune_formatter": [textual_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
 	#  {
	#  	"numerical_model": ValueBaseEmbeddings,
  	#  	"finetune_formatter": [textual_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	# {
	# 	"numerical_model": ValueTransformerEmbeddings,
  	# 	"finetune_formatter": [textual_prompt_formatter],
	# 	"include_numerical_features_in_textual": True,
	# 	"textual_embedding_size": 128,
	# },
   
	#   {
	#  	"numerical_model": ValueValueEmbeddings,
  	#  	"finetune_formatter": [textual_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	#  {
	#  	"numerical_model": DICEEmbeddingAggregator,
  	#  	"finetune_formatter": [complete_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	# {
	# 	"numerical_model": ValueTransformerEmbeddings,
  	# 	"finetune_formatter": [complete_prompt_formatter],
	# 	"include_numerical_features_in_textual": True,
	# 	"textual_embedding_size": 128,
	# },
 	#  {
	#  	"numerical_model": ValueBaseEmbeddings,
  	#  	"finetune_formatter": [complete_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	#   {
	#  	"numerical_model": ValueValueEmbeddings,
  	#  	"finetune_formatter": [complete_prompt_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
    # {
	#  	"numerical_model": DICEEmbeddingAggregator,
  	#  	"finetune_formatter": [pair_based_ditto_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	# {
	# 	"numerical_model": ValueTransformerEmbeddings,
  	# 	"finetune_formatter": [pair_based_ditto_formatter],
	# 	"include_numerical_features_in_textual": True,
	# 	"textual_embedding_size": 128,
	# },
 	#  {
	#  	"numerical_model": ValueBaseEmbeddings,
  	#  	"finetune_formatter": [pair_based_ditto_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
	#   {
	#  	"numerical_model": ValueValueEmbeddings,
  	#  	"finetune_formatter": [pair_based_ditto_formatter],
	#  	"include_numerical_features_in_textual": True,
	#  	"textual_embedding_size": 128,
	#  },
   {
	 	"numerical_model": None,
  	 	"finetune_formatter": finetune_formatters,
	 	"include_numerical_features_in_textual": True,
	 	"textual_embedding_size": 256,
	 },
   
#   {
# 		"numerical_model": ValueBaseEmbeddings,
#   		"finetune_formatter": [complete_prompt_formatter],
# 		"include_numerical_features_in_textual": False,
# 		"textual_embedding_size": 128,
# 	},
# 	{
# 		"numerical_model": ValueTransformerEmbeddings,
#   		"finetune_formatter": [complete_prompt_formatter],
# 		"include_numerical_features_in_textual": False,
# 		"textual_embedding_size": 128,
# 	},
#  	{
# 		"numerical_model": DICEEmbeddingAggregator,
#   		"finetune_formatter": [complete_prompt_formatter],
# 		"include_numerical_features_in_textual": False,
# 		"textual_embedding_size": 128,
# 	},
]



sets = {
	'fast': {  
 
	"x3_merged": {
		"config": {
			"embitto": {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": DICEEmbeddingAggregator,
							"pretrain_formatter": dummy_formatter,
       						"finetune_formatter": pair_based_numeric_formatter
						},
					"textual_config":
						{
							"model": BaseRoberta,
							"max_length": 256,
							"embedding_size": 128,
							"pretrain_formatter": ditto_formatter,
							"finetune_formatter": pair_based_ditto_formatter
						},
					"num_pretrain_epochs": 50,
					"pretrain_batch_size": 50,
					"finetune_batch_size": 50,
					"num_finetune_epochs": 2,#30
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True,
     				"include_numerical_features_in_textual": False
				},
				"test": {
					"cluster": False
				}
			},
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
				'epochs': 2,
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
	#"vsx": {
	#	"config": {
	#		"embitto": embitto_only_textual,
	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": "True",
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
   	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": True,
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
	#	"deep_matcher": {
	#		"train": {
	#			'epochs': 40,
    	#		'batch_size': 50,
    	#		'pos_neg_ratio':3
	#		},
	#		"test": {
	#		}
	#	}},'blocking':{
      	#	'sampler': SimilarityBasedSampler,
	#		'distance_path': 'similarity.npy',
	#		'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
	#		'train_window_size': 25,
	#		'test_window_size': 10,
      	#'attributes': ["depth", "mag_value"]}},
	#}, 
 #"protein": {
	#	"config": {
	#		"embitto": embitto_only_textual,
	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": "True",
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
   	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": True,
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
	#	"deep_matcher": {
	#		"train": {
	#			'epochs': 40,
    	#		'batch_size': 50,
    	#		'pos_neg_ratio':3
	#		},
	#		"test": {
	#		}
	#	}},'blocking':{
      	#	'sampler': SimilarityBasedSampler,
	#		'distance_path': 'similarity.npy',
	#		'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
	#		'train_window_size': 25,
	#		'test_window_size': 10,
      	#'attributes': ["depth", "mag_value"]}},
	#}, 
 	#"vsx_combined": {
	#	"config": {
	#		"embitto": embitto_only_textual,
	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": "True",
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
   	#		"ditto": {
	#			"train": {
	#				"batch_size": 50,
	#				"n_epochs": 40,
	#				"lr": 3e-5,
	#				"max_len": 256,
	#				"lm": "roberta",
	#				"fp16": True,
	#			},
	#			"test": {
	#				"batch_size": 32,
	#				"lm": "roberta",
	#				"max_len": 256,
	#			}
	#		},
	#	"deep_matcher": {
	#		"train": {
	#			'epochs': 40,
    	#		'batch_size': 50,
    	#		'pos_neg_ratio':3
	#		},
	#		"test": {
	#		}
	#	}},'blocking':{
      	#	'sampler': SimilarityBasedSampler,
	#		'distance_path': 'similarity.npy',
	#		'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
	#		'train_window_size': 25,
	#		'test_window_size': 10,
      	#'attributes': ["depth", "mag_value"]}},
	#}, 
    },

'numeric_1': {
 "books3_numeric": {
		"config": {
			"embitto": embitto_only_textual,
            "lightgbm": {
                "train": {
					"epochs": 10,
                    "params": {'num_leaves': 31, 'objective': 'binary', "metric": "auc"},
				},
				"test": {
                    
				}
			},
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
	"baby_products_numeric": {
		"config": {
            "lightgbm": {
                "train": {
					"epochs": 50,
                    "params": {'num_leaves': 31, 'objective': 'binary', "metric": "auc"},
				},
				"test": {
                    
				}
			},
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
    
    "x3_numeric": {
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
'numeric_2': {
    "x2_numeric": {
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
'numeric_3':{},
'merged_1': {
    "x3_merged": {
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
    "baby_products_merged": {
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
     "baby_products_all": {
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

'merged_2': {
    "x3_merged": {
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
#   "books3_all": {
# 		"config": {
# 			"embitto": embitto_only_textual,
# 			"ditto": {
# 				"train": {
# 					"batch_size": 50,
# 					"n_epochs": 40,
# 					"lr": 3e-5,
# 					"max_len": 256,
# 					"lm": "roberta",
# 					"fp16": "True",
# 				},
# 				"test": {
# 					"batch_size": 32,
# 					"lm": "roberta",
# 					"max_len": 256,
# 				}
# 			},
#    			"ditto": {
# 				"train": {
# 					"batch_size": 50,
# 					"n_epochs": 40,
# 					"lr": 3e-5,
# 					"max_len": 256,
# 					"lm": "roberta",
# 					"fp16": True,
# 				},
# 				"test": {
# 					"batch_size": 32,
# 					"lm": "roberta",
# 					"max_len": 256,
# 				}
# 			},
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 40,
#     			'batch_size': 50,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		}},'blocking':{
#       		'sampler': SimilarityBasedSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["depth", "mag_value"]}},
# 	},   
},
'merged_3': {
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
    "books3_merged": {
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
'merged_4': {
    "books3_merged_no_isbn": {
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
    },
	'books3_all_no_isbn':{
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
	}, },
'x2': {
    "x2_numeric": {
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
	"x2_merged": {
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
'numeric_4': {
    "x3_numeric": {
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
	"books3_all": {
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
 
    },
	'combined_3': {
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
	}, },
 'combined_4': {
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
 
 '2MASS_small': {
	 "2MASS_small_no_n": {
		"config": {
			"embitto": embitto_only_textual,
            "lightgbm": {
                "train": {
					"epochs": 50,
                    "params": {'num_leaves': 31, 'objective': 'binary', "metric": "auc"},
				},
				"test": {
                    
				}
			},
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 },
  'books3_merged_new': {
	 "books3_merged_new": {
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
  
 },
  'books3_merged_new_no_isbn': {
	  "books3_merged_new_no_isbn": {
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
  },
  'baby_products_merged_new': {
	  "baby_products_merged_new": {
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
  },
'earthquakes': {
	 "earthquakes": {
		"config": {
			"embitto": embitto_only_textual,
            "lightgbm": {
                "train": {
					"epochs": 50,
                    "params": {'num_leaves': 31, 'objective': 'binary', "metric": "auc"},
				},
				"test": {
                    
				}
			},
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
'protein_small': {
	 "protein_small": {
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
			'config': {'n_most_similar':6, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	}, 
 },
'vsx_small': {
	 "vsx_small": {
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
		}, 
},
'deep_matcher': {
    "books3_numeric_no_isbn": {
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
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
			'config': {'n_most_similar':5, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
		}, 
	
},
  #"vsx_small_numeric": {
'x3_merged': {
	 "x3_merged": {
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
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 25,
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
} ,
'x3_all': {
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
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 25,
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
'x3_numeric': {
	 "x3_numeric": {
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
} ,
'books3_merged_no_isbn': {
	 "books3_merged_no_isbn": {
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
		"deep_matcher": {
			"train": {
				'epochs': 40,
    			'batch_size': 40,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		}},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity_cosine.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
	},
} 
}

#		"config": {
#			"embitto": embitto_only_textual,
#			"ditto": {
#				"train": {
#					"batch_size": 50,
#					"n_epochs": 40,
#					"lr": 3e-5,
#					"max_len": 256,
#					"lm": "roberta",
#					"fp16": "True",
#				},
#				"test": {
#					"batch_size": 32,
#					"lm": "roberta",
#					"max_len": 256,
#				}
#			},
 #  			"ditto": {
#				"train": {
#					"batch_size": 50,
#					"n_epochs": 40,
#					"lr": 3e-5,
#					"max_len": 256,
#					"lm": "roberta",
#					"fp16": True,
#				},
#				"test": {
#					"batch_size": 32,
#					"lm": "roberta",
#					"max_len": 256,
#				}
#			},
#		"deep_matcher": {
#			"train": {
#				'epochs': 40,
 #   			'batch_size': 50,
  #  			'pos_neg_ratio':3
#			},
#			"test": {
#			}
#		}},'blocking':{
 #     		'sampler': SimilarityBasedSampler,
#			'distance_path': 'similarity.npy',
#			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
#			'train_window_size': 25,
#			'test_window_size': 10,
 #     	'attributes': ["depth", "mag_value"]}},
#	}, 
experiment_configs = {
	'fast': sets['fast'],
	'numerical_datasets': sets['numerical_datasets'],
	'numeric_1': {**sets['2MASS_small'], **sets['earthquakes'], **sets['numeric_1']},
	'numeric_2': {**sets['numeric_2'], **sets['vsx_small'], **sets['protein_small']},
	'numeric_3': sets['numeric_3'],
	'numeric_4': sets['numeric_4'],
	'combined_1': sets['combined_1'],
	'combined_2': sets['combined_2'],
	'combined_3': sets['combined_3'],
	'combined_4': sets['combined_4'],
	'all_1': {**sets['numeric_1'], **sets['merged_1'], **sets['2MASS_small']},#neu ausführen
 	'all_2': {**sets['numeric_2'], **sets['merged_2']}, # **sets['earthquakes']},#neu ausführen
 	'all_3': {**sets['numeric_3'], **sets['merged_3']}, # **sets['protein_small']},
 	'all_4': {**sets['numeric_4'], **sets['merged_4'], **sets['vsx_small']},
	'all_all_1': {**sets['numeric_1'], **sets['numeric_2'], **sets['combined_3'], **sets['combined_4'], **sets['2MASS_small'], **sets['earthquakes'], },
	'all_all_2': { **sets['numeric_3'], **sets['numeric_4'], **sets['combined_1'], **sets['combined_2'], **sets['protein_small'], **sets['vsx_small']},
    '2MASS_small': sets['2MASS_small'],
     'earthquakes': sets['earthquakes'],
     'protein_small': sets['protein_small'],
     'vsx_small': sets['vsx_small'],
    'similarity_tests': {**sets['numeric_1'], **sets['numeric_4'], **sets['earthquakes']},
    'deep_matcher': sets['deep_matcher'],
    'merged_1': sets['merged_1'],
	'merged_2': sets['merged_2'],
	'books3_merged_no_isbn': sets['books3_merged_no_isbn'],
	'x3_all': sets['x3_all'],
	'x3_merged': sets['x3_merged'],
	'x3_numeric': sets['x3_numeric'],
	'remaining_ditto': {**sets['x3_all'], **sets['x3_merged'], **sets['x2']},
	'x3': sets['x3_merged'],# {**sets['x3_all'], **sets['x3_merged']},#, **sets['x3_numeric']},
	'books3_merged_new': sets['books3_merged_new'],
	'books3_merged_new_no_isbn': sets['books3_merged_new_no_isbn'],
	'baby_products_merged_new': sets['baby_products_merged_new'],
 'merged_new': {**sets['books3_merged_new'], **sets['books3_merged_new_no_isbn'], **sets['baby_products_merged_new']},
	'books3_all_no_isbn': sets['books3_all_no_isbn'],
}
