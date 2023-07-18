from NumbER.matching_solutions.utils.sampler.similarity_based import SimilarityBasedSampler
from NumbER.matching_solutions.utils.sampler.sorted_neighbourhood import SortedNeighbourhoodSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddingAggregator
from NumbER.matching_solutions.embitto.numerical_components.value_embeddings import ValueTransformerEmbeddings, ValueBaseEmbeddings
from NumbER.matching_solutions.embitto.numerical_components.numeric_roberta import NumericRoberta
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter,textual_prompt_formatter, complete_prompt_formatter, ditto_formatter, numeric_prompt_formatter, pair_based_numeric_formatter, complete_prompt_formatter_min_max_scaled,complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific

#from NumbER.matching_solutions.utils.sampler.jedai_based import JedaiBasedSampler
collections = {
	'numeric': ['vsx', 'earthquakes', 'books3_numeric', 'books3_numeric_no_isbn', 'x2_numeric', 'x3_numeric']
}
embitto_only_textual = {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": ValueBaseEmbeddings,
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
'fast': {
	"earthquakes": {
		"config": {
			"embitto": {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": ValueBaseEmbeddings,
							"pretrain_formatter": dummy_formatter,
       						"finetune_formatter": pair_based_numeric_formatter
						},
					"textual_config":
						{
							"model": BaseRoberta,
							"max_length": 256,
							"embedding_size": 128,
							"pretrain_formatter": ditto_formatter,
							"finetune_formatter": complete_prompt_formatter_min_max_scaled
						},
					"include_numerical_features_in_textual": False,
					"num_pretrain_epochs": 50,
					"pretrain_batch_size": 100,
					"finetune_batch_size": 50,
					"num_finetune_epochs": 40,#30
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True
				},
				"test": {
					"cluster": False
				}
			},
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 30,
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
		"xgboost": {
			"train": {"early_stopping_rounds": 100},
			"test": {}
		},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'config': {'n_most_similar':3, 'test_window_size': 10,'train_window_size': 25,'train_fraction': 0.6, 'valid_fraction': 0.2, 'test_fraction': 0.2,
      		'attributes': ["Price"]}}
	}
},
'fast_2': {
	"earthquakes": {
		"config": {
			"embitto": {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": ValueBaseEmbeddings,
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
					"num_finetune_epochs": 30,
					"output_embedding_size": 256,
					"include_numerical_features_in_textual": True,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True
				},
				"test": {
					"cluster": False
				}
			},
			"ditto": {
				"train": {
					"batch_size": 32,
					"n_epochs": 30,
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
		"xgboost": {
			"train": {"early_stopping_rounds": 100},
			"test": {}
		},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'config': {'n_most_similar':3, 'test_window_size': 10,'train_window_size': 25,'train_fraction': 0.6, 'valid_fraction': 0.2, 'test_fraction': 0.2,
      		'attributes': ["Price"]}}
	}
},
'fast_3': {
	"earthquakes": {
		"config": {
			"embitto": {
				"train": {
					"numerical_config":
						{
							"embedding_size": 128,
							"model": ValueBaseEmbeddings,
							"pretrain_formatter": dummy_formatter,
       						"finetune_formatter": complete_prompt_formatter
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
					"num_finetune_epochs": 30,
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True
				},
				"test": {
					"cluster": False
				}
			},
			"ditto": {
				"train": {
					"batch_size": 32,
					"n_epochs": 30,
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
		"xgboost": {
			"train": {"early_stopping_rounds": 100},
			"test": {}
		},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'config': {'n_most_similar':3, 'test_window_size': 10,'train_window_size': 25,'train_fraction': 0.6, 'valid_fraction': 0.2, 'test_fraction': 0.2,
      		'attributes': ["Price"]}}
	}
},
'fast_4': {
	"earthquakes": {
		"config": {
			"embitto": {
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
							"embedding_size": 256,
							"pretrain_formatter": ditto_formatter,
							"finetune_formatter": pair_based_ditto_formatter
						},
					"num_pretrain_epochs": 50,
					"pretrain_batch_size": 50,
					"finetune_batch_size": 50,
					"num_finetune_epochs": 30,#30
					"output_embedding_size": 256,
					"lr": 3e-5,
					"should_pretrain": False,
					"should_finetune": True
				},
				"test": {
					"cluster": False
				}
			},
			"ditto": {
				"train": {
					"batch_size": 50,
					"n_epochs": 30,
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
		"xgboost": {
			"train": {"early_stopping_rounds": 100},
			"test": {}
		},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'config': {'n_most_similar':3, 'test_window_size': 10,'train_window_size': 25,'train_fraction': 0.6, 'valid_fraction': 0.2, 'test_fraction': 0.2,
      		'attributes': ["Price"]}}
	}
},
'test': {
	'earthquakes': {
		"config": {
				"ditto": {
					"train": {
						"batch_size": 10,
						"n_epochs": 10,
						"lr": 3e-5,
						"max_len": 256,
						"lm": "roberta",
						"fp16": "True",
					},
					"test": {
						"batch_size": 16,
						"lm": "roberta",
						"max_len": 256,
					}
				},
			"md2m": {
				"train": {
				},
				"test": {
				}
			},
			"deep_matcher": {
				"train": {
					'epochs': 1,
					'batch_size':16,
					'pos_neg_ratio':3
				},
				"test": {
				}
			},
			},
			'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}}
  }
},
'test_2': {
	'earthquakes': {
		"config": {
				"ditto": {
					"train": {
						"batch_size": 10,
						"n_epochs": 10,
						"lr": 3e-5,
						"max_len": 256,
						"lm": "roberta",
						"fp16": "True",
					},
					"test": {
						"batch_size": 16,
						"lm": "roberta",
						"max_len": 256,
					}
				},
			"md2m": {
				"train": {
				},
				"test": {
				}
			},
			"deep_matcher": {
				"train": {
					'epochs': 1,
					'batch_size':16,
					'pos_neg_ratio':3
				},
				"test": {
				}
			},
			},
			'blocking':{
      		'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}}
  }
},
'test_3': {
	'earthquakes': {
		"config": {
				"ditto": {
					"train": {
						"batch_size": 10,
						"n_epochs": 10,
						"lr": 3e-5,
						"max_len": 256,
						"lm": "roberta",
						"fp16": "True",
					},
					"test": {
						"batch_size": 16,
						"lm": "roberta",
						"max_len": 256,
					}
				},
			"md2m": {
				"train": {
				},
				"test": {
				}
			},
			"deep_matcher": {
				"train": {
					'epochs': 1,
					'batch_size':16,
					'pos_neg_ratio':3
				},
				"test": {
				}
			},
			},
			'blocking':{
      		'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}}
  }
},
'numeric': {  
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
 },
	"books3_numeric": {
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["ISBN13"]}}
	},
 "baby_products_numeric": {
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["width","height","length","weight_lb","weight_oz","price"]}}
	},
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},
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
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 6,
      	'attributes': ["storage_refine"]}}
	},
# 	'vsx': {
# 		"config": {
#       "embitto": embitto_only_textual,
# 				"ditto": {
# 					"train": {
# 						"batch_size": 50,
# 						"n_epochs": 40,
# 						"lr": 3e-5,
# 						"max_len": 256,
# 						"lm": "roberta",
# 						"fp16": "True",
# 					},
# 					"test": {
# 						"batch_size": 16,
# 						"lm": "roberta",
# 						"max_len": 256,
# 					}
# 				},
# 			"deep_matcher": {
# 				"train": {
# 					'epochs': 10,
# 					'batch_size':16,
# 					'pos_neg_ratio':3
# 				},
# 				"test": {
# 				}
# 			},
# 			},
# 			'blocking':{
# 				'sampler': SimilarityBasedSampler,
# 				'distance_path': 'similarity.npy',
# 				'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 				'train_window_size': 25,
# 				'test_window_size': 10,
#       			'attributes': ["RAdeg","DEdeg"]
#          	}}
#   },
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 9,
      	'attributes': ["storage_refine"]}}
	},
#   	"protein": {
# 		"config": {
# 			"ditto": {
# 				"train": {
# 					"batch_size": 10,
# 					"n_epochs": 3,
# 					"lr": 3e-5,
# 					"max_len": 256,
# 					"lm": "roberta",
# 					"fp16": "True",
# 				},
# 				"test": {
# 					"batch_size": 16,
# 					"lm": "roberta",
# 					"max_len": 256,
# 				}
# 			},
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 5,
#     			'batch_size':16,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		},
# 		},'blocking':{
#       		'sampler': SimilarityBasedSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
#  },
},
'numeric_sorted': { 
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
				'epochs': 20,
    			'batch_size':8,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 9,
      	'attributes': ["storage_refine"]}}
	},
	"books3_numeric": {
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["ISBN13"]}}
	},
 "baby_products_numeric": {
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["width","height","length","weight_lb","weight_oz","price"]}}
	},
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},
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
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 6,
      	'attributes': ["storage_refine"]}}
	},
	
#   "earthquakes": {
# 		"config": {
# 			"ditto": {
# 				"train": {
# 					"batch_size": 32,
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
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 10,
#     			'batch_size':10,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		},
# 		},'blocking':{
#       		'sampler': SortedNeighbourhoodSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["depth", "mag_value"]}},
#  },
#    "protein": {
# 		"config": {
# 			"ditto": {
# 				"train": {
# 					"batch_size": 32,
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
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 10,
#     			'batch_size':16,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		},
# 		},'blocking':{
#       		'sampler': SortedNeighbourhoodSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
#  },
# 	'vsx': {
# 		"config": {
# 				"ditto": {
# 					"train": {
# 						"batch_size": 32,
# 						"n_epochs": 40,
# 						"lr": 3e-5,
# 						"max_len": 256,
# 						"lm": "roberta",
# 						"fp16": "True",
# 					},
# 					"test": {
# 						"batch_size": 32,
# 						"lm": "roberta",
# 						"max_len": 256,
# 					}
# 				},
# 			"deep_matcher": {
# 				"train": {
# 					'epochs': 7,
# 					'batch_size':8,
# 					'pos_neg_ratio':3
# 				},
# 				"test": {
# 				}
# 			},
# 			},
# 			'blocking':{
# 				'sampler': SortedNeighbourhoodSampler,
# 				'distance_path': 'similarity.npy',
# 				'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 				'train_window_size': 25,
# 				'test_window_size': 10,
#       			'attributes': ["RAdeg","DEdeg"]
#          	}}
#   },
#   "2MASS": {
# 		"config": {
# 			"ditto": {
# 				"train": {
# 					"batch_size": 32,
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
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 10,
#     			'batch_size':16,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		},
# 		},'blocking':{
# 			'sampler': SortedNeighbourhoodSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["glat", "glon", "j_m", "h_m", "k_m"]}}
# 	},
},

'numeric_naive': { 
     
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
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
      		'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
 },
	"books3_numeric": {
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
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["ISBN13"]}}
	},
 "baby_products_numeric": {
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
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["width","height","length","weight_lb","weight_oz","price"]}}
	},
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
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},
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
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 6,
      	'attributes': ["storage_refine"]}}
	},
# 	'vsx': {
# 		"config": {
# 				"ditto": {
# 					"train": {
# 						"batch_size": 32,
# 						"n_epochs": 40,
# 						"lr": 3e-5,
# 						"max_len": 256,
# 						"lm": "roberta",
# 						"fp16": "True",
# 					},
# 					"test": {
# 						"batch_size": 16,
# 						"lm": "roberta",
# 						"max_len": 256,
# 					}
# 				},
# 			"deep_matcher": {
# 				"train": {
# 					'epochs': 5,
# 					'batch_size':16,
# 					'pos_neg_ratio':3
# 				},
# 				"test": {
# 				}
# 			},
# 			},
# 			'blocking':{
# 				'sampler': NaiveSampler,
# 				'distance_path': 'similarity.npy',
# 				'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 				'train_window_size': 25,
# 				'test_window_size': 10,
#       			'attributes': ["RAdeg","DEdeg"]
#          	}}
#   },
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
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 20,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': NaiveSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 9,
      	'attributes': ["storage_refine"]}}
	},
#   "protein": {
# 		"config": {
# 			"ditto": {
# 				"train": {
# 					"batch_size": 32,
# 					"n_epochs": 10,
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
# 		"deep_matcher": {
# 			"train": {
# 				'epochs': 5,
#     			'batch_size':16,
#     			'pos_neg_ratio':3
# 			},
# 			"test": {
# 			}
# 		},
# 		},'blocking':{
#       		'sampler': NaiveSampler,
# 			'distance_path': 'similarity.npy',
# 			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
# 			'train_window_size': 25,
# 			'test_window_size': 10,
#       	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
#  },
	# "2MASS": {
	# 	"config": {
	# 		"embitto": embitto_only_textual,
	# 		"ditto": {
	# 			"train": {
	# 				"batch_size": 32,
	# 				"n_epochs": 40,
	# 				"lr": 3e-5,
	# 				"max_len": 256,
	# 				"lm": "roberta",
	# 				"fp16": "True",
	# 			},
	# 			"test": {
	# 				"batch_size": 32,
	# 				"lm": "roberta",
	# 				"max_len": 256,
	# 			}
	# 		},
	# 	"deep_matcher": {
	# 		"train": {
	# 			'epochs': 10,
    # 			'batch_size':16,
    # 			'pos_neg_ratio':3
	# 		},
	# 		"test": {
	# 		}
	# 	},
	# 	},'blocking':{
	# 		'sampler': SimilarityBasedSampler,
	# 		'distance_path': 'similarity.npy',
	# 		'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
	# 		'train_window_size': 25,
	# 		'test_window_size': 10,
    #   	'attributes': ["glat", "glon", "j_m", "h_m", "k_m"]}}
	# },
},

'experiment_config_1': {
 "earthquakes": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["depth", "mag_value"]}},
 },
 "baby_products_all": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["price"]}}
	},"baby_products_numeric": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
            'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["price"]}}
	},
 "books3_all": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 8,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["ISBN13"]}}
	},
 "books3_numeric": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["ISBN13"]}}
	},
 "books3_all_no_isbn": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
      		'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},
 "books3_numeric_no_isbn": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},

 "books4_all": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	},
 "books4_numeric": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 15,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["Price"]}}
	}
},

'experiment_config_2': {
 
 "x2_all": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 10,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 6,
      	'attributes': ["brand"]}}
	},
 "x2_numeric": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 10,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		},'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 6,
      	'attributes': ["storage_refine"]}}
	},
 "x3_all": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 10,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["brand"]}}
	},
 "x3_numeric": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 10,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 9,
      	'attributes': ["storage_refine"]}}
	},
 "vsx": {
		"config": {
			"ditto": {
				"train": {
					"batch_size": 10,
					"n_epochs": 10,
					"lr": 3e-5,
					"max_len": 256,
					"lm": "roberta",
					"fp16": "True",
				},
				"test": {
					"batch_size": 16,
					"lm": "roberta",
					"max_len": 256,
				}
			},
		"deep_matcher": {
			"train": {
				'epochs': 10,
    			'batch_size':16,
    			'pos_neg_ratio':3
			},
			"test": {
			}
		},
		}, 'blocking':{
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["RAdeg","DEdeg"]}}
	},
	# "2MASS": {
	# 	"config": {
	# 		"ditto": {
	# 			"train": {
	# 				"batch_size": 10,
	# 				"n_epochs": 10,
	# 				"lr": 3e-5,
	# 				"max_len": 256,
	# 				"lm": "roberta",
	# 				"fp16": "True",
	# 			},
	# 			"test": {
	# 				"batch_size": 16,
	# 				"lm": "roberta",
	# 				"max_len": 256,
	# 			}
	# 		},
	# 	"deep_matcher": {
	# 		"train": {
	# 			'epochs': 10,
    # 			'batch_size':16,
    # 			'pos_neg_ratio':3
	# 		},
	# 		"test": {
	# 		}
	# 	},
	# 	},'blocking':{
	# 		'sampler': SimilarityBasedSampler,
	# 		'distance_path': 'similarity.npy',
	# 		'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
	# 		'train_window_size': 25,
	# 		'test_window_size': 10,
    #   	'attributes': ["glat", "glon", "j_m", "h_m", "k_m"]}}
	# }
}}
