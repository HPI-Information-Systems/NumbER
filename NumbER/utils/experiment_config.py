from NumbER.matching_solutions.utils.sampler.similarity_based import SimilarityBasedSampler
from NumbER.matching_solutions.utils.sampler.sorted_neighbourhood import SortedNeighbourhoodSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
#from NumbER.matching_solutions.utils.sampler.jedai_based import JedaiBasedSampler
collections = {
	'numeric': ['vsx', 'earthquakes', 'books3_numeric', 'books3_numeric_no_isbn', 'x2_numeric', 'x3_numeric']
}
experiment_configs = {
'fast': {
	"x3_numeric": {
		"config": {
			"embitto": {
				"train": {
				},
				"test": {
					"cluster": False
				}
			},
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
					"n_epochs": 20,
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
						"run_id": 1,
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
						"run_id": 1,
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
						"run_id": 1,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
	'vsx': {
		"config": {
				"ditto": {
					"train": {
						"run_id": 1,
						"batch_size": 32,
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
					'epochs': 10,
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
      			'attributes': ["RAdeg","DEdeg"]
         	}}
  },
	"x3_numeric": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
  	"protein": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 10,
					"n_epochs": 3,
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
				'epochs': 5,
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
      	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
 },
},
'numeric_sorted': { 
	
	"x3_numeric": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
	
  "earthquakes": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
    			'batch_size':10,
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
      	'attributes': ["depth", "mag_value"]}},
 },
   "protein": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
      		'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
 },
	'vsx': {
		"config": {
				"ditto": {
					"train": {
						"run_id": 1,
						"batch_size": 32,
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
					'epochs': 7,
					'batch_size':8,
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
      			'attributes': ["RAdeg","DEdeg"]
         	}}
  },
  "2MASS": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			'sampler': SortedNeighbourhoodSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["glat", "glon", "j_m", "h_m", "k_m"]}}
	},
},

'numeric_naive': {  
    "earthquakes": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
	'vsx': {
		"config": {
				"ditto": {
					"train": {
						"run_id": 1,
						"batch_size": 32,
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
					'epochs': 5,
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
      			'attributes': ["RAdeg","DEdeg"]
         	}}
  },
	"x3_numeric": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 10,
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
  "protein": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
					"n_epochs": 10,
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
				'epochs': 5,
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
      	'attributes': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol"]}}#,"ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]}},
 },
	"2MASS": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
					"batch_size": 32,
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
			'sampler': SimilarityBasedSampler,
			'distance_path': 'similarity.npy',
			'config': {'n_most_similar':3, 'train_fraction': 0.5, 'valid_fraction': 0.25, 'test_fraction': 0.25,
			'train_window_size': 25,
			'test_window_size': 10,
      	'attributes': ["glat", "glon", "j_m", "h_m", "k_m"]}}
	},
},

'experiment_config_1': {
 "earthquakes": {
		"config": {
			"ditto": {
				"train": {
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
					"run_id": 1,
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
	# 				"run_id": 1,
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