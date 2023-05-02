experiment_configs = {
'test': {
	'baby_products_numeric': {
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
			},
			'blocking_attributes': ["width", "height"],
  }
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
		}},
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
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
		}
	},
 "2MASS": {
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
		}
	}
}}