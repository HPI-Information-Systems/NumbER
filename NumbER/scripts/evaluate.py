import pandas as pd
#from NumbER.utils.experiment_config import experiment_configs, combinations
from NumbER.matching_solutions.embitto.numerical_components.dice import DICEEmbeddingAggregator
from NumbER.matching_solutions.embitto.numerical_components.value_embeddings import ValueTransformerEmbeddings, ValueBaseEmbeddings
from NumbER.matching_solutions.embitto.numerical_components.numeric_roberta import NumericRoberta
from NumbER.matching_solutions.embitto.textual_components.base_roberta import BaseRoberta
from NumbER.matching_solutions.embitto.formatters import dummy_formatter, pair_based_ditto_formatter,textual_prompt_formatter, complete_prompt_formatter, ditto_formatter, numeric_prompt_formatter, pair_based_numeric_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific

#print(aggregate)
#aggregate.to_csv("summary.csv")
data = pd.read_csv("./NumbER/scripts/runs.csv")
finetune_formatters = [pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific]
combinations = [
	{
		"numerical_model": None,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 256,
	},
 	{
		"numerical_model": ValueBaseEmbeddings,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
	{
		"numerical_model": ValueTransformerEmbeddings,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
 	{
		"numerical_model": DICEEmbeddingAggregator,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": True,
		"textual_embedding_size": 128,
	},
  {
		"numerical_model": ValueBaseEmbeddings,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
	{
		"numerical_model": ValueTransformerEmbeddings,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
 	{
		"numerical_model": DICEEmbeddingAggregator,
  		"finetune_formatter": finetune_formatters,
		"include_numerical_features_in_textual": False,
		"textual_embedding_size": 128,
	},
]
for combination in combinations:
    df = data.copy()
    df = df[df["matching_solution"] == "embitto"]
    numerical_model = combination["numerical_model"].__module__ + "." + combination["numerical_model"].__name__ if combination["numerical_model"] is not None else "None"
    if numerical_model == "None":
        df = df[df["numerical_config_model"].isnull()]
    else:
        df = df[df["numerical_config_model"] == numerical_model]
    for finetune_formatter in combination["finetune_formatter"]:
        df_filtered = df[((df["textual_config_finetune_formatter"] == finetune_formatter.__module__ + "." + finetune_formatter.__name__)
                #& (df["include_numerical_features_in_textual"] == combination["include_numerical_features_in_textual"])
                & (df["textual_config_embedding_size"] == combination["textual_embedding_size"])
                )]
        if len(df_filtered) == 0:
            continue
        aggregate = df_filtered.groupby(["dataset","should_finetune","should_pretrain", "tags"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"]})
        aggregate_2 = df_filtered.groupby(["dataset", "tags", "should_finetune","should_pretrain","finetune_batch_size","num_finetune_epochs","num_pretrain_epochs","pretrain_batch_size","output_embedding_size","include_numerical_features_in_textual","lm","fp16","max_len","n_epochs","batch_size","numerical_config_model","numerical_config_embedding_size","numerical_config_finetune_formatter","numerical_config_pretrain_formatter","numerical_config_0","textual_config_model","textual_config_max_length","textual_config_embedding_size","textual_config_finetune_formatter","textual_config_pretrain_formatter","textual_config_0"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"]})
        print("===========================")
        print("Finetune formatter: ", finetune_formatter.__name__)
        print("Numerical model: ", numerical_model)
        print("Include numerical features in textual: ", combination["include_numerical_features_in_textual"])
        print("Textual embedding size: ", combination["textual_embedding_size"])
        print(aggregate_2)
        #df_filtered.to_csv("BLA.csv")
        #break
    #break

df = df[df["matching_solution"] == "ditto"]
#df = df[df["epochs"] == 40]
