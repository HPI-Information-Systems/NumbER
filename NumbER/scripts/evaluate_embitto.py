import pandas as pd

df = pd.read_csv("./NumbER/scripts/runs.csv")
filter= {
	'lr': 3e-5,
 	'should_finetune': True,
	'should_pretrain': False,
	'matching_solution': 'embitto',
	'finetune_batch_size': 50,
 	'num_finetune_epochs': 40,
	#'num_pretrain_epochs': ??,
	#'pretrain_batch_size': 50,
	'output_embedding_size': 256,
	#'include_numerical_features_in_textual': ??,
	#'lm': 'roberta',
	#'fp16': True,
	#'max_len': 256,
	'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.value_embeddings.ValueTransformerEmbeddings",
	#'numerical_config_embedding_size': 128,
	# 'numerical_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.pair_based_numeric_formatter',
	# 'numerical_config_pretrain_formatter': 'NumbER.matching_solutions.embitto.formatters.dummy_formatter',
	# 'numerical_config_0': None,
	'textual_config_model': 'NumbER.matching_solutions.embitto.textual_components.base_roberta.BaseRoberta',
	'textual_config_max_length': 256,
	'textual_config_embedding_size': 128,
	'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.complete_prompt_formatter',#_scientific',
	#'textual_config_pretrain_formatter': ??,
	'textual_config_0': None, 
}
filtered_df = df

for key, value in filter.items():
    if value is not None:
        #print(key, value)
        filtered_df = filtered_df[filtered_df[key] == value]
        #print(len(filtered_df))
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
print(len(filtered_df))
filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
filtered_df = filtered_df[~filtered_df["run"].isnull()]
print(len(filtered_df))
# filtered_df = filtered_df[filtered_df["dataset"]== "books3_numeric"]
# filtered_df = filtered_df[filtered_df["tags"] == "final_numeric_sorted"]
#print(filtered_df[["i", "f1_not_closed", "training_time"]])
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
print(aggregate)
# mask = df.isin(filter).all(axis=1)
# print(df[mask])
# #group_by: i, dataset, tags, state
