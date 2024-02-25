import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("./NumbER/scripts/runs.csv")
 # pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific,
    # text_sim_formatter, complete_prompt_formatter_min_max_scaled
    # textual_min_max_scaled, textual_scientific
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
	#'include_numerical_features_in_textual': True,
	#'lm': 'roberta',
	#'fp16': True,
	#'max_len': 256,
	'numerical_config_model': None,
	#'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.value_embeddings.ValueBaseEmbeddings",
	#'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.dice.DICEEmbeddingAggregator",
 #'numerical_config_embedding_size': 128,
	# 'numerical_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.pair_based_numeric_formatter',
	# 'numerical_config_pretrain_formatter': 'NumbER.matching_solutions.embitto.formatters.dummy_formatter',
	# 'numerical_config_0': None,
	'textual_config_model': 'NumbER.matching_solutions.embitto.textual_components.base_roberta.BaseRoberta',
	'textual_config_max_length': 256,
	'textual_config_embedding_size': 256,
	'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.textual_prompt_formatter',#_scientific',
	#'textual_config_pretrain_formatter': ??,
	'textual_config_0': None, 
}

 # pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific,
    # text_sim_formatter, complete_prompt_formatter_min_max_scaled
    # textual_min_max_scaled, textual_scientific
filtered_df = df

for key, value in filter.items():
    if value is not None:
        filtered_df = filtered_df[filtered_df[key] == value]
        #print(len(filtered_df))
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
#print(len(filtered_df))
filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
filtered_df = filtered_df[~filtered_df["run"].isnull()]
filtered_df = filtered_df[filtered_df["dataset"].isin(["x2_all", "x2_combined", "x3_all", "x3_combined", "books3_all", "books3_combined", "books3_all_no_isbn", "books3_combined_no_isbn"])]

aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
#print(aggregate)
aggregate = aggregate[aggregate["f1_not_closed"]["count"] > 4]

# data = {
# 	'X2': (aggregate[aggregate['dataset'] == "x2_all"]["f1_not_closed"]["mean"].values[0], aggregate[aggregate['dataset'] == "x2_combined"]["f1_not_closed"]["mean"].values[0]),
# 	'X3': (aggregate[aggregate['dataset'] == "x3_all"]["f1_not_closed"]["mean"].values[0], aggregate[aggregate['dataset'] == "x3_combined"]["f1_not_closed"]["mean"].values[0]),
# 	'Books3': (aggregate[aggregate['dataset'] == "books3_all"]["f1_not_closed"]["mean"].values[0], aggregate[aggregate['dataset'] == "books3_combined"]["f1_not_closed"]["mean"].values[0]),
# 	'Books3_no_isbn': (aggregate[aggregate['dataset'] == "books3_all_no_isbn"]["f1_not_closed"]["mean"].values[0], aggregate[aggregate['dataset'] == "books3_combined_no_isbn"]["f1_not_closed"]["mean"].values[0]),
# }
data = {
	'combined': (
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x2_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x3_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_combined_no_isbn']["f1_not_closed"]["mean"].values[0],2),
	),
 	'all': (
		0.92,
		0.89,
		1,
		0.99,
	)
}
species = ("X2", "X3", "Books3", "Books3_no_isbn")

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in data.items():
    #print(round(measurement, 2))
    
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1
