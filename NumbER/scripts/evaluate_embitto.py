import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
	#'numerical_config_model': None,
	#'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.value_embeddings.ValueTransformerEmbeddings",
	#'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.dice.DICEEmbeddingAggregator",
 #'numerical_config_embedding_size': 128,
	# 'numerical_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.pair_based_numeric_formatter',
	# 'numerical_config_pretrain_formatter': 'NumbER.matching_solutions.embitto.formatters.dummy_formatter',
	# 'numerical_config_0': None,
	'textual_config_model': 'NumbER.matching_solutions.embitto.textual_components.base_roberta.BaseRoberta',
	'textual_config_max_length': 256,
	'textual_config_embedding_size': 256,
	'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.complete_prompt_formatter_min_max_scaled',#_scientific',
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
filtered_df = filtered_df[filtered_df["dataset"] != "vsx"]
filtered_df = filtered_df[filtered_df["dataset"] != "vsx_small_numeric"]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["books3_numeric_no_isbn","x2_numeric","x3_numeric","earthquakes","vsx_small","2MASS_small"])]

#filtered_df = filtered_df["similar_sampling" in filtered_df["tags"]]
#print(len(filtered_df))
# filtered_df = filtered_df[filtered_df["dataset"]== "books3_numeric"]
# filtered_df = filtered_df[filtered_df["tags"] == "final_numeric_sorted"]
#print(filtered_df[["i", "f1_not_closed", "training_time"]])
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
#print(aggregate)
aggregate = aggregate[aggregate["f1_not_closed"]["count"] > 4]
print(aggregate['f1_not_closed']['mean'])

#aggregate = filtered_df[(filtered_df['dataset'].str.contains("numeric")) | (filtered_df['dataset'].str.contains("earthquake"))].groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
aggregate["f1_not_closed"]["mean"].to_csv("result.csv")
data = aggregate['f1_not_closed']
print(len(data))
group_mean = data.mean()['mean']
print(group_mean)
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.boxplot(x='mean', data=aggregate['f1_not_closed'])

# Add labels and title
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Boxplot Example')

# Save the plot as an image (choose the desired format, e.g., PNG, PDF, etc.)
plt.savefig('boxplot.png')
# mask = df.isin(filter).all(axis=1)
# print(df[mask])
# #group_by: i, dataset, tags, state
