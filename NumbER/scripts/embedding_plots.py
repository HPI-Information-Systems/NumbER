import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
	'textual_config_embedding_size': 128,
	'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.textual_prompt_formatter',#_scientific',
	#'textual_config_pretrain_formatter': ??,
	'textual_config_0': None, 
}


# Create the 3x3 grid of boxplots
plt.figure(figsize=(6, 12))  # Adjust the figure size if needed

# Iterate through each unique group and create a boxplot for each one
for i, group in enumerate([
    "dice.DICEEmbeddingAggregator", "value_embeddings.ValueTransformerEmbeddings", "value_embeddings.ValueBaseEmbeddings"
]):
    filtered_df = df
    filter["numerical_config_model"] = f"NumbER.matching_solutions.embitto.numerical_components.{group}"
    for key, value in filter.items():
        if value is not None:
            filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = filtered_df[filtered_df[key].isnull()]
    filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
    filtered_df = filtered_df[~filtered_df["run"].isnull()]
    #filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_all","books3_numeric","books3_numeric_no_isbn","earthquakes","vsx_small","x2_all","x2_combined","x3_all","x3_combined","x3_numeric" ])]
    filtered_df = filtered_df[filtered_df["dataset"].isin(["books3_numeric_no_isbn","x2_numeric","x3_numeric","earthquakes","vsx_small","2MASS_small"])]
    aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
    aggregate = aggregate[aggregate["f1_not_closed"]["count"] > 4]
    mean = aggregate['f1_not_closed']['mean'].mean()
    print("Group: ", group)
    print(aggregate)
    plt.subplot(3, 1, i + 1)
    plt.xlim(left=0.2, right=1.0)
    sns.boxplot(x='mean', data=aggregate['f1_not_closed'], showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 2})
    plt.title(f'{group}\n Mean: {mean}')
    plt.xlabel('F1-Score')

# Adjust layout and spacing
plt.tight_layout()

# Save the plot as an image (choose the desired format, e.g., PNG, PDF, etc.)
plt.savefig('embedding_boxplotgrid.png')  # Change the filename and extension accordingly

# Show the plot (optional)
plt.show()