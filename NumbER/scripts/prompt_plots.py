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
	'textual_config_embedding_size': 256,
	'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.textual_scientific',#_scientific',
	#'textual_config_pretrain_formatter': ??,
	'textual_config_0': None, 
}


# Create the 3x3 grid of boxplots
#plt.figure(figsize=(12, 24))  # Adjust the figure size if needed
fig, ax = plt.subplots()
y_data = []
labels=[]
means = []
# Iterate through each unique group and create a boxplot for each one
for i, group in enumerate([
    "complete_prompt_formatter", "complete_prompt_formatter_scientific", "complete_prompt_formatter_min_max_scaled",
    "textual_prompt_formatter", "textual_scientific", "textual_min_max_scaled",
    "pair_based_ditto_formatter", "pair_based_ditto_formatter_scientific",
    "text_sim_formatter",
]):
    filtered_df = df
    filter["textual_config_finetune_formatter"] = f"NumbER.matching_solutions.embitto.formatters.{group}"
    
    for key, value in filter.items():
        if value is not None:
            filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = filtered_df[filtered_df[key].isnull()]
    filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
    filtered_df = filtered_df[~filtered_df["run"].isnull()]
    #filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_all","books3_numeric","books3_numeric_no_isbn","earthquakes","vsx_small","x2_all","x2_combined","x3_all","x3_combined","x3_numeric" ])]
    filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn", "x2_numeric","x3_numeric","earthquakes","vsx_small","2MASS_small"])]
    filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
    #print(filtered_df['training_time'])
    aggregate = filtered_df.groupby(["dataset", "state"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
    #aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
    aggregate = aggregate[aggregate["f1_not_closed"]["count"] > 4]
    print(group)
    print(aggregate)
    mean = aggregate['f1_not_closed'].mean()['mean']
    means.append(mean)
    #plt.subplot(3, 3, i + 1)
    #plt.xlim(left=0.0, right=1.0)
    y_data.append(aggregate['f1_not_closed']['mean'].values)
    labels.append(group)
    #sns.boxplot(x='mean', data=aggregate['f1_not_closed'], showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 2})
    #plt.title(f'{group}	\nMean: {mean:.2f}')
    #plt.xlabel('F1-Score')

# Adjust layout and spacing
#ax.xtickslabels(labels)
for i, col in enumerate(y_data):
    mean = means[i]
    ax.annotate(f'{mean:.2f}', xy=(i+1, 0.3), xytext=(0, -40),
             textcoords='offset points', ha='center', va='bottom')
sns.boxplot(y_data, showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 1}, orient='h')
ax.set_yticklabels(labels, rotation=90, ha='right')
plt.tight_layout()

# Save the plot as an image (choose the desired format, e.g., PNG, PDF, etc.)
plt.savefig('prompts_boxplotgrid_inified.png')  # Change the filename and extension accordingly

# Show the plot (optional)
plt.show()