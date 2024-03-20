import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# df = pd.read_csv("./NumbER/scripts/runs_books3_merged.csv")
df = pd.read_csv("./NumbER/scripts/runs_embitto_rerun_2.csv") # experiments for quality
df = pd.read_csv("./NumbER/scripts/runs_ensemble_embitto.csv") # experiments for quality
df = pd.read_csv("./NumbER/scripts/runs_single_dm.csv") # experiments for quality
# df = pd.read_csv("./NumbER/scripts/runs_holiday.csv") # experiments for runtime
# df = pd.read_csv("/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/scripts/runs_restart_2_emb.csv") #experiments for embeddings

 # pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific,
    # text_sim_formatter, complete_prompt_formatter_min_max_scaled
    # textual_min_max_scaled, textual_scientific
print(df[df['matching_solution'] == "embitto"])
filter= {
	## 'lr': 3e-5,
 	# #'should_finetune': True,
	# #'should_pretrain': False,
	'matching_solution': 'embitto',
	# #'finetune_batch_size': 50,
 	## 'num_finetune_epochs': 40,
	#'num_pretrain_epochs': ??,
	#'pretrain_batch_size': 50,
	## 'output_embedding_size': 256,
	#'include_numerical_features_in_textual': True,
	#'lm': 'roberta',
	#'fp16': True,
	#'max_len': 256,
	#'numerical_config_model': None,
	# 'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.value_embeddings.ValueValueEmbeddings",
	#'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.value_embeddings.ValueBaseEmbeddings",
	# 'numerical_config_model': "NumbER.matching_solutions.embitto.numerical_components.dice.DICEEmbeddingAggregator",
	# 'numerical_config_embedding_size': 128,
	# 'numerical_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.pair_based_numeric_formatter',
	# 'numerical_config_pretrain_formatter': 'NumbER.matching_solutions.embitto.formatters.dummy_formatter',
	# 'numerical_config_0': None,
	# #'textual_config_model': 'NumbER.matching_solutions.embitto.textual_components.base_roberta.BaseRoberta',
	## 'textual_config_max_length': 256,
	## 'textual_config_embedding_size': 128,
	# 'textual_config_finetune_formatter': 'NumbER.matching_solutions.embitto.formatters.complete_prompt_formatter',
	#'textual_config_pretrain_formatter': ??,
	#'textual_config_0': None, 
}

 # pair_based_ditto_formatter, textual_prompt_formatter, complete_prompt_formatter, complete_prompt_formatter_scientific, pair_based_ditto_formatter_scientific,
    # text_sim_formatter, complete_prompt_formatter_min_max_scaled
    # textual_min_max_scaled, textual_scientific
filtered_df = df

for key, value in filter.items():
    if value is not None:
        filtered_df = filtered_df[filtered_df[key] == value]
        print(len(filtered_df))
        print(key)
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
#print(len(filtered_df))
filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
filtered_df = filtered_df[~filtered_df["run"].isnull()]
filtered_df = filtered_df[filtered_df["dataset"] != "vsx"]
filtered_df = filtered_df[filtered_df["dataset"] != "vsx_small_numeric"]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["books3_numeric_no_isbn","x2_numeric","x3_numeric","earthquakes","vsx_small","2MASS_small"])]

#filtered_df = filtered_df["similar_sampling" in filtered_df["tags"]]
#print(len(filtered_df))
# filtered_df = filtered_df[filtered_df["dataset"]== "books3_numeric"]
# filtered_df = filtered_df[filtered_df["tags"] == "final_numeric_sorted"]
#print(filtered_df[["i", "f1_not_closed", "training_time"]])
# filtered_df = filtered_df[filtered_df["dataset"].isin(["2MASS_small_no_n", "baby_products_numeric", "books3_numeric", "earthquakes", "x2_numeric", "x3_numeric", "vsx_small","protein_small"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric"])][["f1_not_closed", "run"]]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged", "books3_all", "books3_merged", "books3_all_no_isbn", "books3_merged_no_isbn", "x2_all", "x2_merged", "x3_all", "x3_merged"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["protein_small","earthquakes","vsx_small","2MASS_small_no_n"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["protein_small","earthquakes","vsx_small","2MASS_small_no_n", "baby_products_numeric","books3_numeric","books3_numeric_no_isbn", "x2_numeric","x3_numeric"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged_new", "books3_all", "books3_merged_new", "x2_all", "x2_merged", "x3_all", "x3_merged"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "books3_all", "books3_merged", "x2_all", "x2_merged", "x3_all", "x3_merged", "books3_all_no_isbn", "books3_merged_no_isbn"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_beer_exp", "single_itunes_amazon", "single_fodors_zagat", "single_dblp_acm", "single_dblp_scholar", "single_amazon_google", "single_walmart_amazon"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_abt_buy"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_dblp_acm_dirty", "single_dblp_scholar_dirty", "single_itunes_amazon_dirty", "single_walmart_amazon_dirty"])]

print(filtered_df[["dataset", "f1_not_closed"]])
print(filtered_df[["f1_not_closed"]].mean())
#print(filtered_df)
filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"], "f1_closed": ["mean", "count"]})
#print(aggregate)
aggregate = aggregate[(aggregate["f1_not_closed"]["count"] > 4)]
print(aggregate)
print("Length", len(aggregate))
print("Trianing time mean", aggregate['training_time'].mean()['mean'])
print("MEAN", aggregate['f1_not_closed'].mean()['mean'])
print("Mean Closed", aggregate['f1_closed'].mean()['mean'])
print("Mean Precision", aggregate['precision_not_closed'].mean()['mean'])
print("Mean Recall", aggregate['recall_not_closed'].mean()['mean'])
#print(aggregate['f1_not_closed']['mean'])

#aggregate = filtered_df[(filtered_df['dataset'].str.contains("numeric")) | (filtered_df['dataset'].str.contains("earthquake"))].groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "var", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
aggregate["f1_not_closed"]["mean"].to_csv("result.csv")

data = aggregate['f1_not_closed']
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
