import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/scripts/runs_dm_comparison.csv")
#df = pd.read_csv("/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/scripts/runs_x3.csv")
embitto_filter= {
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
	#'textual_config_0': None, 
}

ditto_filter = {
	"batch_size": 50,
	"n_epochs": 40,
	"lr": 3e-5,
	"max_len": 256,
	"lm": "roberta",
	"fp16": True,
 	"matching_solution": "ditto",
}
deep_matcher_filter = {
	#"batch_size": 50,
	#"epochs": 40,
 	"matching_solution": "deep_matcher",
}
# Create the 3x3 grid of boxplots
#plt.figure(figsize=(12, 24))  # Adjust the figure size if needed
y_data = []
labels=[]
means = []
ditto_mean = 0.0
ditto_y_data = []
result_df = pd.DataFrame(columns=["dataset", "f1_not_closed", "recall_not_closed", "precision_not_closed", "training_time"])
# Iterate through each unique group and create a boxplot for each one
for i, group in enumerate([
    "complete_prompt_formatter", "complete_prompt_formatter_scientific", "complete_prompt_formatter_min_max_scaled",
    "textual_prompt_formatter", "textual_scientific", "textual_min_max_scaled",
    "pair_based_ditto_formatter", "pair_based_ditto_formatter_scientific",
    "text_sim_formatter", 
    "deep_matcher", "ditto"
]):
    filtered_df = df
    if group == "ditto":
        filter = ditto_filter
    elif group == "deep_matcher":
        filter = deep_matcher_filter
    else:
        embitto_filter["textual_config_finetune_formatter"] = f"NumbER.matching_solutions.embitto.formatters.{group}"
        filter = embitto_filter
    for key, value in filter.items():
        if value is not None:
            filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = filtered_df[filtered_df[key].isnull()]
    filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
    print("s", filtered_df)
    filtered_df = filtered_df[~filtered_df["run"].isnull()]
    #filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_all","books3_numeric","books3_numeric_no_isbn","earthquakes","vsx_small","x2_all","x2_combined","x3_all","x3_combined","x3_numeric" ])]
    #NUMERICAL DATASETS
    #filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn", "x2_numeric","x3_numeric","earthquakes","vsx_small", "protein_small", "2MASS_small_no_n"])]
    #TEXTUAL DATASETS
    #filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all","baby_products_combined","books3_all", "books3_all_no_isbn","books3_combined","books3_combined_no_isbn","x2_all","x2_combined", "x3_all", "x3_combined"])]
    filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
    
    aggregate = filtered_df.groupby(["dataset", "state"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"]})
    aggregate = aggregate[aggregate["f1_not_closed"]["count"] > 4]
    print(aggregate)
    for idx, row in aggregate.iterrows():
        if group == "ditto":
            solution = "SOTA"
        elif group == "deep_matcher":
            solution = "SOTA"
        else:
            solution = "NumbER"
        result_df = result_df.append({"dataset": idx[0], "matching_solution": solution, "f1_not_closed": row["f1_not_closed"]["mean"], "recall_not_closed": row["recall_not_closed"]["mean"], "precision_not_closed": row["precision_not_closed"]["mean"], "training_time": row["training_time"]["mean"],"algorithm": group},  ignore_index=True)
    mean = aggregate['f1_not_closed'].mean()['mean']
    if group == "ditto":
        ditto_mean = mean
        ditto_y_data = aggregate['f1_not_closed']['mean'].values
        continue
    elif group == "deep_matcher":
        print(filtered_df)
        aggregate = filtered_df.groupby(["dataset", "state"]).agg({"f1_reported": ["mean", "std", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
        deep_matcher_mean = aggregate['f1_reported'].mean()['mean']
        deep_matcher_y_data = aggregate['f1_reported']['mean'].values
        continue
    print(group)
    print(mean)
    means.append(mean)
    y_data.append(aggregate['f1_not_closed']['mean'].values)
    labels.append(f"{group}")#\nMean: {mean:.2f}")
    means_sorted = np.argsort(means)
print(result_df)
result_df["algo_display_name"] = result_df["algorithm"]
result_df["algo_training_type"] = result_df["matching_solution"]
result_df["precision"] = result_df["precision_not_closed"]
result_df["f1"] = result_df["f1_not_closed"]
result_df["recall"] = result_df["recall_not_closed"]
result_df["algo_input_dimensionality"] = "TEST"
result_df["dataset_training_type"] = "TEST"
result_df["dataset_input_dimensionality"] = "TEST"
result_df["train_preprocess_time"] = 0
result_df["error_category"] = "- OK -"
result_df["algo_family"] = "forecasting"
result_df["algo_area"] = "Statistics (Regression & Forecasting)"
result_df["collection"] ="TEST"
rename = {
     'pair_based_ditto_formatter': 'Naive',
     'pair_based_ditto_formatter_scientific': 'NaiveScientific',
     'complete_prompt_formatter': 'Distance',
     'complete_prompt_formatter_scientific': 'DistanceScientific',
     'complete_prompt_formatter_min_max_scaled': 'MinMaxDistance',
     'text_sim_formatter': 'TextualDistance',
     'textual_prompt_formatter': 'AttributePair',
     'textual_scientific': 'AttributePairScientific',
     'textual_min_max_scaled': 'AttributePairMinMax',
     'deep_matcher': 'DeepMatcher',
     'ditto': 'Ditto',
}
#rename all values of result_df in column algorithm according to rename
result_df['algorithm'] = result_df['algorithm'].apply(lambda x: rename[x])
result_df.to_csv("dm_comparison.csv", index=False)