import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#df = pd.read_csv("./NumbER/scripts/runs_x3.csv")
df = pd.read_csv("./NumbER/scripts/runs_embeddings.csv")
#df = pd.read_csv("./NumbER/scripts/runs_holiday.csv")
#df = pd.read_csv("./NumbER/scripts/runs_deep.csv")
#df = pd.read_csv("./NumbER/scripts/runs_books3_merged.csv")
df = pd.read_csv("./NumbER/scripts/runs_dm_data.csv")
df = pd.read_csv("./NumbER/scripts/runs_single_dm.csv") #runtime for ditto

filter= {
	#"batch_size": 50,
	#"epochs": 40,
 	"matching_solution": "ensemble_learner",
}
filtered_df = df
print(filtered_df["tags"].unique())
for key, value in filter.items():
    if value is not None:
        filtered_df = filtered_df[filtered_df[key] == value]
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
#filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'i'])
#filtered_df = filtered_df[filtered_df["dataset"]== "baby_products_numeric"]
#filtered_df = filtered_df[filtered_df["tags"] == "final_numeric_naive"]
#print(filtered_df[["i", "f1_not_closed", "training_time"]])
#filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: x == "similar_sampling")]#
filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
# filtered_df = filtered_df[filtered_df["dataset"] != "vsx"]
# filtered_df = filtered_df[filtered_df["dataset"] != "vsx_small_numeric"]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn","x2_numeric","protein_small", "x3_numeric","earthquakes","vsx_small","2MASS_small"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["2MASS_small_no_n", "baby_products_numeric", "books3_numeric", "books3_numeric_no_isbn", "earthquakes", "x2_numeric", "x3_numeric", "vsx_small","protein_small"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["2MASS_small_no_n", "earthquakes", "vsx_small","protein_small"])]


# filtered_df = filtered_df[filtered_df["dataset"].isin([ "baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin([ "baby_products_all","books3_all","x2_all","x3_all"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged_new", "books3_all", "books3_merged_new", "x2_all", "x2_merged", "x3_all", "x3_merged"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn","x2_numeric","x3_numeric"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["protein_small","earthquakes","vsx_small","2MASS_small_no_n", "baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]#aggregate = filtered_df.groupby(["dataset", "state"]).agg({"f1_not_closed": ["mean", "std", "count"], "training_time": ["mean", "std"]})
print(filtered_df[filtered_df['dataset']=="books3_all"]['f1_not_closed'])
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"]})
print("Runtime", aggregate["training_time"]["mean"].mean())
print(aggregate)
print(len(aggregate))
data = aggregate['f1_not_closed']
group_mean = data.mean()['mean']
print(group_mean)
print("Recall", aggregate['recall_not_closed'].mean())
print("Precision", aggregate['precision_not_closed'].mean())
plt.text(0.5, group_mean, f'Mean: {group_mean:.2f}', ha='center', va='bottom', color='red')
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.boxplot(x='mean',  data=aggregate["f1_not_closed"], showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 2})

# Add labels and title
plt.xlabel('Group')
plt.ylabel('Value')
plt.title(f'Deepmatcher Mean: {group_mean:.2f}')
plt.xlim(left=0.2, right=1.0)
# Show the plot (optional)
plt.show()
plt.savefig('deepmatcher_boxplot.png')
# mask = df.isin(filter).all(axis=1)
# print(df[mask])
# #group_by: i, dataset, tags, state
