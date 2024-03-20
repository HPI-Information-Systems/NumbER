import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("./NumbER/scripts/runs_books3_merged.csv")
df = pd.read_csv("./NumbER/scripts/runs_dittodm_final.csv")
df = pd.read_csv("./NumbER/scripts/runs_holiday.csv") #runtime for ditto
df = pd.read_csv("./NumbER/scripts/runs_ensemble_embitto.csv") #runtime for ditto
df = pd.read_csv("./NumbER/scripts/runs_dm_data.csv") #runtime for ditto
df = pd.read_csv("./NumbER/scripts/runs_single_dm.csv") #runtime for ditto

# df = pd.read_csv("./NumbER/scripts/runs_holiday.csv")
filter= {
	"batch_size": 50,
	"n_epochs": 40,
	"lr": 3e-5,
	"max_len": 256,
	"lm": "roberta",
	"fp16": True,
 	"matching_solution": "ditto",
}
filtered_df = df
print(filtered_df["tags"].unique())
for key, value in filter.items():
    if value is not None:
        filtered_df = filtered_df[filtered_df[key] == value]
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'run'])
#filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: x == "similar_sampling")]#

#filtered_df = filtered_df[filtered_df["dataset"] != "vsx"]
#filtered_df = filtered_df[filtered_df["dataset"] != "vsx_small_numeric"]

#print(filtered_df)
filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_beer_exp", "single_itunes_amazon", "single_fodors_zagat", "single_dblp_acm", "single_dblp_scholar", "single_amazon_google", "single_walmart_amazon"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_abt_buy"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["single_dblp_acm_dirty", "single_dblp_scholar_dirty", "single_itunes_amazon_dirty", "single_walmart_amazon_dirty"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_combined", "books3_all", "books3_combined", "books3_all_no_isbn", "books3_combined_no_isbn", "x2_all", "x2_combined", "x3_all", "x3_combined"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged_new", "books3_all", "books3_merged","books3_merged_no_isbn", "books3_all_no_isbn", "x2_all", "x2_merged", "x3_all", "x3_merged"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["protein_small","earthquakes","vsx_small","2MASS_small_no_n", "baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","x2_numeric","x3_numeric"])]


# filtered_df = filtered_df[filtered_df["dataset"].isin(["2MASS_small_no_n", "earthquakes",  "vsx_small","protein_small", "baby_products_numeric", "books3_numeric",  "x2_numeric", "x3_numeric"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged", "books3_all", "books3_merged", "books3_all_no_isbn", "books3_merged_no_isbn", "x2_all", "x2_merged", "x3_all", "x3_merged"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn","x2_numeric","x3_numeric"])]
#filtered_df = filtered_df[filtered_df["dataset"].isin(["protein_small","earthquakes","vsx_small","2MASS_small_no_n"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged", "books3_all", "books3_merged", "books3_all_no_isbn", "books3_merged_no_isbn", "x2_all", "x2_merged", "x3_all", "x3_merged"])]
# filtered_df = filtered_df[filtered_df["dataset"].isin([ "baby_products_all","books3_all","x2_all","x3_all"])]

# filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_all", "baby_products_merged_new", "books3_all", "books3_merged_new", "x2_all", "x2_merged", "x3_all", "x3_merged"])]
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"], "f1_closed": ["mean", "count"], "recall_closed": ["mean"], "precision_closed": ["mean"], "training_time": ["mean"]})
print(aggregate)
print(len(aggregate))

data = aggregate['f1_not_closed']
group_mean = data.mean()['mean']
print("Trianing time mean", aggregate['training_time'].mean()['mean'])
print("MEAN", group_mean)
print("Mean Closed", aggregate['f1_closed'].mean()['mean'])
print("Mean Precision", aggregate['precision_not_closed'].mean()['mean'])
print("Mean Precision Closed", aggregate['precision_closed'].mean()['mean'])
print("Mean Recall Closed", aggregate['recall_closed'].mean()['mean'])
print("Mean Recall", aggregate['recall_not_closed'].mean()['mean'])
plt.text(0.5, group_mean, f'Mean: {group_mean:.2f}', ha='center', va='bottom', color='red')
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.boxplot(x='mean',  data=data, showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 2})

# Add labels and title
plt.xlabel('Group')
plt.ylabel('Value')
plt.title(f'Ditto Mean: {group_mean:.2f}')
plt.xlim(left=0.2, right=1.0)

# Show the plot (optional)
plt.show()
plt.savefig('ditto_boxplot.png')

data = {
	'combined': (
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x2_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x3_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_combined']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_combined_no_isbn']["f1_not_closed"]["mean"].values[0],2),
	),
 	'all': (
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x2_all']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'x3_all']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_all']["f1_not_closed"]["mean"].values[0],2),
		round(aggregate[aggregate.index.get_level_values('dataset') == 'books3_all_no_isbn']["f1_not_closed"]["mean"].values[0],2),
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

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-Score')
ax.set_title('ditto')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0.6, 1.2)

plt.show()
plt.savefig("penguins.png")
