import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("./NumbER/scripts/runs.csv")
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
filtered_df = filtered_df[filtered_df["tags"].apply(lambda x: x == "similar_sampling")]#
filtered_df = filtered_df[filtered_df["dataset"] != "vsx"]
filtered_df = filtered_df[filtered_df["dataset"] != "vsx_small_numeric"]
filtered_df['training_time'] = filtered_df['training_time'].apply(lambda x: -x/60)
#filtered_df = filtered_df[filtered_df["dataset"].isin(["baby_products_numeric","books3_numeric","books3_numeric_no_isbn","x2_numeric","x3_numeric","earthquakes","vsx_small","2MASS_small"])]
aggregate = filtered_df.groupby(["dataset", "state"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
print(aggregate)
print(len(aggregate))

data = aggregate['training_time']
group_mean = data.mean()['mean']
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
