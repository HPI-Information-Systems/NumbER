import pandas as pd

df = pd.read_csv("./NumbER/scripts/runs.csv")
filter= {
	"batch_size": 50,
	"n_epochs": 40,
	"lr": 3e-5,
	"max_len": 256,
	"lm": "roberta",
	"fp16": True,
}
filtered_df = df

for key, value in filter.items():
    if value is not None:
        filtered_df = filtered_df[filtered_df[key] == value]
    else:
        filtered_df = filtered_df[filtered_df[key].isnull()]
print(len(filtered_df))
#filtered_df = filtered_df.drop_duplicates(subset=['tags', 'dataset', 'i'])
print(len(filtered_df))
#filtered_df = filtered_df[filtered_df["dataset"]== "baby_products_numeric"]
#filtered_df = filtered_df[filtered_df["tags"] == "final_numeric_naive"]
#print(filtered_df[["i", "f1_not_closed", "training_time"]])
aggregate = filtered_df.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "std", "count"], "recall_not_closed": ["mean"], "training_time": ["mean", "std"]})
print(aggregate)
# mask = df.isin(filter).all(axis=1)
# print(df[mask])
# #group_by: i, dataset, tags, state
