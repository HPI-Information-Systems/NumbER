import pandas as pd
import numpy as np
import os



def get_numeric_columns(df):        
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	return list(df.select_dtypes(include=numerics).columns)

agg_datasets = ["baby_products_all", "books3_all", "x2_merged", "x3_merged", "2MASS_small_no_n","baby_products_numeric","books3_numeric","x2_numeric","x3_numeric", "protein_small", "earthquakes", "vsx_small"]
single_datasets = [ "single_dblp_acm_dirty", "single_abt_buy", "single_dblp_scholar_dirty", "single_itunes_amazon_dirty", "single_walmart_amazon_dirty", "single_beer_exp", "single_itunes_amazon", "single_fodors_zagat", "single_dblp_acm", "single_dblp_scholar", "single_amazon_google", "single_walmart_amazon"]
all_datasets = []
all_datasets.extend(agg_datasets)
all_datasets.extend(single_datasets)
#datasets = agg_datasets.append(single_datasets)

final_df = []
for dataset in all_datasets:
    base_path = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/features.csv"
    df = pd.read_csv(base_path)
    df.drop(['prediction', 'id'], axis=1, inplace=True, errors="ignore") 
    numerical_columns = len(get_numeric_columns(df))
    all_columns = len(df.columns)
    print(dataset)
    if dataset == "single_dblp_scholar_dirty": 
        continue
    if dataset not in agg_datasets:
        ensemble_data = pd.read_csv("./NumbER/scripts/runs_single_dm.csv")
        ditto_data = pd.read_csv("./NumbER/scripts/runs_single_dm.csv")
        ditto_data = ditto_data[ditto_data["matching_solution"] == "ditto"]
        ensemble_data = ensemble_data[ensemble_data["matching_solution"] == "ensemble_learner"]
        ditto_result = ditto_data[ditto_data["dataset"] == dataset]["f1_not_closed"].values[0]
        ensemble_result = ensemble_data[ensemble_data["dataset"] == dataset]["f1_not_closed"].values[0]
    else:
        ensemble_data = pd.read_csv("./NumbER/scripts/runs_ensemble.csv")
        ensemble_data = ensemble_data[ensemble_data["dataset"] == dataset]
        ensemble_data = ensemble_data[ensemble_data["matching_solution"] == "ensemble_learner"]
        ensemble_data = ensemble_data.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"], "f1_closed": ["mean", "count"]})
        #print(ensemble_data)
        ensemble_data = ensemble_data[(ensemble_data["f1_not_closed"]["count"] > 3)]
        ditto_data = pd.read_csv("./NumbER/scripts/runs_dittodm_final.csv")
        ditto_data = ditto_data[ditto_data["dataset"] == dataset]
        ditto_data = ditto_data[ditto_data["matching_solution"] == "ditto"]
        ditto_data = ditto_data.groupby(["dataset", "tags", "state"]).agg({"f1_not_closed": ["mean", "count"], "recall_not_closed": ["mean"], "precision_not_closed": ["mean"], "training_time": ["mean", "std"], "f1_closed": ["mean", "count"]})
        ditto_data = ditto_data[(ditto_data["f1_not_closed"]["count"] > 4)]
        ditto_result = ditto_data["f1_not_closed"]['mean'].values[0]
        ensemble_result = ensemble_data["f1_not_closed"]['mean'].values[0]
    

    final_df.append({
        "dataset": dataset,
        'no_of_numerical_cols': numerical_columns,
        'relation': numerical_columns / all_columns,
        'result_diff': ensemble_result - ditto_result,
        'ensemble_res': ensemble_result,
        'ditto_res': ditto_result,
    })

df = pd.DataFrame(final_df).sort_values(by=["relation"])
print(df)
print("DO NOT USE FOR FINAL RESULT! 3 runs are sufficient for result evaluation here. This is not good!")
aggregate = df.groupby(["relation"]).agg({"result_diff": ["mean", "count"], "ensemble_res": ["mean"], "ditto_res": ["mean"]})
print(aggregate)
#print(pd.DataFrame(final_df).groupby(["relation", ]).agg({"f1_not_closed": ["mean", "count"], (['relation']).mean())