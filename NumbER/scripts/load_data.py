
import pandas as pd
import wandb
import argparse
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tags", required=True, nargs='+', help="Tags of experiments to be loaded")
args = vars(ap.parse_args())
print(args)
tags = args["tags"]

api = wandb.Api()
# Project is specified by <entity/project-name>
runs = api.runs("lasklu/NumbER")

summary_list, config_list, name_list, tag_list, state_list = [], [], [], [], []
for run in tqdm(runs): 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)
    tag_list.append(run.tags)
    state_list.append(run.state)
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)


#tag = "similar_sampling"
for tag in tags:#, "similar_sampling"]:
    runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
    "tags": tag_list,
    "state": state_list
    })
    runs_df = runs_df[runs_df["tags"].apply(lambda x: tag in x[0])]
    runs_df = runs_df[runs_df["state"] == "finished"]
    for col in runs_df.columns:
        if col in ["summary", "config"]:
            #convert dictionary to columns in dataframe
            runs_df = pd.concat([runs_df.drop([col], axis=1), runs_df[col].apply(pd.Series)], axis=1)
            #convert numerical_config column to real columns in dataframe but add numeric_config prefix
    if "numerical_config" in runs_df.columns:
        runs_df = pd.concat([runs_df.drop(["numerical_config"], axis=1), runs_df["numerical_config"].apply(pd.Series).add_prefix("numerical_config_")], axis=1)#!
        runs_df = pd.concat([runs_df.drop(["textual_config"], axis=1), runs_df["textual_config"].apply(pd.Series).add_prefix("textual_config_")], axis=1)#!
    runs_df["tags"] = runs_df["tags"].apply(lambda x: x[0])
    #print(runs_df)
    runs_df.to_csv(f"./NumbER/scripts/runs_{tag}.csv")

df_sim = pd.read_csv("./NumbER/scripts/runs_similar_sampling.csv")
df_res = pd.read_csv("./NumbER/scripts/runs_restart.csv")

df_sim = df_sim[df_sim["matching_solution"] != "embitto"]
#df_res = df_res[df_res["matching_solution"] == "embitto"]

# combine = pd.concat([df_sim, df_res], axis=0)
# combine.to_csv('./NumbER/scripts/runs_test.csv')


