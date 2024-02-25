import pandas as pd

#deepmatcher data
df_dmditto = pd.read_csv("./NumbER/scripts/runs_dittodm.csv")

df_sim = pd.read_csv("./NumbER/scripts/runs_similar_sampling.csv")
df_res = pd.read_csv("./NumbER/scripts/runs_restart.csv")

df_sim = df_sim[df_sim["matching_solution"] != "embitto"]
#df_res = df_res[df_res["matching_solution"] == "embitto"]

# combine = pd.concat([df_sim, df_res], axis=0)
# combine.to_csv('./NumbER/scripts/runs_test.csv')

