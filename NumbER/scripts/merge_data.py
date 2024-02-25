import pandas as pd
import numpy as np

df_sim = pd.read_csv("./NumbER/scripts/runs_restart.csv")
df_books3_merged_no_isbn = pd.read_csv("./NumbER/scripts/runs_books3_merged.csv")
df_res = pd.read_csv("./NumbER/scripts/runs_x3.csv")

#df_sim = df_sim[df_sim["matching_solution"] != "embitto"]
df_sim = df_sim[(df_sim["dataset"] != "books3_merged_no_isbn")]
df_sim = df_sim[(df_sim["dataset"] != "x3_merged") | (df_sim["matching_solution"] != "deep_matcher")]
print(df_sim[df_sim["dataset"] == "x3_merged"]['matching_solution'])
df_sim = df_sim[(df_sim["dataset"] != "x3_all") | (df_sim["matching_solution"] != "deep_matcher")]
#df_res = df_res[df_res["matching_solution"] == "embitto"]

combine = pd.concat([df_sim, df_res], axis=0)
combine = pd.concat([combine, df_books3_merged_no_isbn], axis=0)
combine.to_csv('./NumbER/scripts/runs_restart_final.csv')