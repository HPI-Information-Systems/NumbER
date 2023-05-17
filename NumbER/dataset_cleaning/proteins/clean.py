import pandas as pd
import numpy as np

records = pd.read_csv("current_state.csv")
matches = pd.read_csv("all_pairs.csv")

records.drop(columns=['cluster', 'name', 'chain', "lig_cov","poc_cov","lig_name","4A_crit","ligSASRatio"], inplace=True)
#matches.drop(columns=['prediction'], inplace=True)

records = records[records['id'] <= 13000]
matches = matches[((matches['p1'] <= 13000) & (matches['p2'] <= 13000))]
#matches = matches[matches['p2'] <= 13000]
#for _, row in matches.iterrows():
#    if row['p1'] > 13000 or row['p2'] > 13000:
#        matches.drop(index=_, inplace=True)
records.to_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/protein/features.csv", index=False)
matches.to_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/protein/matches.csv", index=False)