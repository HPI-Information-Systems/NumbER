import pandas as pd
import os
import sys
from tqdm import tqdm

folder = "/hpi/fs00/share/fg-naumann/lukas.laskowski/2MASS/psc/"
final_file = pd.DataFrame()
for file in tqdm(os.listdir(folder)):
    df = pd.read_csv(folder + file, sep="|", header=None)
    df = df[(df[0]>40) & (df[0]<60) & (df[1]>20)&(df[1]<40)&(df[56]!="\\N")]
    final_file = pd.concat([final_file, df])
    final_file.to_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/2MASS/filtered_match_psc.csv", index=False, header=False)
    print(f"File:  {file} done.", file=sys.stdout)
