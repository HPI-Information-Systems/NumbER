import pandas as pd
import os
from shutil import copyfile

datasets = ['books3']
path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"

for dataset in datasets:
    original = pd.read_csv(f"{path}/{dataset}_all_no_isbn/features.csv")
    numerical = pd.read_csv(f"{path}/{dataset}_numeric_no_isbn/features.csv")
    #matches = pd.read_csv(f"{path}/{dataset}/matches.csv")
    original_columns = list(set(original.columns) - set(numerical.columns))
    #extracted_features = list(set(extracted.columns) - set(original.columns))
    #print("Extracted", extracted_features)
    df = pd.concat([numerical, original[original_columns]], axis=1)
    os.makedirs(f"{path}/{dataset}_merged_no_isbn", exist_ok=True) 
    df.to_csv(f"{path}/{dataset}_merged_no_isbn/features.csv", index=False)
    copyfile(f"{path}/{dataset}_all_no_isbn/matches_closed.csv", f"{path}/{dataset}_merged_no_isbn/matches_closed.csv")
    copyfile(f"{path}/{dataset}_all_no_isbn/similarity_cosine.npy", f"{path}/{dataset}_merged_no_isbn/similarity_cosine.npy")
    
    
    
    