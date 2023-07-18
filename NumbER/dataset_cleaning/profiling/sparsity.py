import pandas as pd
import numpy as np
import os

def sparsity(df):
    #this defines the percentage of missing values aggregated over all columns except the id column
	return df.drop(['id'], axis=1).isnull().sum().sum() / (df.shape[0] * df.shape[1])

def amount_of_attributes(df):
	return df.shape[1] - 1

def amount_of_records(df):
	return df.shape[0]

def amount_of_matches(matches):
	return matches.shape[0]

def amount_of_numerical_columns(df):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	cols = list(df.select_dtypes(include=numerics).columns)
	return len(cols)

def average_length_of_samples(df):
    pass


base_path = ""
result = []
for dataset in os.listdir(base_path):
    df = pd.read_csv(os.path.join(base_path, dataset, "features.csv"))
    matches = pd.read_csv(os.path.join(base_path, dataset, "matches.csv"))
    result.append({
		"dataset": dataset,
		"sparsity": sparsity(df),
		"amount_of_attributes": amount_of_attributes(df),
		"amount_of_records": amount_of_records(df),
		"amount_of_matches": amount_of_matches(matches),
		"amount_of_numerical_columns": amount_of_numerical_columns(df)
	})