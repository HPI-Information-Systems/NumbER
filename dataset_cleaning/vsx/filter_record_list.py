#This script filters the paris list to only include records that occur in the records list.

import pandas as pd
import numpy as np

pairs = pd.read_csv('pairs.csv')
records = pd.read_csv('test.csv')
for index, row in pairs.iterrows():
	if row['id1'] not in records['Name'].values or row['id2'] not in records['Name'].values:
		pairs.drop(index, inplace=True)


pairs.to_csv('./filtered.csv', index=False)