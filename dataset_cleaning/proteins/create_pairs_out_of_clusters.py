#This script takes loads an npy file, which includes a list of clusters. It then creates a list of pairs from the clusters.
import pandas as pd
import numpy as np

all_clusters = np.load('all_clusters.npy', allow_pickle=True)
all_pairs = []
for cluster in all_clusters:
	for i in range(len(cluster)):
		for j in range(i+1, len(cluster)):
			all_pairs.append([cluster[i], cluster[j]])
pd.DataFrame(all_pairs, columns=['id1', 'id2']).to_csv('all_pairs.csv', index=False)