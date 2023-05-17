import os
import pandas as pd
import networkx as nx

base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"
for dataset in os.listdir(base_path):
	print("Doing dataset", dataset)
	pairs = pd.read_csv(os.path.join(base_path, dataset, "matches.csv"))
	pairs=[tuple(sorted((pair['p1'], pair['p2']))) for _,pair in pairs.iterrows()]
	print("Loaded pairs")
	#groundtruth = groundtruth[['p1', 'p2']].to_numpy()
	G = nx.Graph()
	G.add_edges_from(pairs)
	print("Created graph")
	clusters = []
	for connected_component in nx.connected_components(G):
		clusters.append(list(connected_component))
	print("Found clusters")
	pairs = []
	for cluster in clusters:
		for i in range(len(cluster)):
			for j in range(i+1, len(cluster)):
				pairs.append(tuple(sorted((cluster[i], cluster[j]))))
	print("Added transitive pairs")
	df = pd.DataFrame(pairs, columns=['p1', 'p2'])
	df['prediction'] = 1
	df.to_csv(os.path.join(base_path, dataset, "matches_closed.csv"), index=False)
	print("Saved to file")

