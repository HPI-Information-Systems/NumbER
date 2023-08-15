import os
import pandas as pd
from NumbER.matching_solutions.utils.transitive_closure import calculate_clusters, convert_to_pairs


base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"
dataset = "2MASS"
new_dataset = "2MASS_small"
dataset_path = os.path.join(base_path, dataset)

features = pd.read_csv(os.path.join(dataset_path, "features.csv"))
#features.drop(columns=['hemis'], inplace=True)
#gt = pd.read_csv(os.path.join(dataset_path, "matches.csv"))
gt = pd.read_csv(os.path.join(dataset_path, "matches_closed.csv"))

clusters = calculate_clusters(gt)
#sample clusters so that approx. 6500 pairs are included in the new dataset
clusters = clusters[:400]
pairs = convert_to_pairs(clusters)
all_record_ids = set(pairs['p1'].to_list()).union(set(pairs['p2'].to_list()))
features = features[features['id'].isin(all_record_ids)]


#to csv
features.to_csv(os.path.join(base_path, new_dataset, "features.csv"), index=False)
pairs.to_csv(os.path.join(base_path, new_dataset, "matches_closed.csv"), index=False)
#gt.to_csv(os.path.join(base_path, new_dataset, "matches.csv"), index=False)
