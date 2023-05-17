import os
import sys
import pandas as pd
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)
from pyjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    # ReciprocalCardinalityWeightPruning,
    ComparisonPropagation
)
from pyjedai.matching import EntityMatching
from pyjedai.utils import print_clusters, print_blocks, print_candidate_pairs
from pyjedai.evaluation import Evaluation
from pyjedai.datamodel import Data
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.clustering import ConnectedComponentsClustering
from pyjedai.block_cleaning import BlockPurging
# d1 = pd.read_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx/features.csv")
# gt = pd.read_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/vsx/matches.csv")
# gt = gt[gt["prediction"] == 1]
# gt = gt[["p1","p2"]]
d1 = pd.read_csv("./abt.csv", sep='|', engine='python', na_filter=False).astype(str)
d2 = pd.read_csv("./buy.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv("./gt.csv", sep='|', engine='python')

print(gt)
for _, (el1, el2) in gt.iterrows():
    print(el1,el2)
    break
#attr = ["V","RAdeg","DEdeg","Type","l_max","max","u_max","n_max","f_min","l_min","min","u_min","n_min","Epoch","u_Epoch","l_Period","Period","u_Period","id"]
data = Data(
            dataset_1=d1,
            attributes_1=['id','name','description'],
            id_column_name_1='id',
            dataset_2=d2,
            attributes_2=['id','name','description'],
            id_column_name_2='id',
            ground_truth=gt,
		)
data.process()
blocks = StandardBlocking().build_blocks(data)
filtered_blocks = BlockFiltering(
    ratio=0.9
).process(blocks,data)
candidate_pairs_blocks = WeightedEdgePruning(weighting_scheme="EJS").process(filtered_blocks, data)
EM = EntityMatching(
    metric='jaccard', 
    similarity_threshold=0.7
)
pairs_graph = EM.predict(filtered_blocks, data)
print(pairs_graph)
clusters = ConnectedComponentsClustering().process(pairs_graph)
print(clusters)
e = Evaluation(data)
e.report(pairs_graph)