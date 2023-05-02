import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph
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
d1 = pd.read_csv("/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/experiments_01_05_2023_10:09/datasets/vsx/dataset.csv")
gt = pd.read_csv("./vsx_goldstandard.csv")
attr = ["V","RAdeg","DEdeg","Type","l_max","max","u_max","n_max","f_min","l_min","min","u_min","n_min","Epoch","u_Epoch","l_Period","Period","u_Period","id"]

data = Data(
    dataset_1=d1,
    id_column_name_1='id',
    ground_truth=gt,
    attributes_1=attr
)

data.process()
blocks = SuffixArraysBlocking(
    suffix_length=2
).build_blocks(data)
filtered_blocks = BlockFiltering(
    ratio=0.9
).process(blocks, data)

# cleaned_blocks = BlockPurging(
#     smoothing_factor=0.008
# ).process(blocks, data)


candidate_pairs_blocks = WeightedEdgePruning(
    weighting_scheme='BLAST'
).process(filtered_blocks, data)
EM = EntityMatching(
    metric='jaccard', 
    similarity_threshold=0.5
)

pairs_graph = EM.predict(filtered_blocks, data)
clusters = ConnectedComponentsClustering().process(pairs_graph)
e = Evaluation(data)
e.report(pairs_graph)