import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph
from pyjedai.datamodel import Data

from pyjedai.utils import print_clusters, print_blocks, print_candidate_pairs
from pyjedai.evaluation import Evaluation
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
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.clustering import ConnectedComponentsClustering

from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

class MagellanMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
        
    def model_train(self, epochs, batch_size, pos_neg_ratio):
        d1 = pd.read_csv("./../data/cora/cora.csv", sep='|')
        gt = pd.read_csv("./../data/cora/cora_gt.csv", sep='|', header=None)
        attr = ['Entity Id','author', 'title']
        data = Data(
			dataset_1=d1,
			id_column_name_1='Entity Id',
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
        candidate_pairs_blocks = WeightedEdgePruning(
			weighting_scheme='CBS'
		).process(filtered_blocks, data)
        attr = ['author', 'title']
        attr = {
			'author' : 0.6,
			'title' : 0.4
		}
        EM = EntityMatching(
			metric='jaccard', 
			similarity_threshold=0.5
		)
        pairs_graph = EM.predict(filtered_blocks, data)
        clusters = ConnectedComponentsClustering().process(pairs_graph)
        e = Evaluation(data)
        e.report(pairs_graph)
        return best_f1, model, None, time.time() - start_time
        
    def model_predict(self, model):
        return {'predict': predictions, 'evaluate': f1}