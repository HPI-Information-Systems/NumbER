import numpy as np
from NumbER.matching_solutions.utils.sampler.base import BaseSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from pyjedai.datamodel import Data
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)
import pandas as pd
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


class JedaiBasedSampler(BaseSampler):
    def __init__(self, records_path, goldstandard_path, attributes):
        super().__init__(records_path, goldstandard_path)
        self.records_path = records_path
        self.goldstandard_path = goldstandard_path
        self.goldstandard = self.goldstandard[self.goldstandard['prediction'] == 1]
        self.goldstandard = self.goldstandard[['p1', 'p2']]
        self.data = Data(
    		dataset_1=self.records,
    		id_column_name_1='id',
    		ground_truth=self.goldstandard,
    		attributes_1=attributes
		)
        self.data.process()


    def sample(self, *args):
        blocks = StandardBlocking().build_blocks(self.data)
        filtered_blocks = BlockFiltering(
            ratio=0.3
        ).process(blocks, self.data)
        candidate_pairs_blocks = BLAST(weighting_scheme="ARCS").process(filtered_blocks, self.data)
        EM = EntityMatching(
            metric='cosine', 
            similarity_threshold=0.98
        )
        pairs_graph = EM.predict(filtered_blocks, self.data)
        print(pairs_graph)
        clusters = ConnectedComponentsClustering().process(pairs_graph)
        pairs = self.convert_to_pairs(clusters)
        pairs['prediction'] = 1
        pairs = set(map(tuple, pairs.values))
        goldstandard = set(map(tuple, self.goldstandard.values))
        false_positives = pd.DataFrame(list(pairs.difference(goldstandard)), columns=['p1', 'p2'])
        false_positives = false_positives.sample(frac=1).reset_index(drop=True)
        train_matches = self.convert_to_pairs(clusters[:int(len(clusters)*train_fraction)])#wenn ein duplikatcluster geteilt wird, kann ein record in mehreren dateien vorkommen
        valid_matches = self.convert_to_pairs(clusters[int(len(clusters)*train_fraction):int(len(clusters)*(train_fraction+valid_fraction))])
        test_matches = self.convert_to_pairs(clusters[int(len(clusters)*(train_fraction+valid_fraction)):])
        #false_positives['prediction'] = 1
        #raise Exception("DHWJKDHW")
        e = Evaluation(self.data)
        e.report(pairs_graph)

        
            
        
        