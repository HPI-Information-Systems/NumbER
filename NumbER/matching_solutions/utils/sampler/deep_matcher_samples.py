from NumbER.matching_solutions.utils.sampler.base import BaseSampler
from pathlib import Path
import pandas as pd
import os

class DeepMatcherBasedSampler(BaseSampler):
    def __init__(self, records_path, goldstandard_path):
        super().__init__(records_path, goldstandard_path)
        self.name = "deep_matcher_datasets"
        print("GONNA BE DeepMatcher Datasets")
        #self.distance_matrix = squareform(np.load(distance_path))
        
    def sample(self, *args):
        base_path = Path(self.records_path).parent
        train = pd.read_csv(os.path.join(base_path, "train.csv"))
        test = pd.read_csv(os.path.join(base_path, "test.csv"))
        valid = pd.read_csv(os.path.join(base_path, "valid.csv"))
        return train, valid, test
        
            
        
        