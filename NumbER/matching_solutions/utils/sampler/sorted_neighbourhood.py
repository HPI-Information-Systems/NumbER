from NumbER.matching_solutions.utils.sampler.base import BaseSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

class SortedNeighbourhoodSampler(BaseSampler):
    def __init__(self, records_path, goldstandard_path):
        super().__init__(records_path, goldstandard_path)
        self.name = "sorted_neighbourhood"
        
    def sample(self, *args):
        config = args[0]
        print(config)
        attributes = config['attributes']
        train_fraction = config['train_fraction']
        valid_fraction = config['valid_fraction']
        test_fraction = config['test_fraction']
        train_window_size = config['train_window_size']
        test_window_size = config['test_window_size']
        assert train_fraction + valid_fraction + test_fraction == 1
        if "prediction" in self.goldstandard.columns:
            groundtruth = self.goldstandard[self.goldstandard['prediction'] == 1]
        naive_sampler = NaiveSampler(self.records_path, self.goldstandard_path)
        train_records, valid_records, test_records, train_matches, valid_matches, test_matches = naive_sampler.sample_records(config)
        train_matches = set(tuple(sorted(pair)) for pair in train_matches[['p1', 'p2']].to_numpy())
        test_matches = set(tuple(sorted(pair)) for pair in test_matches[['p1', 'p2']].to_numpy())
        valid_matches = set(tuple(sorted(pair)) for pair in valid_matches[['p1', 'p2']].to_numpy())
        train_pairs = self.create_pairs(train_records, attributes, 'id', train_window_size, groundtruth, train_matches)
        valid_pairs = self.create_pairs(valid_records, attributes, 'id', test_window_size, groundtruth, valid_matches)
        test_pairs = self.create_pairs(test_records, attributes, 'id', test_window_size, groundtruth, test_matches)
        return train_pairs, valid_pairs, test_pairs
    
    def map_to_groundtruth(self, pairs, gt):
        if 'prediction' in gt.columns:
            gt = gt[gt['prediction'] == 1]
        gt = gt[['p1', 'p2']].to_numpy()
        gt = set(tuple(sorted(pair)) for pair in gt)
        return [1 if pair in gt else 0 for pair in pairs]
    
    def create_pairs(self, data, attributes, id_column, size, matches, additional_pairs=None):
        pairs = self.sorted_neighbourhood(data, attributes, id_column, size)
        if additional_pairs is not None:
            pairs.update(additional_pairs)
        l = self.map_to_groundtruth(pairs, matches)
        pairs = pd.DataFrame(pairs, columns=['p1', 'p2'])
        pairs['prediction'] = l
        return pairs
    
    def sorted_neighbourhood(self,records, attributes, id_column, size):
        all_pairs = set()
        for attribute in attributes:
            records_temp = records.sort_values(by=[attribute])[id_column].to_numpy()
            sliding_windows = sliding_window_view(records_temp, size)
            pairs = set()
            for window in sliding_windows:
                for i in range(len(window)):
                    for j in range(i+1, len(window)):
                        pairs.add(tuple(sorted((window[i], window[j]))))
            all_pairs.update(pairs)
        return all_pairs