from NumbER.matching_solutions.utils.sampler.base import BaseSampler
import pandas as pd
import random
import numpy as np
import time

class NaiveSampler(BaseSampler):
    def __init__(self, records, goldstandard):
        print("Be naive like you")
        super().__init__(records, goldstandard)
        self.name = "naive"

    def sample_records(self, *args):
        config = args[0]
        train_fraction = config['train_fraction']
        valid_fraction = config['valid_fraction']
        test_fraction = config['test_fraction']
        print("fractions: ", train_fraction, valid_fraction, test_fraction)
        assert train_fraction + valid_fraction + test_fraction == 1
        groundtruth = self.goldstandard.sample(frac=1).reset_index(drop=True)
        if "prediction" in groundtruth.columns:
            groundtruth = groundtruth[groundtruth['prediction'] == 1]
        clusters_ = self.create_clusters(groundtruth)
        print("clusters", clusters_)
        
        random.seed(time.clock())
        random.shuffle(clusters_)
        
        #print("Shuffled clusters", clusters_)
        #np.random.shuffle(clusters_)
        #print("shuffled", clusters_)
        #print(len(clusters_))
        #print("SAMPLED", random.sample(clusters_, k=len(clusters_)))
        print("avergae cluster size: ", sum([len(cluster) for cluster in clusters_])/len(clusters_))
        train_matches = self.convert_to_pairs(clusters_[:int(len(clusters_)*train_fraction)])#wenn ein duplikatcluster geteilt wird, kann ein record in mehreren dateien vorkommen
        #print("train matches", train_matches)
        valid_matches = self.convert_to_pairs(clusters_[int(len(clusters_)*train_fraction):int(len(clusters_)*(train_fraction+valid_fraction))])
        #print("valid matches", valid_matches)
        test_matches = self.convert_to_pairs(clusters_[int(len(clusters_)*(train_fraction+valid_fraction)):])
        #print("test matches", test_matches)
        train_matches['prediction'] = 1
        valid_matches['prediction'] = 1
        test_matches['prediction'] = 1
        train_ids = set(train_matches['p1'].to_list() + train_matches['p2'].to_list())
        valid_ids = set(valid_matches['p1'].to_list() + valid_matches['p2'].to_list())
        test_ids = set(test_matches['p1'].to_list() + test_matches['p2'].to_list())
        train_records = self.records.loc[self.records['id'].isin(list(train_ids))]
        valid_records = self.records.loc[self.records['id'].isin(list(valid_ids))]
        test_records = self.records.loc[self.records['id'].isin(list(test_ids))]
        return train_records.reset_index(), valid_records.reset_index(), test_records.reset_index(), train_matches.reset_index(), valid_matches.reset_index(), test_matches.reset_index()
    
    def generate_unique_pairs(self, input_pairs, output_length):
        amount_of_possible_non_matches = (len(input_pairs) * (len(input_pairs) - 1)) / 2
        output_length = min(output_length, amount_of_possible_non_matches)
        input_pairs_set = set(tuple(sorted((pair['p1'], pair['p2']))) for _,pair in input_pairs.iterrows())
        #ids = list(set(id for _,pair in input_pairs.iterrows() for id in pair))
        ids = list(set(list(input_pairs['p1'].to_list()) + list(input_pairs['p2'].to_list())))
        unique_pairs = []
        while len(unique_pairs) < output_length:
            r1, r2 = random.sample(ids, 2)
            pair = (r1, r2)
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in input_pairs_set:
                input_pairs_set.add(sorted_pair)
                unique_pairs.append(pair)
        df = pd.DataFrame(unique_pairs, columns=['p1', 'p2'])
        df['prediction'] = 0
        return pd.concat([input_pairs, df])

    def sample(self, *args):
        config = args[0]
        train_records, valid_records, test_records, train_matches, valid_matches, test_matches = self.sample_records(config)
        #train_matches = set(tuple(sorted(pair)) for pair in train_matches[['p1', 'p2']].to_numpy())
        
        #valid_matches = set(tuple(sorted(pair)) for pair in valid_matches[['p1', 'p2']].to_numpy())
        #test_matches = set(tuple(sorted(pair)) for pair in test_matches[['p1', 'p2']].to_numpy())
        train_data = self.generate_unique_pairs(train_matches, len(train_matches))
        valid_data = self.generate_unique_pairs(valid_matches, len(valid_matches))
        test_data = self.generate_unique_pairs(test_matches, len(test_matches))
        return train_data, valid_data, test_data