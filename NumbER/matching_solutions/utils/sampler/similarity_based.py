import numpy as np
from NumbER.matching_solutions.utils.sampler.base import BaseSampler
from NumbER.matching_solutions.utils.sampler.naive import NaiveSampler
from scipy.spatial.distance import squareform
import pathlib
import pandas as pd
from tqdm import tqdm
import os

class SimilarityBasedSampler(BaseSampler):
    def __init__(self, records_path, goldstandard_path):
        super().__init__(records_path, goldstandard_path)
        print("Doing similarity based samplign")
        self.records_path = records_path
        print("records path", self.records_path)
        print("records similarity", self.records)
        self.goldstandard_path = goldstandard_path
        print("goldstandard path", self.goldstandard_path)
        print("goldstandard similarity", self.goldstandard)
        #self.distance_matrix = squareform(np.load(distance_path))
    
    def get_n_most_similar_records(self,record_id, available_records, n):
        distances = self.distance_matrix[int(record_id)]
        sorted = np.argsort(distances)[::-1]
        s = available_records['id'].values
        #print(n)
        return sorted[np.isin(sorted, s)][1:n+1]
        #return [index for index in sorted if index in available_records['id'].values][1:n+1]
        #return pd.concat(indices)
    def get_match_status(self,record_id_1, record_id_2):
        pair = self.goldstandard.loc[((self.goldstandard['p1'] == record_id_1) & (self.goldstandard['p2'] == record_id_2)) | ((self.goldstandard['p1'] == record_id_2) & (self.goldstandard['p2'] == record_id_1))]
        if len(pair) > 0:
            return 1
        else:
            return 0

    def sample(self, *args):
        config = args[0]
        print("config", config)
        self.distance_matrix = squareform(np.load(os.path.join(pathlib.Path(self.records_path).parent,'similarity.npy')))
        naive_sampler = NaiveSampler(self.records_path, self.goldstandard_path)
        print("LOading naive sampler with paths", self.records_path, self.goldstandard_path)
        train_records, valid_records, test_records, train_matches, valid_matches, test_matches = naive_sampler.sample_records(config)
        print("Sampled records naiveley", train_records, valid_records, test_records, train_matches, valid_matches, test_matches)
        self.check_no_leakage(train_matches, valid_matches, test_matches)
        print(len(train_records), len(valid_records), len(test_records), len(train_matches), len(valid_matches), len(test_matches))
        result = []
        train_matches.drop(['index'], axis=1, inplace=True)
        valid_matches.drop(['index'], axis=1, inplace=True)
        test_matches.drop(['index'], axis=1, inplace=True)
        for records, matches in [[train_records, train_matches], [valid_records, valid_matches], [test_records, test_matches]]:
            clusterings = []
            for _,record in tqdm(records.iterrows()):
                most_similar = [*self.get_n_most_similar_records(record['id'], records, config['n_most_similar']), record['id']]
                clusterings.append(most_similar)
            match_status = []
            subset_pairs = self.convert_to_pairs(clusterings)
            for _,pair in subset_pairs.iterrows():
                match_status.append(self.get_match_status(pair['p1'], pair['p2']))
            similarity_pairs = self.convert_to_dataframe(subset_pairs, match_status)
            all_pairs = pd.concat([similarity_pairs, matches], ignore_index=True)
            all_pairs.drop_duplicates(inplace=True)
            result.append(all_pairs) 
        print(len(result[0]), len(result[1]), len(result[2]))
        self.check_no_leakage(result[0], result[1], result[2])
        return result[0], result[1], result[2]
    
    def convert_to_dataframe(self, pairs, match_status):
        result = []
        for pair, status in zip(pairs.iterrows(), match_status):
            pair = pair[1]
            result.append([pair[0], pair[1], status])
        return pd.DataFrame(result, columns=["p1", "p2", "prediction"])
        
            
        
        