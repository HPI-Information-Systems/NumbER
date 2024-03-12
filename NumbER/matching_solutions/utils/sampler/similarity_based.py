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
        self.all_pairs = set()
        self.name = "similarity"
        print("GONNA BE SIMILAR")
        #self.distance_matrix = squareform(np.load(distance_path))
    
    def get_n_most_similar_records(self,record_id, available_records, n):
        distances = self.distance_matrix[int(record_id)]
        sorted_array = np.argsort(distances)[::-1]
        s = available_records['id'].values
        #print(n)
        ids = sorted_array[np.isin(sorted_array, s)][1:]#[1:n+1]
        #print("DGHWHG DW")
        result_set = []
        for id in ids:
            if not tuple(sorted([record_id, id])) in self.all_pairs:
                already_included = False
                for id_2 in result_set:
                    if tuple(sorted([id, id_2])) in self.all_pairs:
                        already_included = True
                if not already_included:
                    result_set.append(id)
                    self.all_pairs.add(tuple(sorted([record_id, id])))
                    for id_2 in result_set:
                        self.all_pairs.add(tuple(sorted([id, id_2])))
            if len(result_set) == n:
                break
        return result_set
        #return [index for index in sorted if index in available_records['id'].values][1:n+1]
        #return pd.concat(indices)
    def get_match_status(self,record_id_1, record_id_2):
        pair = self.goldstandard.loc[((self.goldstandard['p1'] == record_id_1) & (self.goldstandard['p2'] == record_id_2)) | ((self.goldstandard['p1'] == record_id_2) & (self.goldstandard['p2'] == record_id_1))]
        #return 1 if len(pair) > 0 else 0
        if len(pair) > 0:
            return 1
        else:
            return 0
        
    def sample(self, *args):
        config = args[0]
        print("config", config)
        self.distance_matrix = squareform(np.load(os.path.join(pathlib.Path(self.records_path).parent,'similarity_cosine.npy')))
        naive_sampler = NaiveSampler(self.records_path, self.goldstandard_path)
        print("LOading naive sampler with paths", self.records_path, self.goldstandard_path)
        train_records, valid_records, test_records, train_matches, valid_matches, test_matches = naive_sampler.sample_records(config)
        
        # path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/dm_amazon_google"
        # print("ITERATION", config["iteration"])
        # train_matches = pd.read_csv(f"{path}/samples/similarity/train_{config['iteration']}_goldstandard.csv")
        # test_matches = pd.read_csv(f"{path}/samples/similarity/test_{config['iteration']}_goldstandard.csv")
        # valid_matches = pd.read_csv(f"{path}/samples/similarity/valid_{config['iteration']}_goldstandard.csv")
        # train_matches = train_matches[train_matches['prediction'] == 1]
        # test_matches = test_matches[test_matches['prediction'] == 1]
        # valid_matches = valid_matches[valid_matches['prediction'] == 1]
        # train_records = pd.read_csv(f"{path}/features.csv")
        # train_records = train_records[train_records['id'].isin(train_matches['p1'].values) | train_records['id'].isin(train_matches['p2'].values)]
        # test_records = pd.read_csv(f"{path}/features.csv")
        # test_records = test_records[test_records['id'].isin(test_matches['p1'].values) | test_records['id'].isin(test_matches['p2'].values)]
        # valid_records = pd.read_csv(f"{path}/features.csv")
        # valid_records = valid_records[valid_records['id'].isin(valid_matches['p1'].values) | valid_records['id'].isin(valid_matches['p2'].values)]

        print("Sampled records naiveley", train_records, valid_records, test_records, train_matches, valid_matches, test_matches)
        self.check_no_leakage(train_matches, valid_matches, test_matches)
        print(len(train_records), len(valid_records), len(test_records), len(train_matches), len(valid_matches), len(test_matches))
        result = []
        if 'index' in train_records.columns:
            train_matches.drop(['index'], axis=1, inplace=True)
        if 'index' in valid_records.columns:
            valid_matches.drop(['index'], axis=1, inplace=True)
        if 'index' in test_records.columns:
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
            similarity_pairs = similarity_pairs[similarity_pairs['prediction'] == 0]
            all_pairs = pd.concat([similarity_pairs, matches], ignore_index=True)
            result.append(all_pairs) 
        self.check_no_leakage(result[0], result[1], result[2])
        #return result[0], result[1], result[2]#!Do this again
        return result[0], result[1], result[2]
    
    def convert_to_dataframe(self, pairs, match_status):
        result = []
        for pair, status in zip(pairs.iterrows(), match_status):
            pair = pair[1]
            result.append([pair[0], pair[1], status])
        return pd.DataFrame(result, columns=["p1", "p2", "prediction"])
        
            
        
        