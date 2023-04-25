import pandas as pd
import random
import networkx as nx
import os
from tqdm import tqdm
from abc import ABC

class OutputFormat(ABC):
    def __init__(self, goldstandard, records):
        self.records = records
        self.goldstandard = goldstandard
    def create_output(self):
        raise NotImplementedError
    def write_to_file(self, filepath):
        if os.path.exists(filepath):
            raise FileExistsError
class DittoFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
           #goldstandard: pair1, pair2, prediction
        result = []
        for _, pair in tqdm(self.goldstandard.iterrows()):
            temp=""
            record1 = self.records.loc[pair['p1']]
            record2 = self.records.loc[pair['p2']]
            for col, val in record1.iteritems():
                temp += f" COL {col} VAL {val}"
            temp += "\t"
            for col, val in record2.iteritems():
                temp += f" COL {col} VAL {val}"
            temp += "\t"
            temp += str(pair['prediction'])
            result.append(temp)
        return result
    def write_to_file(self, filepath):
        filepath = f"{filepath}.txt" if filepath[-4:] != ".txt" else filepath
        try:
            super().write_to_file(filepath)
        except FileExistsError:
            return filepath
        with open(filepath, "w") as file:
            for element in self.create_output():
                file.write(element + "\n")
        return filepath

class DeepMatcherFormat(OutputFormat):
    def __init__(self, goldstandard: pd.DataFrame, records: pd.DataFrame):
        self.goldstandard = goldstandard
        self.records = records
        
    def create_output(self):
           #goldstandard: pair1, pair2, prediction
        left_columns = [f"left_{column}" for column in self.records.columns]
        right_columns = [f"right_{column}" for column in self.records.columns]
        result = []
        for _, pair in tqdm(self.goldstandard.iterrows()):
            result.append([pair['prediction']] + list(self.records.loc[pair['p1']]) + list(self.records.loc[pair['p2']]))
        df = pd.DataFrame(result, columns=['prediction'] + left_columns + right_columns)
        df['id'] = df.index
        return df
    
    def write_to_file(self, filepath):
        try:
            super().write_to_file(filepath)
        except FileExistsError:
            return filepath
        filepath = f"{filepath}.csv" if filepath[-4:] != ".csv" else filepath
        self.create_output().to_csv(filepath, index=False)
        return filepath
        

def create_magellan_format(records, goldstandard):
    pass

def generate_unique_pairs(input_pairs, output_length):
    
    input_pairs_set = set(tuple(sorted(pair)) for _,pair in input_pairs.iterrows())
    ids = list(set(id for _,pair in input_pairs.iterrows() for id in pair))
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

def create_clusters(groundtruth):
    groundtruth = groundtruth[['p1', 'p2']].to_numpy()
    G = nx.Graph()
    G.add_edges_from(groundtruth)
    clusters = []
    for connected_component in nx.connected_components(G):
        clusters.append(list(connected_component))
    return clusters

def convert_to_pairs(clusters):
    #out of a list of clusters create a list of pairs
    pairs = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                pairs.append((cluster[i], cluster[j]))
    return pd.DataFrame(pairs, columns=['p1', 'p2'])

def sample_data(groundtruth, train_fraction, valid_fraction, test_fraction):
    assert train_fraction + valid_fraction + test_fraction == 1
    groundtruth = groundtruth.sample(frac=1).reset_index(drop=True)
    clusters = create_clusters(groundtruth)
    train_matches = convert_to_pairs(clusters[:int(len(clusters)*train_fraction)])#wenn ein duplikatcluster geteilt wird, kann ein record in mehreren dateien vorkommen
    valid_matches = convert_to_pairs(clusters[int(len(clusters)*train_fraction):int(len(clusters)*(train_fraction+valid_fraction))])
    test_matches = convert_to_pairs(clusters[int(len(clusters)*(train_fraction+valid_fraction)):])
    train_matches['prediction'] = 1
    valid_matches['prediction'] = 1
    test_matches['prediction'] = 1
    train_data = generate_unique_pairs(train_matches, len(groundtruth))
    valid_data = generate_unique_pairs(valid_matches, len(groundtruth))
    test_data = generate_unique_pairs(test_matches, len(groundtruth))
    return train_data, valid_data, test_data
    
def create_format(path_to_records, path_to_groundtruth, output_format, output_folder):
    groundtruth = pd.read_csv(path_to_groundtruth, index_col=None)
    records = pd.read_csv(path_to_records, index_col=None)
    train_data, valid_data, test_data = sample_data(groundtruth, 0.7, 0.15, 0.15)
    Formatter = DittoFormat if output_format == 'ditto' else DeepMatcherFormat
    return Formatter(train_data, records), Formatter(valid_data, records), Formatter(test_data, records)
    train_path = Formatter(train_data, records)#.write_to_file("train")
    valid_path = Formatter(valid_data, records)#.write_to_file("valid")
    test_path = Formatter(test_data, records)#.write_to_file("test")
    