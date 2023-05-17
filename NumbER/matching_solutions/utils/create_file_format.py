import pandas as pd
import random
import networkx as nx
import os
from tqdm import tqdm
from NumbER.matching_solutions.utils.output_formats import OutputFormat, DittoFormat, DeepMatcherFormat
from NumbER.matching_solutions.utils.sampling import sorted_neighbourhood, map_to_groundtruth
def create_magellan_format(records, goldstandard):
    pass

def generate_unique_pairs(input_pairs, output_length):
    amount_of_possible_non_matches = len(input_pairs) * (len(input_pairs) - 1) / 2
    output_length = min(output_length, amount_of_possible_non_matches)
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
    s = pd.concat([input_pairs, df])
    return s

def create_clusters(groundtruth):
    groundtruth = groundtruth[['p1', 'p2']].to_numpy()
    G = nx.Graph()
    G.add_edges_from(groundtruth)
    clusters = []
    for connected_component in nx.connected_components(G):
        clusters.append(list(connected_component))
    return clusters

def convert_to_pairs(clusters):
    pairs = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                pairs.append((cluster[i], cluster[j]))
    return pd.DataFrame(pairs, columns=['p1', 'p2'])

def sample_record_based(groundtruth, records, attributes,  train_fraction, valid_fraction, test_fraction, train_window_size,test_window_size):
    assert train_fraction + valid_fraction + test_fraction == 1
    #records = records.sample(frac=1).reset_index(drop=True)
    if "prediction" in groundtruth.columns:
        groundtruth = groundtruth[groundtruth['prediction'] == 1]
    #clusters = create_clusters(groundtruth)
    train_records, valid_records, test_records, train_matches, valid_matches, test_matches = sample_data(groundtruth, train_fraction, valid_fraction, test_fraction, records)
    train_matches = set(tuple(sorted(pair)) for pair in train_matches[['p1', 'p2']].to_numpy())
    valid_matches = set(tuple(sorted(pair)) for pair in valid_matches[['p1', 'p2']].to_numpy())
    test_matches = set(tuple(sorted(pair)) for pair in test_matches[['p1', 'p2']].to_numpy())
    #train_records = records[:int(len(records)*train_fraction)]#.reset_index(drop=True)
    #valid_records = records[int(len(records)*train_fraction):int(len(records)*(train_fraction+valid_fraction))]#.reset_index(drop=True)
    #test_records = records[int(len(records)*(train_fraction+valid_fraction)):]#.reset_index(drop=True)
    train_data = create_pairs(train_records, attributes, 'id', train_window_size, groundtruth, train_matches)
    valid_data = create_pairs(valid_records, attributes, 'id', test_window_size, groundtruth, valid_matches)
    test_data = create_pairs(test_records, attributes, 'id', test_window_size, groundtruth, test_matches)
    return train_data, valid_data, test_data

def create_pairs(data, attributes, id_column, size, matches, additional_pairs=None):
    pairs = sorted_neighbourhood(data, attributes, id_column, size)
    if additional_pairs is not None:
        pairs.update(additional_pairs)
    l = map_to_groundtruth(pairs, matches)
    pairs = pd.DataFrame(pairs, columns=['p1', 'p2'])
    pairs['prediction'] = l
    return pairs

def sample_data(groundtruth, train_fraction, valid_fraction, test_fraction, records):
    assert train_fraction + valid_fraction + test_fraction == 1
    groundtruth = groundtruth.sample(frac=1).reset_index(drop=True)
    if "prediction" in groundtruth.columns:
        groundtruth = groundtruth[groundtruth['prediction'] == 1]
    clusters = create_clusters(groundtruth)
    train_matches = convert_to_pairs(clusters[:int(len(clusters)*train_fraction)])#wenn ein duplikatcluster geteilt wird, kann ein record in mehreren dateien vorkommen
    valid_matches = convert_to_pairs(clusters[int(len(clusters)*train_fraction):int(len(clusters)*(train_fraction+valid_fraction))])
    test_matches = convert_to_pairs(clusters[int(len(clusters)*(train_fraction+valid_fraction)):])
    train_matches['prediction'] = 1
    valid_matches['prediction'] = 1
    test_matches['prediction'] = 1
    train_ids = set(train_matches['p1'].unique() + train_matches['p2'].unique())
    valid_ids = set(valid_matches['p1'].unique() + valid_matches['p2'].unique())
    test_ids = set(test_matches['p1'].unique() + test_matches['p2'].unique())
    train_records = records.loc[records['id'].isin(list(train_ids))]
    valid_records = records.loc[records['id'].isin(list(valid_ids))]
    test_records = records.loc[records['id'].isin(list(test_ids))]
    return train_records, valid_records, test_records, train_matches, valid_matches, test_matches
    #train_data = generate_unique_pairs(train_matches, len(train_matches))
    #valid_data = generate_unique_pairs(valid_matches, len(valid_matches))
    #test_data = generate_unique_pairs(test_matches, len(test_matches))
    #return train_data, valid_data, test_data

def create_format(path_to_records, path_to_groundtruth, output_format, attributes, train_window_size, test_window_size):
    groundtruth = pd.read_csv(path_to_groundtruth, index_col=None)
    records = pd.read_csv(path_to_records, index_col=None)
    records = records.replace({'\t': ''}, regex=True)
    train_pairs, valid_pairs, test_pairs = sample_record_based(groundtruth, records, attributes, 0.5, 0.25, 0.25, train_window_size, test_window_size)
    print("OUTPUTFORMAT: ", output_format)
    Formatter = DittoFormat if output_format == 'ditto' else DeepMatcherFormat
    return Formatter(train_pairs, records), Formatter(valid_pairs, records), Formatter(test_pairs, records)
    
if __name__ == '__main__':
    base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/"
    create_format(base_path+"x2_all" + "/features.csv", base_path+"x2_all" + "/matches.csv", "ditto", "x2_all")
    