import pandas as pd
import networkx as nx

def calculate_transitive_closure(pairs: pd.DataFrame):
    clusters = calculate_clusters(pairs)
    pairs = convert_to_pairs(clusters)
    return pairs

def calculate_clusters(pairs: pd.DataFrame):
    pairs = pairs[pairs['prediction'] == 1]
    pairs=[tuple(sorted((pair['p1'], pair['p2']))) for _,pair in pairs.iterrows()]
    G = nx.Graph()
    G.add_edges_from(pairs)
    clusters = []
    for connected_component in nx.connected_components(G):
        clusters.append(list(connected_component))
    return clusters


def convert_to_pairs(clusters):
    pairs = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                pairs.append(tuple(sorted((cluster[i], cluster[j]))))
    pairs = pd.DataFrame(pairs, columns=['p1', 'p2'])
    pairs['prediction'] = 1
    return pairs