import numpy as np
from sklearn.metrics import f1_score
import networkx as nx
import pandas as pd
from networkx.algorithms.dag import transitive_closure
import os

# def calculate_f1(gt, pred):
# 	print("gt", gt)
# 	print("pred", pred)
# 	gt = gt[gt['prediction'] == 1]
# 	pred = pred[pred['prediction'] == 1]
# 	#pred['tuple'] = pred.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)#.astype(str)
# 	pred['tuple'] = pred.apply(lambda row: str(sorted((row['p1'], row['p2']))[0])+str(sorted((row['p1'], row['p2']))[1]), axis=1)#.astype(str)

# 	gt = calculate_transitive_closure(gt)
# 	gt['prediction_gt'] = gt['prediction']
# 	pred['prediction_pred'] = pred['prediction']
# 	#merged = pd.merge(gt, pred, on='tuple', how='outer')
# 	#merged = pd.concat([gt,pred], axis=1, keys=['tuple'], join="outer")
# 	#merged = pd.concat([gt, pred], axis=1, join='inner', keys=)
# 	print(merged)
# 	merged['prediction_gt'].fillna(0, inplace=True)
# 	merged['prediction_pred'].fillna(0, inplace=True)
# 	print("merged", merged)
# 	#gt = np.array(all_pairs['gt'])
# 	#pred = np.array(all_pairs['pred'])
# 	gt = np.array(merged['prediction_gt'])
# 	pred = np.array(merged['prediction_pred'])
# 	print("hh", f1_score(gt, pred))
# 	return f1_score(gt, pred)

def calculate_transitive_closure(pairs):
    pairs = pairs[pairs['prediction'] == 1]
    pairs=[tuple(sorted((pair['p1'], pair['p2']))) for _,pair in pairs.iterrows()]
    G = nx.Graph()
    G.add_edges_from(pairs)
    clusters = []
    for connected_component in nx.connected_components(G):
        clusters.append(list(connected_component))
    pairs = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                pairs.append(tuple(sorted((cluster[i], cluster[j]))))
    df = pd.DataFrame(pairs, columns=['p1', 'p2'])
    df['prediction'] = 1
    return df
    # pairs = pairs[pairs['prediction'] == 1]
    # G = nx.from_pandas_edgelist(pairs, 'p1', 'p2', create_using=nx.DiGraph())
    # G_tc = transitive_closure(G)
    # df2_tc = pd.DataFrame(list(G_tc.edges()), columns=['p1', 'p2'])
    # df2_tc['prediction'] = 1
    # return df2_tc
    
def calculate_f1(gold, pred, close_gold, close_pred, dataset=None):
    # Transitively close df2
	gold = gold[gold['prediction'] == 1]
	pred = pred[pred['prediction'] == 1]
	if close_gold:
		#base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"
		#groundtruth = pd.read_csv(os.path.join(base_path, dataset, "matches.csv"))
		gold = calculate_transitive_closure(gold)
	if close_pred:
		pred = calculate_transitive_closure(pred)
	predicted_pairs = set(tuple(sorted(x)) for x in pred[['p1', 'p2']].values)
	gold_standard_pairs = set(tuple(sorted(x)) for x in gold[['p1', 'p2']].values)
	true_positives = len(predicted_pairs & gold_standard_pairs)
	precision = true_positives / len(predicted_pairs)
	recall = true_positives / len(gold_standard_pairs)
	f1_score = 2 * (precision * recall) / (precision + recall)
	false_positives = len(predicted_pairs - gold_standard_pairs)
	false_negatives = len(gold_standard_pairs - predicted_pairs)
	return f1_score, precision, recall, true_positives, false_positives, false_negatives