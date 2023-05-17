import pandas as pd
import networkx as nx

class Evaluator():
    def __init__(self, pred, gold):
        pass
    
    def confusion_matrix(self, pred: pd.DataFrame, gold: pd.DataFrame):
        #dataframes consist of p1,p2,prediction (label in case of goldstandard) and threshold
        gold = gold[gold['prediction'] == 1]
        predicted_pairs = set(tuple(sorted(x)) for x in pred[['p1', 'p2']].values)
        gold_standard_pairs = set(tuple(sorted(x)) for x in gold[['p1', 'p2']].values)
        true_positives = len(predicted_pairs.intersection(gold_standard_pairs))
        false_positives = len(predicted_pairs.difference(gold_standard_pairs))
        false_negatives = len(gold_standard_pairs.difference(predicted_pairs))
        return {'tp': true_positives, 'fp': false_positives, 'fn': false_negatives}
    
    def calculate_metrics(self, pred: pd.DataFrame, gold: pd.DataFrame):
        confusion_matrix = self.confusion_matrix(pred, gold)
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'])
        recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'])
        f1 = 2 * (precision * recall) / (precision + recall)
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def apply_threshold(pred: pd.DataFrame, threshold: float):
        pred['prediction'] = 0
        pred.loc[pred['sim'] > threshold, 'prediction'] = 1
        return pred
    
    def calculate_transitive_closure(self, pairs: pd.DataFrame):
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
        pairs = pd.DataFrame(pairs, columns=['p1', 'p2'])
        pairs['prediction'] = 1
        return pairs