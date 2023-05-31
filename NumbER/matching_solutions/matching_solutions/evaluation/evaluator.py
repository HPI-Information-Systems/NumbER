import pandas as pd
import networkx as nx
import numpy as np

from NumbER.matching_solutions.utils.transitive_closure import calculate_transitive_closure

class Evaluator():
    def __init__(self, pred, gold, threshold=None):
        self.gold = calculate_transitive_closure(gold)
        self.pred = pred
        self.threshold_valid = threshold
        
    def evaluate(self):
        result_not_closed = self.calculate_metrics(self.apply_threshold(self.pred, self.threshold_valid), self.gold)
        best_threshold = self.find_best_threshold(self.pred, self.gold)[0]
        result_best_threshold = self.calculate_metrics(self.apply_threshold(self.pred, best_threshold), self.gold)
        result_closed = self.calculate_metrics(calculate_transitive_closure(self.pred), self.gold)
        return {'best_threshold': best_threshold, 'result_best_threshold': result_best_threshold, 'result_not_closed': result_not_closed, 'result_closed': result_closed}
    
    def confusion_matrix(self, pred: pd.DataFrame, gold: pd.DataFrame):
        #dataframes consist of p1,p2,prediction (label in case of goldstandard) and threshold
        gold = gold[gold['prediction'] == 1]
        pred = pred[pred['prediction'] == 1]
        predicted_pairs = set(tuple(sorted(x)) for x in pred[['p1', 'p2']].values)
        gold_standard_pairs = set(tuple(sorted(x)) for x in gold[['p1', 'p2']].values)
        true_positives = len(predicted_pairs.intersection(gold_standard_pairs))
        false_positives = len(predicted_pairs.difference(gold_standard_pairs))
        false_negatives = len(gold_standard_pairs.difference(predicted_pairs))
        return {'tp': true_positives, 'fp': false_positives, 'fn': false_negatives}
    
    def find_best_threshold(self, pred: pd.DataFrame, gold: pd.DataFrame):
        best_threshold = None
        best_f1 = 0
        l = pred.copy()
        for threshold in np.arange(0, 1, 0.01):
            metrics = self.calculate_metrics(self.apply_threshold(l, threshold), gold)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold
        return best_threshold, best_f1
    
    def calculate_metrics(self, pred: pd.DataFrame, gold: pd.DataFrame):
        confusion_matrix = self.confusion_matrix(pred, gold)
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp']) if confusion_matrix['tp'] + confusion_matrix['fp'] > 0 and confusion_matrix['tp'] > 0 else 0
        recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn']) if confusion_matrix['tp'] + confusion_matrix['fn'] > 0 and confusion_matrix['tp'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision * recall > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def apply_threshold(self, pred: pd.DataFrame, threshold: float = None):
        if threshold is not None:
            pred['prediction'] = 0
            pred.loc[pred['score'] > threshold, 'prediction'] = 1
        return pred