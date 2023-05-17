import time
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from NumbER.matching_solutions.md2m.table import Table
from NumbER.matching_solutions.md2m.column import Column
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution
import networkx as nx

class MD2MMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
    def model_train(self):
        
        return None, None, None, 0
        
    def model_predict(self, test_records_path, valid_records_path, model):
        f1 = 0
        valid_data = pd.read_csv(valid_records_path)
        test_data = pd.read_csv(test_records_path)
        gold_valid = pd.read_csv(self.valid_path)
        gold_test = pd.read_csv(self.test_path)
        columns = []
        for col in valid_data:
            try:
                valid_data[col].astype(float)
            except Exception as e:
                continue
            columns.append(Column(col, valid_data[col]))
        val_similarity = Table("Table1", columns, valid_data).calculate_candidates()#['sim']
        threshold = self.calculate_threshold(val_similarity, 'tune', gold_valid)
        columns = []
        for col in test_data:
            try:
                test_data[col].astype(float)
            except Exception as e:
                continue
            columns.append(Column(col, valid_data[col]))
        test_similarity = Table("Table1", columns, valid_data).calculate_candidates()
        test_similarity['prediction'] = 0
        test_similarity.loc[test_similarity['sim'] > threshold, 'prediction'] = 1
        self.calculate_f1(gold_test, test_similarity)
        print("F1", f1)
        #pred = [1 if p['sim'] > threshold else 0 for _,p in test_similarity.iterrows()]
        #print(f1_score(gold_test, test_similarity['prediction'].to_values()))
        #matches = test_similarity[test_similarity['sim'] >= threshold]
        #test_file = pd.read_csv(self.test_path)
        return {'predict': test_similarity['prediction'].to_values(), 'evaluate': f1}
    
    def calculate_f1(self, gold, pred):
        matches = set()
        pred['tuple'] = pred.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        gt_matches = self.transitively_close(gold)
        #gold['tuple'] = gold.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        
        matches = set(pred[pred['prediction'] == 1]['tuple'])
        #gt_matches = set(gold[gold['prediction'] == 1]['tuple'])
        
        true_positives = len(matches.intersection(gt_matches))
        false_positives = len(matches.difference(gt_matches))
        false_negatives = len(gt_matches.difference(matches))
        print("tp", true_positives, "fp", false_positives, "fn", false_negatives)
        if true_positives == 0 or true_positives + false_positives == 0 or true_positives + false_negatives == 0:
            return 0
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("F1", f1, "precision", precision, "recall", recall)
        return f1
    
    def calculate_threshold(self, result: pd.DataFrame, strategy, gold=None):
        thresholds = result['sim']
        if strategy == 'naive':
            return 0.5
        elif strategy == 'min':
            return (thresholds.max() - thresholds.min()) / 2
        elif strategy == 'tune':
            best_th = 0.5
            f1 = 0.0 # metrics.f1_score(all_y, all_p)
            step_size = (thresholds.max() - thresholds.min()) / 20
            print("min", thresholds.min(), "max", thresholds.max(), "step", step_size)
            for th in np.arange(thresholds.min(), thresholds.max(), step_size):
                #pred = [1 if p > th else 0 for p in thresholds]
                result['prediction'] = 0
                result.loc[result['sim'] > th, 'prediction'] = 1
                #new_f1 = f1_score(gold, pred)
                new_f1 = self.calculate_f1(gold, result)
                if new_f1 > f1:
                    f1 = new_f1
                    best_th = th
            print("Best threshold: ", best_th)
            return best_th
        
    def transitively_close(self, pairs):
        pairs = pairs[pairs['prediction'] == 1]
        G = nx.from_pandas_edgelist(pairs, 'p1', 'p2', create_using=nx.DiGraph())
        G_tc = nx.transitive_closure(G)
        df_tc = pd.DataFrame(list(G_tc.edges()), columns=['p1', 'p2'])
        df_tc['tuple'] = df_tc.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        list_of_ids = list(set(df_tc['p1'].unique()).union(set(df_tc['p2'].unique())))
        reflexive = [tuple(sorted((val, val))) for val in list_of_ids]
        print("Reflexive", reflexive)
        df_tc = set(df_tc['tuple']).union(set(reflexive))
        return df_tc
        