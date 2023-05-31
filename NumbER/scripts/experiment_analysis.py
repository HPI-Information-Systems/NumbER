import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from NumbER.matching_solutions.utils.calculate_f1 import calculate_f1#calculate_f1, calculate_f2
import glob
def calculate_quality_metrics_no_closure(gt, pred):
    f1 = f1_score(gt['prediction'], pred['prediction'])
    true_positives = len(pred[pred['prediction'] == 1][pred['prediction'] == gt['prediction']])
    false_positives = len(pred[pred['prediction'] == 1][pred['prediction'] != gt['prediction']])
    false_negatives = len(pred[pred['prediction'] == 0][pred['prediction'] != gt['prediction']])
    prec = precision_score(gt['prediction'], pred['prediction'])
    rec = recall_score(gt['prediction'], pred['prediction'])
    return f1, prec, rec, true_positives, false_positives, false_negatives

# def transitively_close(matches):
#     matches = matches[matches['prediction'] == 1]
#     matches = matches[['p1', 'p2']].to_numpy()
#     G = nx.Graph()
#     G.add_edges_from(matches)
#     clusters = []
#     for connected_component in nx.connected_components(G):
#         clusters.append(list(connected_component))
#     pairs = set()
#     for cluster in clusters:
#         for i in range(len(cluster)):
#             for j in range(i+1, len(cluster)):
#                 pairs.add(tuple(sorted((cluster[i], cluster[j]))))
#     return pairs
def transitively_close(pairs):
    pairs = pairs[pairs['prediction'] == 1]
    G = nx.from_pandas_edgelist(pairs, 'p1', 'p2', create_using=nx.DiGraph())
    G_tc = nx.transitive_closure(G)
    df_tc = pd.DataFrame(list(G_tc.edges()), columns=['p1', 'p2'])
    #df_tc = df_tc.query('p1 != p2')
    df_tc['tuple'] = df_tc.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
    return set(df_tc['tuple'])

def calculate_transitively_closed_f1(gt, matches: set):
    gt = transitively_close(gt)
    true_positives = len(matches.intersection(gt))
    false_positives = len(matches.difference(gt))
    false_negatives = len(gt.difference(matches))
    tp = pd.DataFrame(list(matches.intersection(gt)))
    fp = pd.DataFrame(list(matches.difference(gt)))
    fn = pd.DataFrame(list(gt.difference(matches)))
    #print("False Positives", matches.difference(gt))
    #print("False Negatives", gt.difference(matches))
    print("TP", true_positives, "FP", false_positives, "FN", false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    print("F1-Score", f1)
    print("Precision", precision)
    print("Recall", recall)
    print("2", 2*true_positives / (2*true_positives + false_positives + false_negatives))
	
def process_experiment_files(goldstandard_file, experiment_file, experiment, dataset_name):
    gt = pd.read_csv(os.path.join(base_path, goldstandard_file))
    all_ids = set(gt['p1'].to_list()).union(set(gt['p2'].to_list()))
    gt_original = pd.read_csv(os.path.join("/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets", dataset_name, "matches.csv"))
    gt = gt_original[gt_original['p1'].isin(all_ids) & gt_original['p2'].isin(all_ids)]
    #gt_original['tuple'] = gt_original.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
    
    pred = pd.read_csv(os.path.join(base_path, experiment_file))
    #print(calculate_quality_metrics_no_closure(gt, pred))
    result = []
    for (close_gold, close_pred) in [(False, False), (True, False), (True, True)]:
        f1, precision, recall, true_positives, false_positives, false_negatives = calculate_f1(gt, pred, close_gold=close_gold, close_pred=close_pred, dataset=dataset_name)
        print("Gold Transitive: ", close_gold, " Pred Transitive: ", close_pred)
        print("F1-Score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("True Positives: ", true_positives)
        print("False Positives: ", false_positives)
        print("False Negatives: ", false_negatives)
        result.append({
            'experiment': experiment,
            'dataset': dataset_name,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'close_gold': close_gold,
            'close_pred': close_pred
        })
    return result
        
if __name__ == '__main__':
    experiments = {
        "experiments_test_19_05_2023_19:49:50": ["baby_products_numeric"],
        # "experiments_sorted_neighbourhood_9_16_05_2023_19:40:07": ["books3_numeric"],
        # "experiments_similarity_based_5_16_05_2023_07:04:35": ["books3_numeric_no_isbn"],
        # "experiments_naive_sampling_7_16_05_2023_23:36:38":
        #     [
        #         "baby_products_numeric", "books3_numeric", "books3_numeric_no_isbn", "earthquakes", "vsx", "x2_numeric", "x3_numeric"
        #     ]
    }
    result = []
    for experiment, datasets in experiments.items():
        base_path = os.path.join("/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments", experiment, "matching_solution/ditto")
        for dataset_name in datasets:
            print("Dataset: ", dataset_name)
            try:
                res = process_experiment_files(os.path.join(base_path, f"{dataset_name}_goldstandard_0.csv"), os.path.join(base_path, f"{dataset_name}_0.csv"), experiment, dataset_name)
                for el in res:
                    result.append(el)
            except Exception as e:
                print("Error", e)
    df = pd.DataFrame(result)
    df.to_csv("results.csv", index=False)

        
        #calculate_f1(gt, pred)
        # pred['tuple'] = pred.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        # pred_set = set(pred[pred['prediction'] == 1]['tuple'])
        # #calculate_transitively_closed_f1(gt, pred_set)
        # pred_set = transitively_close(pred)
        # all_ids = list(set(gt['p1'].to_list() + gt['p2'].to_list()))
        # pred_set = pred_set.union(set([(i, i) for i in all_ids]))
        # calculate_transitively_closed_f1(gt, pred_set)
        #pred['tuple'] = pred.apply(lambda row: tuple(sorted((row['p1'], row['p2']))), axis=1)
        #pred_set = set(pred[pred['prediction'] == 1]['tuple'])
        