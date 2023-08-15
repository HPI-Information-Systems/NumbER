import time
import pandas as pd
import deepmatcher as dm
import torch
import random
import numpy as np
from pathlib import Path
import pandas as pd
import os
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

class DeepMatcherMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
    def model_train(self, epochs, batch_size, pos_neg_ratio, wandb_id, seed, i):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        assert str(Path(self.train_path).parent.absolute()) == str(Path(self.valid_path).parent.absolute()) == str(Path(self.test_path).parent.absolute())
        main_path = Path(self.train_path).parent.absolute()
        print("main", main_path)
        model = dm.MatchingModel()
        train, validation, self.test = dm.data.process(path=main_path, train=f'deep_matcher_train_{i}.csv', validation=f'deep_matcher_valid_{i}.csv', test=f'deep_matcher_test_{i}.csv', label_attr='prediction')
        start_time = time.time()
        best_f1 = model.run_train(
			train,
			validation,
			epochs=epochs,
			batch_size=batch_size,
            best_save_path=os.path.join(main_path,f"deepmatcher_{i}.pth"),#"/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/garbage/best_model.pth",
			pos_neg_ratio=pos_neg_ratio)
        return best_f1, model, None, time.time() - start_time
        
    def model_predict(self, model):
        f1 = model.run_eval(self.test).item()
        test_file = pd.read_csv(self.test_path)
        goldstandard = test_file
        match_status = goldstandard['prediction']
        #test_file.rename(columns={'id': '_id'},
        #  inplace=True, errors='raise')
        test_file.drop(columns=['prediction'], inplace=True)
        test_file.to_csv(f"{str(self.test_path)[:-4]}_withoutpred.csv", index=False)
        unlabeled = dm.data.process_unlabeled(
    		path=f"{str(self.test_path)[:-4]}_withoutpred.csv",
    		trained_model=model)
        predictions = model.run_prediction(unlabeled, device='cuda')
        predictions['label'] = 0
        predictions.loc[predictions['match_score']>0.5, 'label'] = 1
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        true_positives = 0
        for i in range(len(predictions)):
            if predictions.iloc[i]['label'] == 1:
                if match_status[i] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if match_status[i] == 1:
                    false_negatives += 1
                else:
                    true_negatives += 1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 and true_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 and true_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0
        #result = pd.concat([predictions, goldstandard], axis=1)[['label', 'match_score']]
        #print("result", result)
        predictions.rename(columns={'label': 'prediction', 'match_score': 'score'}, inplace=True, errors='raise')
        print("prediction", predictions)
        #result['p1'] = np.nan
        #result['p2'] = np.nan
        return {'predict': [predictions], 'evaluate': f1}