import time
import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
import NumbER.matching_solutions.matching_solutions.xgboost_old as xgb
from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

class XGBoostMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
    def model_train(self, early_stopping_rounds, i):
        print("trainus path", self.train_path, "valid path", self.valid_path, "test path", self.test_path)
        assert str(Path(self.train_path).parent.absolute()) == str(Path(self.valid_path).parent.absolute()) == str(Path(self.test_path).parent.absolute())
        main_path = Path(self.train_path).parent.absolute()
        print("main", main_path)
        train_data = pd.read_csv(self.train_path)
        train_y = train_data['prediction']
        train_data.drop(columns=['prediction'], inplace=True)
        valid_data = pd.read_csv(self.valid_path)
        valid_y = valid_data['prediction']
        valid_data.drop(columns=['prediction'], inplace=True)
        dtrain = xgb.DMatrix(train_data, label=train_y, enable_categorical=True)
        dval = xgb.DMatrix(valid_data, label=valid_y, enable_categorical=True)
        watchlist = [(dtrain, 'train'), (dval, 'validation')]
        params = {
			'objective': 'binary:logistic',
			'eval_metric': 'logloss',
			'max_depth': 3,
			'learning_rate': 0.1,
			'n_estimators': 100
		}
        #xgb_model = xgb.XGBClassifier(random_state=42, enable_categorical=True) 
        start_time = time.time()
        xgb_model = xgb.train(params, dtrain, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
        #xgb_model.fit(train_data, train_y, early_stopping_rounds=early_stopping_rounds, eval_set=[(valid_data, valid_y)])
        return None, xgb_model, None, time.time() - start_time
        
    def model_predict(self, model):
        test_file = pd.read_csv(self.test_path)
        goldstandard = test_file
        match_status = goldstandard['prediction']
        test_file.drop(columns=['prediction'], inplace=True)
        test_file.to_csv(self.test_path, index=False)
        dtest = xgb.DMatrix(test_file, enable_categorical=True)
        output = model.predict(dtest)
        scores = model.predict(dtest, output_margin=True)
        # output = model.predict(test_file)
        # scores = model.predict_proba(test_file)[:, 1]
        output = pd.DataFrame({
        	'prediction': output,
        	'score': scores
    	})
        predictions = output#[output['prediction'] == 1]
        print("predictions", predictions)
        #predictions['label'] = 0
        #predictions.loc[predictions['match_score']>0.5, 'label'] = 1 #gucken ob das richtig ist, aber schein zu stimmen
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        true_positives = 0
        for i in range(len(predictions)):
            if predictions.iloc[i]['prediction'] == 1:
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
        print("prediction", predictions)
        #result = pd.concat([predictions, goldstandard], axis=1)[['label', 'match_score']]
        #print("result", result)
        #predictions.rename(columns={'label': 'prediction', 'match_score': 'score'}, inplace=True, errors='raise')
        #result['p1'] = np.nan
        #result['p2'] = np.nan
        return {'predict': [output], 'evaluate': f1}