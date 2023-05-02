import time
import pandas as pd
import deepmatcher as dm
from pathlib import Path

from NumbER.matching_solutions.matching_solutions.matching_solution import MatchingSolution

class DeepMatcherMatchingSolution(MatchingSolution):
    def __init__(self, dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path):
        super().__init__(dataset_name, train_dataset_path, valid_dataset_path, test_dataset_path)
        
    def model_train(self, epochs, batch_size, pos_neg_ratio):
        model = dm.MatchingModel()
        assert str(Path(self.train_path).parent.absolute()) == str(Path(self.valid_path).parent.absolute()) == str(Path(self.test_path).parent.absolute())
        main_path = Path(self.train_path).parent.absolute()
        train, validation, self.test = dm.data.process(path=main_path, train='train.csv', validation='valid.csv', test='test.csv', label_attr='prediction')
        start_time = time.time()
        best_f1 = model.run_train(
			train,
			validation,
			epochs=epochs,
			batch_size=batch_size,
            best_save_path="/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/garbage/best_model.pth",
			pos_neg_ratio=pos_neg_ratio)
        return best_f1, model, None, time.time() - start_time
        
    def model_predict(self, model):
        f1 = model.run_eval(self.test).item()
        test_file = pd.read_csv(self.test_path)
        #test_file.rename(columns={'id': '_id'},
        #  inplace=True, errors='raise')
        test_file.drop(columns=['prediction'], inplace=True)
        test_file.to_csv(self.test_path, index=False)
        unlabeled = dm.data.process_unlabeled(
    		path=self.test_path,
    		trained_model=model)
        predictions = model.run_prediction(unlabeled, device='cuda')
        print(predictions)
        return {'predict': predictions, 'evaluate': f1}