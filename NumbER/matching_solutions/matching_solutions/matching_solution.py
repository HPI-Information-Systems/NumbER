from abc import ABC

class MatchingSolution:
    def __init__(self, dataset_name, train_path, valid_path, test_path):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

    def model_train(self):
        pass
    def model_predict(self, output_path):
        pass
