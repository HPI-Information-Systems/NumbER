import pandas as pd
import deepmatcher as dm

train, validation, test = dm.data.process(path='exp_data',
    train='train.csv', validation='valid.csv', test='test.csv',left_prefix='ltable_', right_prefix='rtable_')
model = dm.MatchingModel()
model.run_train(train, validation, best_save_path='best_model.pth')
print("Training finished")
model.run_eval(test)
print("Trainin finished")

#unlabeled = dm.data.process_unlabeled(path='e/unlabeled.csv', trained_model=model)
#model.run_prediction(unlabeled)