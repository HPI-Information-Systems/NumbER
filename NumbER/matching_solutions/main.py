from NumbER.matching_solutions.matching_solutions import DittoMatchingSolution

s = DittoMatchingSolution('earthquake', 'NumbER/matching_solutions/utils/train.txt', 'NumbER/matching_solutions/utils/valid.txt', 'NumbER/matching_solutions/utils/test.txt')
best_f1, model, threshold, train_time = s.model_train(1, 32, 256, 3e-5, 1, 'roberta', True)
print("best_f1: ", best_f1)
prediction = s.model_predict(model, 256, 'roberta', 256, threshold=threshold)

print(prediction['predict'])
print(prediction['evaluate'])