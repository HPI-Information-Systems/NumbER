import torch
import pandas as pd
import deepmatcher as dm
def merge_data(labeled, table_a, table_b, output):
  merged_csv = pd.read_csv(labeled).rename(columns={'ltable_id': 'left_id', 'rtable_id': 'right_id'})
  table_a_csv = pd.read_csv(table_a)
  table_a_csv = table_a_csv.rename(columns={col: 'left_' + col for col in table_a_csv.columns})
  table_b_csv = pd.read_csv(table_b)
  table_b_csv = table_b_csv.rename(columns={col: 'right_' + col for col in table_b_csv.columns})
  merged_csv = pd.merge(merged_csv, table_a_csv, on='left_id')
  merged_csv = pd.merge(merged_csv, table_b_csv, on='right_id')
  merged_csv['id'] = merged_csv[['left_id', 'right_id']].apply(lambda row: '_'.join([str(c) for c in row]), axis=1)
  del merged_csv['left_id']
  del merged_csv['right_id']
  merged_csv.to_csv(output, index=False)
  
def main():
	merge_data(
		'data/Structured/iTunes-Amazon/train.csv', 
		'data/Structured/iTunes-Amazon/tableA.csv', 
		'data/Structured/iTunes-Amazon/tableB.csv', 
		'data/Structured/iTunes-Amazon/joined_train.csv')
	merge_data(
		'data/Structured/iTunes-Amazon/valid.csv', 
		'data/Structured/iTunes-Amazon/tableA.csv', 
		'data/Structured/iTunes-Amazon/tableB.csv', 
		'data/Structured/iTunes-Amazon/joined_valid.csv')
	merge_data(
		'data/Structured/iTunes-Amazon/test.csv', 
		'data/Structured/iTunes-Amazon/tableA.csv', 
		'data/Structured/iTunes-Amazon/tableB.csv', 
		'data/Structured/iTunes-Amazon/joined_test.csv')
	pd.read_csv('data/Structured/iTunes-Amazon/joined_train.csv').head()
	train, validation, test = dm.data.process(
		path='data/Structured/iTunes-Amazon/',
		train='joined_train.csv',
		validation='joined_valid.csv',
		test='joined_test.csv')
	model = dm.MatchingModel(attr_summarizer='hybrid')
	model.run_train(
		train,
		validation,
		epochs=10,
		batch_size=16,
		best_save_path='model.pth',
		pos_neg_ratio=3)
	print(model.run_eval(test))
	# unlabeled = dm.data.process_unlabeled(
	# path='sample_data/itunes-amazon/unlabeled.csv',
	# trained_model=model)
	#predictions = model.run_prediction(unlabeled)
	#predictions.head()

if __name__ == '__main__':
	main()