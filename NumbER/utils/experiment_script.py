import os
import datetime
import pandas as pd

from NumbER.matching_solutions.matching_solutions import DittoMatchingSolution
from NumbER.matching_solutions.utils.create_file_format import create_format

datasets = {
	'earthquake': {
     #dataset has to have structure: 'p1, p2'
		'records': 'NumbER/dataset_cleaning/earthquakes/earthquakes_associated_us.csv',
		'groundtruth': 'NumbER/dataset_cleaning/earthquakes/groundtruth_associated_us.csv',
	}
}

matching_solutions = {
	'ditto': {
		'class': DittoMatchingSolution,
		'train': {
			'run_id': 1,
			'batch_size': 64,
			'n_epochs': 1,
			'lr': 3e-5,
			'max_len': 256,
			'lm': 'roberta',
			'fp16': True,
		},
		'test': {
			'batch_size': 256,
			'lm': 'roberta',
			'max_len': 256,
		}
	}
}

result = []
date_time = datetime.datetime.now()
base_output_path = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/experiments_{date_time.strftime('%d/%m/%Y_%H:%M')}"

for dataset, paths in datasets.items():
    for matching_solution, config in matching_solutions.items():
        dataset_path = os.path.join(base_output_path, 'datasets', dataset)
        os.makedirs(dataset_path, exist_ok=True)
        train_formatter, valid_formatter, test_formatter = create_format(paths['records'], paths['groundtruth'], matching_solution,dataset_path)
        train_path, valid_path, test_path = train_formatter.write_to_file(os.path.join(dataset_path, 'train')), valid_formatter.write_to_file(os.path.join(dataset_path, 'valid')), test_formatter.write_to_file(os.path.join(dataset_path, 'test'))
        solution = config['class'](dataset, train_path, valid_path, test_path)
        best_f1, model, threshold, train_time = solution.model_train(**config['train'])
        prediction = solution.model_predict(**config['test'], model=model, threshold=threshold)
        matching_solution_path = os.path.join(base_output_path, 'matching_solution', matching_solution)
        os.makedirs(matching_solution_path, exist_ok=True)
        pred = pd.merge(test_formatter.goldstandard.reset_index(), prediction['predict'][0].reset_index(), left_index=True, right_index=True)[['id1', 'id2', 'match', 'match_confidence']]
        pred.to_csv(os.path.join(matching_solution_path, dataset + '.csv'), index=False, header=None)
        test_formatter.goldstandard.to_csv(os.path.join(matching_solution_path, dataset + '_goldstandard.csv'), index=False)
        result.append({
			'dataset': dataset,
			'matching_solution': matching_solution,
			'train_time': train_time,
			'f1': prediction['evaluate']
		})
result = pd.DataFrame(result)
result.to_csv('results.csv')