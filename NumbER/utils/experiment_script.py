import os
import datetime
import pandas as pd
import shutil
import argparse
import importlib
import sys
import statistics
import traceback
import wandb


#from NumbER.matching_solutions.matching_solutions.ditto import DittoMatchingSolution
#, DeepMatcherMatchingSolution
from NumbER.matching_solutions.utils.create_file_format import create_format
from NumbER.utils.experiment_config import experiment_configs

def main(matching_solution, dataset_config, use_wandb):
	result = []
	date_time = datetime.datetime.now()
	base_input_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"
	base_output_path = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/experiments_{date_time.strftime('%d_%m_%Y_%H:%M')}"
	experiment_config = experiment_configs[dataset_config]
	

	for dataset, paths in experiment_config.items():
		results_f1 = []
		times = []
		for i in range(3):
			wandb.init(
				project="NumbER",
    			entity="Lasklu",
       			mode="enabled" if use_wandb else "disabled",
				config={
					**paths['config'][matching_solution]["train"],
					'dataset': dataset,
					'matching_solution': matching_solution,
					'run': i
				},
			)
			try: 
				print("DOING it for ", dataset, file=sys.stdout)
				#for matching_solution, config in matching_solutions.items():
				config = paths['config'][matching_solution]
				if matching_solution == "ditto":
					module = importlib.import_module("NumbER.matching_solutions.matching_solutions.ditto")
					algorithm = getattr(module, 'DittoMatchingSolution')
				elif matching_solution == "deep_matcher":
					module = importlib.import_module("NumbER.matching_solutions.matching_solutions.deep_matcher")
					algorithm = getattr(module, 'DeepMatcherMatchingSolution')
				else:
					module = None
					#exit(-1)
				dataset_path = os.path.join(base_output_path, 'datasets', dataset)
				os.makedirs(dataset_path, exist_ok=True)
				record_path = os.path.join(base_input_path, dataset, 'features.csv')
				matches_path = os.path.join(base_input_path, dataset, 'matches.csv')
				print("Loding data", file=sys.stdout)
				print(paths['blocking_attributes'])
				train_formatter, valid_formatter, test_formatter = create_format(record_path, matches_path, matching_solution, paths['blocking_attributes'])
				pd.read_csv(record_path).replace({"\t", ""}, regex=True).to_csv(os.path.join(dataset_path, 'dataset.csv'), index=False)
				print("3")
				train_path, valid_path, test_path = train_formatter.write_to_file(os.path.join(dataset_path, 'train')), valid_formatter.write_to_file(os.path.join(dataset_path, 'valid')), test_formatter.write_to_file(os.path.join(dataset_path, 'test'))
				print("4")
				solution = algorithm(dataset, train_path, valid_path, test_path)
				try: 
					print("TRaining model", file=sys.stdout)
					best_f1, model, threshold, train_time = solution.model_train(**config['train'])
					print("Predicting", file=sys.stdout)
					prediction = solution.model_predict(**config['test'], model=model)
					wandb.log({"f1": prediction['evaluate'], "time": train_time})
				except Exception as e:
					print(traceback.format_exc(), file=sys.stdout)
					print(f"An error occured in train/predict: {e}", file=sys.stdout)
					wandb.finish(1)
					continue
				matching_solution_path = os.path.join(base_output_path, 'matching_solution', matching_solution)
				os.makedirs(matching_solution_path, exist_ok=True)
				pred = pd.merge(test_formatter.goldstandard.reset_index(), prediction['predict'][0].reset_index(), left_index=True, right_index=True)[['p1', 'p2', 'match', 'match_confidence']]
				pred.rename(columns={'match': 'prediction'},
				inplace=True, errors='raise')
				pred.to_csv(os.path.join(matching_solution_path, dataset + '.csv'), index=False)
				test_formatter.goldstandard.to_csv(os.path.join(matching_solution_path, dataset + '_goldstandard.csv'), index=False)
				results_f1.append(prediction['evaluate'])
				times.append(train_time)
				print(f"On dataset {dataset} with matching solution {matching_solution} in time {train_time} scored {prediction['evaluate']} in iteration {i}", file=sys.stdout)
				shutil.make_archive('coded', 'zip', base_output_path)
				wandb.finish()
			except Exception as e:
				print(traceback.format_exc(), file=sys.stdout)
				print(f"An error occurec: {e}", file=sys.stdout)
				wandb.finish(1)
				continue
		if len(times) == 0 or len(results_f1) == 0:
			result.append({
				'dataset': dataset,
				'matching_solution': matching_solution,
				'train_time': 0,
				'f1': 0
			})
		else:
			result.append({
						'dataset': dataset,
						'matching_solution': matching_solution,
						'train_time': statistics.mean(times),
						'f1': statistics.mean(results_f1)
					})
		print("FINAL RESULT", result, file=sys.stdout)
		#pd.DataFrame(result)].to_csv(os.path.join(base_output_path, 'results.csv'), index=False)

	result = pd.DataFrame(result)
	result.to_csv(f'results_{dataset_config}.csv')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="what the program does")
	parser.add_argument("--matching-solution", required=True, help="Matching solution")
	parser.add_argument("--datasets", required=True, help="Matching solution")
	parser.add_argument("--wandb", action='store_true', help="Matching solution")
	args = parser.parse_args()
	main(args.matching_solution, args.datasets, args.wandb)