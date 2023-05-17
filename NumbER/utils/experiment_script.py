import os
import datetime
import pandas as pd
import shutil
from threading import Timer
import argparse
import importlib
import sys
import statistics
import traceback
import wandb

#from NumbER.matching_solutions.matching_solutions.ditto import DittoMatchingSolution
#, DeepMatcherMatchingSolution
from NumbER.utils.experiment_config import experiment_configs

def main(matching_solution, dataset_config, use_wandb, tag):
	print("TAG", tag)
	result = []
	date_time = datetime.datetime.now()
	base_input_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets"
	base_output_path = f"/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/experiments_{tag}_{date_time.strftime('%d_%m_%Y_%H:%M:%S')}"
	experiment_config = experiment_configs[dataset_config]
	print("Base input path", base_input_path)
	print("Base output path", base_output_path)
	print("xperiment config", experiment_config)

	for dataset, paths in experiment_config.items():
		results_f1 = []
		times = []
		for i in range(3):
			print("DOING it for ", dataset, " run ", i)
			wandb.init(
				project="NumbER",
    			entity="Lasklu",
       			mode="online" if use_wandb else "disabled",
          		tags=[tag],
				config={
					**paths['config'][matching_solution]["train"],
					'dataset': dataset,
					'matching_solution': matching_solution,
					'run': str(i)
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
				elif matching_solution == "md2m":
					module = importlib.import_module("NumbER.matching_solutions.matching_solutions.md2m")
					algorithm = getattr(module, 'MD2MMatchingSolution')
				else:
					module = None
				dataset_path = os.path.join(base_output_path, 'datasets', dataset)
				print("Dataset path", dataset_path)
				os.makedirs(dataset_path, exist_ok=True)
				record_path = os.path.join(base_input_path, dataset, 'features.csv')
				matches_path = os.path.join(base_input_path, dataset, 'matches_closed.csv')
				print("Sampler is", paths['blocking']['sampler'])
				sampler = paths['blocking']['sampler'](record_path, matches_path)#, os.path.join(base_input_path, dataset, 'similarity.npy'))
				train_formatter, valid_formatter, test_formatter = sampler.create_format(matching_solution,paths['blocking']['config'])# record_path, matches_path, matching_solution, paths['blocking']['attributes'], paths['blocking']['train_window_size'], paths['blocking']['test_window_size']),#)
				print("Created format")
				pd.read_csv(record_path).replace({"\t", ""}, regex=True).to_csv(os.path.join(dataset_path, 'dataset.csv'), index=False)
				print("Doing the formatter")
				(train_path, train_record_path), (valid_path, valid_record_path), (test_path, test_record_path) = train_formatter.write_to_file(os.path.join(dataset_path, f'train_{i}')), valid_formatter.write_to_file(os.path.join(dataset_path, f'valid_{i}')), test_formatter.write_to_file(os.path.join(dataset_path, f'test_{i}'))
				print("Train path", train_path, "valid path", valid_path, "test path", test_path)
				if train_record_path is not None and valid_record_path is not None and test_record_path is not None:
					#config['test']['train_records_path'] = train_records_path
					config['test']['valid_records_path'] = valid_record_path
					config['test']['test_records_path'] = test_record_path
				train_path = "/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/matching_solutions/ditto/data/er_magellan/Structured/Beer/train.txt"
				valid_path = "/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/matching_solutions/ditto/data/er_magellan/Structured/Beer/valid.txt"
				test_path = "/hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER/NumbER/matching_solutions/ditto/data/er_magellan/Structured/Beer/test.txt"
				train_record_path = None
				valid_record_path = None
				test_record_path = None	
				solution = algorithm(dataset, train_path, valid_path, test_path)
				print("Initialized solution")
				try: 
					print("Training model...", file=sys.stdout)
					config['train']['i'] = str(i)
					best_f1, model, threshold, train_time = solution.model_train(**config['train'])
					print(f"Predicting... Best f1 achieved: {best_f1}", file=sys.stdout)
					if threshold is not None:
						print("Setting threshold to: ", threshold)
						config['test']['threshold'] = threshold
					else:
						print("No threshold set")
					prediction = solution.model_predict(**config['test'], model=model)
					print("Predicted", prediction)
					wandb.log({"f1": prediction['evaluate'], "time": train_time})
				except Exception as e:
					print(traceback.format_exc(), file=sys.stdout)
					print(f"An error occured in train/predict: {e}", file=sys.stdout)
					wandb.finish(1)
					continue
				matching_solution_path = os.path.join(base_output_path, 'matching_solution', matching_solution)
				print("Matching solution path", matching_solution_path)
				os.makedirs(matching_solution_path, exist_ok=True)
				print(prediction['predict'][0])
				print("tes",test_formatter.goldstandard.reset_index())
				pred = pd.merge(test_formatter.goldstandard.reset_index(), prediction['predict'][0].reset_index(), left_index=True, right_index=True)[['p1', 'p2', 'match', 'match_confidence']]
				pred.rename(columns={'match': 'prediction'},inplace=True, errors='raise')
				pred.to_csv(os.path.join(matching_solution_path, dataset + "_" + str(i) + '.csv'), index=False)
				print("Wrote prediction", pred)
				test_formatter.goldstandard.to_csv(os.path.join(matching_solution_path, dataset + f'_goldstandard_{i}.csv'), index=False)
				results_f1.append(prediction['evaluate'])
				times.append(train_time)
				print(f"On dataset {dataset} with matching solution {matching_solution} in time {train_time} scored {prediction['evaluate']} in iteration {i}", file=sys.stdout)
				shutil.make_archive('coded', 'zip', base_output_path)
				print("Finished", i)
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

	#result = pd.DataFrame(result)
	#result.to_csv(f'results_{dataset_config}.csv')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="what the program does")
	parser.add_argument("--matching-solution", required=True, help="Matching solution")
	parser.add_argument("--datasets", required=True, help="Matching solution")
	parser.add_argument("--wandb", action='store_true', help="Matching solution")
	parser.add_argument("--tag")
	args = parser.parse_args()
	try:
		main(args.matching_solution, args.datasets, args.wandb, args.tag)
	except Exception as e:
		print(traceback.format_exc(), file=sys.stdout)
		print(f"An error occurec: {e}", file=sys.stdout)