import io
import csv
import glob

import os
import requests
BASE_URL = 'http://localhost:8123/api/v1'

def abc(path, del_old_exp=False):
    if del_old_exp:
    	#remove all old runs
        pass
    for fname in glob.glob(os.path.join(path,'datasets','*.csv')):
        csv_file = open(os.path.join(path,'datasets',fname), 'r')
        create_dataset(fname[:-4], csv_file)
    for folder in os.listdir(os.path.join(path, 'matching_solution')):
        #create matching solution
        matching_solution_id = create_matching_solution(folder)
        for file in os.listdir(folder):
            if file[:-16] == "goldstandard.csv":
                continue
            path_to_file = os.path.join(path, folder)
            upload_experiment("groundtruth", fname[:-4], matching_solution_id, os.path.join(path_to_file, f"{file[:-4]}_goldstandard.csv"))
            upload_experiment(file[:-4], fname[:-4], matching_solution_id, os.path.join(path_to_file, file))
        # upload experiment
        pass
abc('/hpi/fs00/share/fg-naumann/lukas.laskowski/experiments/experiments_1/')
def create_dataset(dataset_name, file, idcolumn="id", description="Automatically generated"):
    id = requests.post(f'{BASE_URL}/datasets/', json={'name': dataset_name, 'description':description, 'tags':[dataset_name]}).json()
    return requests.put(f'{BASE_URL}/datasets/{id}/file?format=pilot', data=file, headers={'Content-Type': 'text/csv'}, json={'idColumn': idcolumn, 'quote':'"', "escape":"'", "separator":","})

def upload_experiment(name_of_experiment, dataset_name, algorithm_id, experiment_file_path):
    new_experiment_payload = {"datasetId": get_dataset_id(dataset_name), "algorithmId": algorithm_id,
                              "name": name_of_experiment, "description": 'automatic-upload'}
    create_experiment_response = requests.post(
        f'{BASE_URL}/experiments', json=new_experiment_payload)
    new_experiment_id = create_experiment_response.text
    upload_pairs_response = requests.put(
        f'{BASE_URL}/experiments/{new_experiment_id}/file?format=pilot', data=open(experiment_file_path), headers={'Content-Type': 'text/csv'})
    #print(upload_pairs_response.status_code)
    # goldstandard_id = get_gold_standard_id_for_dataset(
    #     dataset_names[dataset_name])
    # print(get_results(new_experiment_id, goldstandard_id))
    return new_experiment_id


def create_matching_solution(name):
	return requests.post(f'{BASE_URL}/algorithms', json={'name':name}).json()

def get_dataset_id(dataset_name):
    datasets = requests.get(f'{BASE_URL}/datasets').json()
    return list(filter(lambda x: x['name'] == dataset_name, datasets))[0]['id']


def get_gold_standard_algorithm_id():
    algorithms = requests.get(f'{BASE_URL}/algorithms').json()
    return list(filter(lambda algorithm: algorithm['name'] == 'Gold Standard', algorithms))[0]['id']


def get_gold_standard_id_for_dataset(dataset_name):
    goldstandard_algorithm_id = get_gold_standard_algorithm_id()
    dataset_id = get_dataset_id(dataset_name)
    experiments = requests.get(f'{BASE_URL}/experiments').json()
    return list(filter(lambda experiment: experiment['algorithmId'] == goldstandard_algorithm_id and experiment['datasetId'] == dataset_id, experiments))[0]['id']


def get_dataset_id_related_to_experiment(experiment_id):
    experiment = requests.get(f'{BASE_URL}/experiments/{experiment_id}').json()
    return experiment['datasetId']


def get_results(experiment_id, goldstandard_id):
    return requests.get(
        f'{BASE_URL}/benchmark/metrics?groundTruthExperimentId={goldstandard_id}&predictedExperimentId={experiment_id}').json()
