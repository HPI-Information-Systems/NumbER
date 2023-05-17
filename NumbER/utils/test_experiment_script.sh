#! /bin/bash
outdir="NumbER/experiments/number"
config_path="NumbER/configs/number.yaml"
eval "$(conda shell.bash hook)"
source .env
function run_experiment {
	echo Doing it for $1
	echo Tagging with $2
	conda activate $1
	python3 NumbER/utils/experiment_script.py --matching-solution $1 --tag $2 --datasets "test"
}
set -e
eval "$(conda shell.bash hook)"
cd /hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER

#run_experiment "deep_matcher"
run_experiment "ditto" $1
