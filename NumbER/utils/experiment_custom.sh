#! /bin/bash
outdir="NumbER/experiments/number"
config_path="NumbER/configs/number.yaml"
eval "$(conda shell.bash hook)"
source .env
function run_experiment {
	echo Doing it for $1
	echo Tagging with $2
	echo Iteration $3
	conda activate $1
	#python3 NumbER/utils/experiment_script.py --matching-solution $1 --datasets "numeric"  --tag $2 --iteration $3 --wandb #> output.out
	#python3 NumbER/utils/experiment_script.py --matching-solution $1 --datasets "earthquakes"  --tag $2 --wandb #> output.out
	#python3 NumbER/utils/experiment_script.py --matching-solution $1 --datasets "x3_all"  --tag $2 --wandb #> output.out
	python3 NumbER/matching_solutions/utils/similarity_calculation_complete.py
}
set -e
eval "$(conda shell.bash hook)"
cd /hpi/fs00/home/lukas.laskowski/Masterarbeit/NumbER

#run_experiment "deep_matcher" $1
#run_experiment "ditto" $1
run_experiment "ensemble_learner" $1
#run_experiment "ditto" $1
#run_experiment "deep_matcher" $1
#run_experiment "ditto" $1
# for i in {0..4}; do
# 	run_experiment "ditto" $1 $i
# done
