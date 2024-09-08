#!/bin/bash
#SBATCH -J eva
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --exclude=i23r06c03s02,i23r06c02s05,i23r06c01s06,i23r05c01s05
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mail-type=end
#SBATCH --mail-user=kai.jungel@tum.de
#SBATCH --export=NONE
#SBATCH --time=22:00:00

if [[ ${RUNNING_CLUSTER} == "BAIS" ]]; then
	module load python/3.8.11
	module load gurobi/10.0.0
else
	module load python/3.8.11-base
	module load gurobi/10.00
fi

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

cd surrogate

MODE="Validate"
for ITERATION in ${VALIDATION_ITERATRIONS}
do
	echo $ITERATION
	python evaluation.py --model=${MODEL} --learning=${LEARNING} --simulation=${SIMULATION} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --read_in_iteration=${ITERATION} --mode=${MODE}  --perturbation=${PERTURBATION} --standardize=${STANDARDIZE}  --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER}  --num_discrete_time_steps=${NUM_DISCRETE_TIME_STEPS} --capacity_individual_theta=${CAPACITY_INDIVIDUAL_THETA} --experiment_name=${EXPERIMENT} --num_bundled_capacities=${NUM_BUNDLED_CAPACITIES} --capacity_multiplicator=${CAPACITY_MULTIPLICATOR} --num_cpus=${SLURM_CPUS_PER_TASK} --simulator=${SIMULATOR} --sd_perturbation=${SD_PERTURBATION} --evaluation_procedure=${EVALUATION_PROCEDURE} --dir_results="./results_paper/" --sd_perturbation_evaluation=${SD_PERTURBATION_EVALUATION} --evaluation_metric=${EVALUATION_METRIC} --num_perturbations=${NUM_PERTURBATIONS} --feature=${FEATURE}
	if [[ ${MODEL} == "Base" ]]; then
		break
	fi
done


python compare_learning_iterations.py --model=${MODEL} --learning=${LEARNING} --simulation=${SIMULATION} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW}  --perturbation=${PERTURBATION} --standardize=${STANDARDIZE} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER}  --num_discrete_time_steps=${NUM_DISCRETE_TIME_STEPS} --capacity_individual_theta=${CAPACITY_INDIVIDUAL_THETA} --experiment_name=${EXPERIMENT} --num_bundled_capacities=${NUM_BUNDLED_CAPACITIES} --capacity_multiplicator=${CAPACITY_MULTIPLICATOR} --num_cpus=${SLURM_CPUS_PER_TASK} --simulator=${SIMULATOR} --sd_perturbation=${SD_PERTURBATION} --evaluation_procedure=${EVALUATION_PROCEDURE} --dir_results="./results_paper/" --sd_perturbation_evaluation=${SD_PERTURBATION_EVALUATION} --evaluation_metric=${EVALUATION_METRIC} --num_perturbations=${NUM_PERTURBATIONS} --feature=${FEATURE}


MODE="Test"
python evaluation.py --model=${MODEL} --learning=${LEARNING} --simulation=${SIMULATION} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --mode=${MODE}  --perturbation=${PERTURBATION} --standardize=${STANDARDIZE} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER}  --num_discrete_time_steps=${NUM_DISCRETE_TIME_STEPS} --capacity_individual_theta=${CAPACITY_INDIVIDUAL_THETA} --experiment_name=${EXPERIMENT} --num_bundled_capacities=${NUM_BUNDLED_CAPACITIES} --capacity_multiplicator=${CAPACITY_MULTIPLICATOR} --num_cpus=${SLURM_CPUS_PER_TASK} --simulator=${SIMULATOR} --sd_perturbation=${SD_PERTURBATION} --evaluation_procedure=${EVALUATION_PROCEDURE} --dir_results="./results_paper/" --sd_perturbation_evaluation=${SD_PERTURBATION_EVALUATION} --evaluation_metric=${EVALUATION_METRIC} --num_perturbations=${NUM_PERTURBATIONS} --feature=${FEATURE}


