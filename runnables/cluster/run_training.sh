#!/bin/bash
#SBATCH -J tr
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --exclude=i23r06c03s02,i23r06c02s05,i23r06c01s06,i23r05c01s05
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mail-type=end
#SBATCH --mail-user=kai.jungel@tum.de
#SBATCH --export=NONE
#SBATCH --time=20:00:00

if [[ ${RUNNING_CLUSTER} == "BAIS" ]]; then
	module load python/3.8.11
	module load gurobi/10.0.0
else
	module load python/3.8.11-base
	module load gurobi/10.00
fi

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

echo CPUS_PER_TASK : $SLURM_CPUS_PER_TASK
echo MEM_PER_NODE : $SLURM_MEM_PER_NODE

cd surrogate

python training.py --model=${MODEL} --learning=${LEARNING} --simulation=${SIMULATION} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --num_training_epochs=${NUM_TRAINING_EPOCHS} --num_perturbations=${NUM_PERTURBATIONS} --perturbation=${PERTURBATION} --standardize=${STANDARDIZE} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --capacity_individual_theta=${CAPACITY_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER} --num_discrete_time_steps=${NUM_DISCRETE_TIME_STEPS} --num_bundled_capacities=${NUM_BUNDLED_CAPACITIES} --capacity_multiplicator=${CAPACITY_MULTIPLICATOR} --num_cpus=${SLURM_CPUS_PER_TASK} --experiment_name=${EXPERIMENT} --max_time=${TIME} --simulator=${SIMULATOR} --sd_perturbation=${SD_PERTURBATION} --max_successive_averages_iteration=${MAX_SUCCESSIVE_AVERAGES_ITERATION} --num_line_search_it=${NUM_LINE_SEARCH_IT} --evaluation_metric=${EVALUATION_METRIC} --feature=${FEATURE}

