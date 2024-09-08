#!/bin/bash


# salloc --ntasks=1 --cpus-per-task=2 --partition=cm2_inter 

# module load python/3.8.11-base
# module load gurobi/10.00

# source venv/bin/activate


# cd surrogate

# srun python training.py --model="NN" --learning="structured" --simulation="low_entropy" --percentage_original="1" --percentage_new="1" --seed_new="1" --num_training_epochs="1" --num_perturbations="1" --perturbation="additive" --standardize="1" --time_variant_theta="0" --trip_individual_theta="0" --capacity_individual_theta="0" --co_optimizer="multicommodityflow" --num_discrete_time_steps="10" --num_bundled_capacities="5" --capacity_multiplicator="500" --num_cpus=2 --experiment_name="" --max_time="10:00:00" --simulator="easySimulation" --sd_perturbation=1 --max_successive_averages_iteration=5 --num_line_search_it=5


