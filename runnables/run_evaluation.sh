#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_evaluation.sh

source venv/bin/activate

ENVIRONMENT="squareWorlds_short"
PERCENTAGE_ORIGINAL="1"
PERCENTAGE_NEW="1"
SEED_NEW="1"
NUM_TRAINING_EPOCHS=2
PERTURBATIONS="additive"
MODELS="NN"
LEARNINGS="structured supervised"
TIME_VARIANT_THETAS="0"
TRIP_INDIVIDUAL_THETAS="0"
OPTIMIZERS="addedshortestpaths multicommodityflow_timeexpanded"
TIME_GRANULARITIES="300"
NUM_PERTURBATIONS=2

running_type=time_expansion_check

if [ ${running_type} == time_expansion_check ]; then
	ABBREVIATION="tec"
	TIME_VARIANT_THETAS="0 1"
	TRIP_INDIVIDUAL_THETAS="0 1"
	OPTIMIZERS="addedshortestpaths multicommodityflow_timeexpanded"
	LEARNINGS="structured supervised"
	MODELS="NN Base"
elif [ ${running_type} == time_expansion_time_granularity ]; then
	ABBREVIATION="teg"
	TIME_GRANULARITIES="120 300 600"
	MODELS="NN"
fi


cd surrogate


for MODEL in ${MODELS}
do
	for LEARNING in ${LEARNINGS}
	do
		for OPTIMIZER in ${OPTIMIZERS}
		do
			for PERTURBATION in ${PERTURBATIONS}
			do
				for TIME_VARIANT_THETA in ${TIME_VARIANT_THETAS}
				do
					for TRIP_INDIVIDUAL_THETA in ${TRIP_INDIVIDUAL_THETAS}
					do
						for TIME_GRANULARITY in ${TIME_GRANULARITIES}
						do
							for ITERATION in 1
							do
								echo "Start validate"
								MODE="Validate"
								python evaluation.py  --model=${MODEL} --learning=${LEARNING} --simulation=${ENVIRONMENT} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --read_in_iteration=${ITERATION} --mode=${MODE} --perturbation=${PERTURBATION} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER} --time_granularity=${TIME_GRANULARITY}
							done
							
							echo "Start compare"
							python compare_learning_iterations.py --model=${MODEL} --learning=${LEARNING} --simulation=${ENVIRONMENT} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --perturbation=${PERTURBATION} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER} --time_granularity=${TIME_GRANULARITY}
							
							echo "Start test"
							MODE="Test"
							python evaluation.py --model=${MODEL} --learning=${LEARNING} --simulation=${ENVIRONMENT} --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --mode=${MODE} --perturbation=${PERTURBATION} --time_variant_theta=${TIME_VARIANT_THETA} --trip_individual_theta=${TRIP_INDIVIDUAL_THETA} --co_optimizer=${OPTIMIZER} --time_granularity=${TIME_GRANULARITY}
						done
					done
				done
			done
		done
	done
done

