#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_training.sh

source venv/bin/activate

MODEL="NN"
LEARNINGS="structured supervised"
SURROGATES="ml-co"
SIMULATION="smallWorlds_pricing"
PERCENTAGE_ORIGINAL="1"
PERCENTAGE_NEW="1.0"
SEED_NEW="1"
SEED="1"
NUM_TRAINING_EPOCHS=50
PERTURBATION="additive"
NUM_EPOCHS=5

cd bayesian

for SURROGATE in ${SURROGATES}
do
	echo $SURROGATE
	for LEARNING in ${LEARNINGS}
	do
		echo $LEARNING
		python optimization.py --simulation=$SIMULATION --percentage_original=$PERCENTAGE_ORIGINAL --percentage_new=$PERCENTAGE_NEW --seed_new=$SEED_NEW --seed=$SEED --num_training_epochs=$NUM_TRAINING_EPOCHS --perturbation=$PERTURBATION --model=$MODEL --learning=$LEARNING --bayesian_verbose=0 --verbose=0 --max_evals=$NUM_EPOCHS
	done
done

