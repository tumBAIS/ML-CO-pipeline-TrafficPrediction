#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_againstsupervisedSimulation_squareWorlds_short.sh

SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"


source venv/bin/activate

for SEED in $SEEDS
do
	echo $SEED
	python ./surrogate/generating/squareWorld_scenarios_short.py --seed=${SEED} --simulation="squareWorlds_short"
done


cd toySimulation/
for SEED in $SEEDS
do
	if [ $SEED -lt "10" ]; then
		MODE="Train"
	elif [ $SEED -lt "15" ]; then
		MODE="Validate"
	elif [ $SEED -lt "20" ]; then
		MODE="Test"
	fi
	echo $SEED
	python run.py --seed=${SEED} --simulation="squareWorlds_short" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
done

