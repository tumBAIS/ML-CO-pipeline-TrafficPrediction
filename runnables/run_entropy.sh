#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_entropy.sh

SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"


source venv/bin/activate

for SEED in $SEEDS
do
	echo $SEED
	#python ./surrogate/generating/entropy_scenarios.py --seed=${SEED} --simulation="low_entropy" --dimension_entropy=6
	#python ./surrogate/generating/entropy_scenarios.py --seed=${SEED} --simulation="high_entropy" --dimension_entropy=6
	#python ./surrogate/generating/entropy_scenarios.py --seed=${SEED} --simulation="low_entropy_street" --dimension_entropy=6
	#python ./surrogate/generating/entropy_scenarios.py --seed=${SEED} --simulation="high_entropy_street" --dimension_entropy=6
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
	#python run.py --seed=${SEED} --simulation="low_entropy" --simulator="easySimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy" --simulator="easySimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="low_entropy_street" --simulator="easySimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy_street" --simulator="easySimulation" --mode=${MODE} --num_discrete_time_steps=20
	python run.py --seed=${SEED} --simulation="squareWorlds_short" --simulator="easySimulation" --mode=${MODE} --num_discrete_time_steps=20
	
	#python run.py --seed=${SEED} --simulation="low_entropy" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="low_entropy_street" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy_street" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
	python run.py --seed=${SEED} --simulation="squareWorlds_short" --simulator="againstsupervisedSimulation" --mode=${MODE} --num_discrete_time_steps=20
	
	#python run.py --seed=${SEED} --simulation="low_entropy" --simulator="easySimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy" --simulator="easySimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="low_entropy_street" --simulator="easySimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy_street" --simulator="easySimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	python run.py --seed=${SEED} --simulation="squareWorlds_short" --simulator="easySimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	
	#python run.py --seed=${SEED} --simulation="low_entropy" --simulator="randomSimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy" --simulator="randomSimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="low_entropy_street" --simulator="randomSimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	#python run.py --seed=${SEED} --simulation="high_entropy_street" --simulator="randomSimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
	python run.py --seed=${SEED} --simulation="squareWorlds_short" --simulator="randomSimulationEquilibrium" --mode=${MODE} --num_discrete_time_steps=20
done

