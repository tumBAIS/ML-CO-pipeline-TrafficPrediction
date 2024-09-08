#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_matsim_squareWorlds_short.sh

SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"


source venv/bin/activate

for SEED in $SEEDS
do
	echo $SEED
	python ./surrogate/generating/squareWorld_scenarios_short.py --seed=${SEED} --simulation="squareWorlds_short"
done

cd matsim-berlin/
for SEED in $SEEDS
do
	echo $SEED
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar ./creation/createScenarios_squareWorld_short.java -s ${SEED} -li 40
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar org.matsim.run.RunBerlinScenario ./scenarios/squareWorlds_short/s-${SEED}/config.xml
done

cd ../surrogate
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
	python calculate_training_data.py --seed=${SEED} --simulation="squareWorlds_short" --mode=${MODE} --simulator="matsim"
done

