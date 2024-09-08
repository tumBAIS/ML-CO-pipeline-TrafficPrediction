#!/bin/bash

PERCENTAGE_ORIGINAL="1"
SEEDS="1 2 12"

source venv/bin/activate

cd matsim-berlin/

for SEED in $SEEDS
do
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar ./creation/createScenarios_sparseWorld.java -po ${PERCENTAGE_ORIGINAL} -s ${SEED} -li 1 
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar org.matsim.run.RunBerlinScenario ./scenarios/sparse_worlds/po-${PERCENTAGE_ORIGINAL}_s-${SEED}/config.xml
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
	python calculate_training_data.py --percentage_original=${PERCENTAGE_ORIGINAL} --seed=${SEED} --simulation="sparseWorlds" --mode=${MODE} --simulator="matsim"
done

