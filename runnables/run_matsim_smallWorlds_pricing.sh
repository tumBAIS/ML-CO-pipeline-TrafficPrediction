#!/bin/bash

# START FROM ./system_response_approximation/: bash runnables/run_matsim_smallWorlds_pricing.sh

SEEDS="1"


source venv/bin/activate

for SEED in $SEEDS
do
	echo $SEED
	python ./surrogate/generating/smallWorld_scenarios_roadpricing.py --seed=${SEED} --simulation="smallWorlds_pricing"
done



cd matsim-berlin/
for SEED in $SEEDS
do
	echo $SEED
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar ./creation/createScenarios_smallWorld.java -s ${SEED} -li 2
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar ./creation/createScenarios_smallWorld_roadPricing.java -s ${SEED} -li 2
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar org.matsim.run.RunBerlinScenarioRoadPricing ./scenarios/smallWorlds_pricing/s-${SEED}/config.xml
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
	python calculate_training_data.py --seed=${SEED} --simulation="smallWorlds_pricing" --mode=${MODE} --simulator="matsim"
done

