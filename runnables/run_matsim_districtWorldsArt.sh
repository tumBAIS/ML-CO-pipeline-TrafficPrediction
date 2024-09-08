#!/bin/bash

PERCENTAGE_NEW="1.0"
SEED_NEW="1"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
PERCENTAGE_ORIGINAL="1"

source venv/bin/activate

cd matsim-berlin/

for SEED in $SEEDS
do
	echo $SEED
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar ./creation/createScenarios_districtWorldArt.java -po ${PERCENTAGE_ORIGINAL} -pn ${PERCENTAGE_NEW} -sn ${SEED_NEW} -s ${SEED} -li 40 #40
	java -Xmx8192m -cp matsim-berlin-5.5.3.jar org.matsim.run.RunBerlinScenario ./scenarios/districtWorldsArt/po-${PERCENTAGE_ORIGINAL}_pn-${PERCENTAGE_NEW}_sn-${SEED_NEW}_s-${SEED}/config.xml
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
	python calculate_training_data.py  --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --seed=${SEED} --simulation="districtWorldsArt" --simulator="matsim" --mode=${MODE}
done

