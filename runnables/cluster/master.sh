#!/bin/bash

# start from system_response_approximation: bash runnables/cluster/master.sh

RUNNING_CLUSTER="LRZ"


ACTION="training"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
MATSIM_ITERATIONS=10
NUM_TRAINING_EPOCHS=100
VALIDATION_ITERATRIONS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
NUM_PERTURBATIONS=50
STANDARDIZE=1
SIMULATIONS="squareWorlds_short"
SIMULATORS="matsim"  #"matsim againstsupervisedSimulation easySimulation"
PERCENTAGE_ORIGINAL="1"
PERCENTAGE_NEW="0.1"
SEED_NEW="1"
PERTURBATIONS="additive"
MODELS="NN Base"
LEARNINGS="structured"
TIME_VARIANT_THETAS="0"
TRIP_INDIVIDUAL_THETAS="0"
CAPACITY_INDIVIDUAL_THETAS="0"
OPTIMIZERS="multicommodityflow"
NUMS_DISCRETE_TIME_STEPS="20"
NUMS_BUNDLED_CAPACITIES="3"
CAPACITY_MULTIPLICATORS="-1"
SD_PERTURBATIONS="1"
SD_PERTURBATIONS_EVALUATION="1"
EVALUATION_PROCEDURES="original"
EVALUATION_METRIC="original"
MAX_SUCCESSIVE_AVERAGES_ITERATION=100
NUM_LINE_SEARCH_IT=5
FEATURES="''"

experiment=benchmarks

if [ ${experiment} == benchmarks ]; then
	ABBREVIATION="ben"
	SIMULATIONS="districtWorldsArt" #"districtWorlds" #"cutoutWorldsWithoutBoundary" #"cutoutWorldsSpeed cutoutWorldsCap" #"squareWorlds_short squareWorlds_short_capacitated cutoutWorlds smallWorlds_short cutoutWorldsCap
	SIMULATORS="matsim"
	TIME_VARIANT_THETAS="0"
	TRIP_INDIVIDUAL_THETAS="0"
	CAPACITY_INDIVIDUAL_THETAS="0 1"
	OPTIMIZERS="multicommodityflow wardropequilibria wardropequilibriaRegularized"
	LEARNINGS="structured supervised"
	MODELS="NN Base GNN" #GNN RandomForest"
elif [ ${experiment} == entropy ]; then
	ABBREVIATION="ent"
	SIMULATIONS="low_entropy high_entropy low_entropy_street high_entropy_street squareWorlds_short"
	SIMULATORS="againstsupervisedSimulation" #easySimulation easySimulationEquilibrium randomSimulationEquilibrium
	TIME_VARIANT_THETAS="0"
	TRIP_INDIVIDUAL_THETAS="0"
	CAPACITY_INDIVIDUAL_THETAS="0 1"
	OPTIMIZERS="multicommodityflow wardropequilibria wardropequilibriaRegularized"
	LEARNINGS="structured supervised"
	MODELS="NN Base"
elif [ ${experiment} == ownsimulation ]; then
	ABBREVIATION="oSi"
	SIMULATORS="easySimulationEquilibrium" #randomSimulationEquilibrium" #"againstsupervisedSimulation"  #"easySimulation"
	TIME_VARIANT_THETAS="0"
	TRIP_INDIVIDUAL_THETAS="0"
	CAPACITY_INDIVIDUAL_THETAS="0 1"
	OPTIMIZERS="multicommodityflow wardropequilibria wardropequilibriaRegularized"
	LEARNINGS="structured supervised"
	MODELS="NN Base"
elif [ ${experiment} == perturbedEvaluation ]; then
	ABBREVIATION="petEv"
	SIMULATIONS="squareWorlds_short" # cutoutWorlds
	TIME_VARIANT_THETAS="0 1"
	TRIP_INDIVIDUAL_THETAS="0 1"
	CAPACITY_INDIVIDUAL_THETAS="0 1"
	OPTIMIZERS="multicommodityflow"
	LEARNINGS="structured"
	EVALUATION_PROCEDURES="perturbed"
	SD_PERTURBATIONS_EVALUATION="1 0.8 0.6 0.4 0.2 0.0"
	MODELS="NN"
elif [ ${experiment} == structuredwardrop ]; then
	ABBREVIATION="stwar"
	SIMULATIONS="squareWorlds_short" # cutoutWorlds
	OPTIMIZERS="wardropequilibria multicommodityflow addedshortestpaths"
	LEARNINGS="structured"  # "structured_wardrop"
	MODELS="NN"
	EVALUATION_PROCEDURES="original"
elif [ ${experiment} == withoutPerturbation ]; then
	ABBREVIATION="wp"
	SIMULATIONS="squareWorlds_short"
	SIMULATORS="matsim"
	TIME_VARIANT_THETAS="1"
	CAPACITY_INDIVIDUAL_THETAS="0"
	NUMS_DISCRETE_TIME_STEPS="40"
	OPTIMIZERS="wardropequilibria wardropequilibriaRegularized multicommodityflow"
	LEARNINGS="structured"
	MODELS="NN"
	EVALUATION_METRIC="timedist"
	NUM_PERTURBATIONS=0
elif [ ${experiment} == numDiscreteTimeStepsWith ]; then
	ABBREVIATION="teg"
	SIMULATIONS="squareWorlds_short" # squareWorlds_short_capacitated cutoutWorlds smallWorlds_short
	SIMULATORS="matsim"
	TIME_VARIANT_THETAS="1"
	CAPACITY_INDIVIDUAL_THETAS="0"
	NUMS_DISCRETE_TIME_STEPS="40"  #"20 30 40"
	OPTIMIZERS="multicommodityflow wardropequilibria wardropequilibriaRegularized"
	LEARNINGS="structured supervised"  # "structured_wardrop"
	MODELS="NN Base"
	EVALUATION_METRIC="timedist"
elif [ ${experiment} == numDiscreteTimeStepsWithTest ]; then
	ABBREVIATION="tT"
	SIMULATIONS="squareWorlds_short"
	SIMULATORS="matsim"
	FEATURES="withoutedgessquare square withoutedges withoutedgeslogic original logic withoutSpawn new"
	TIME_VARIANT_THETAS="1"
	CAPACITY_INDIVIDUAL_THETAS="0 1"
	NUMS_DISCRETE_TIME_STEPS="40"
	OPTIMIZERS="multicommodityflow"
	LEARNINGS="structured supervised"
	MODELS="NN"
	EVALUATION_METRIC="timedist"
elif [ ${experiment} == numDiscreteTimeStepsUncapacitated ]; then
	ABBREVIATION="tunc"
	SIMULATIONS="squareWorlds_short" # squareWorlds_short_capacitated cutoutWorlds smallWorlds_short
	SIMULATORS="matsim"
	TIME_VARIANT_THETAS="1"
	NUMS_DISCRETE_TIME_STEPS="20 40 60 80 100"
	OPTIMIZERS="multicommodityflow_uncapacitated"
	LEARNINGS="structured supervised"
	MODELS="NN Base"
	EVALUATION_METRIC="timedist"
elif [ ${experiment} == numBundledCapacities ]; then
	ABBREVIATION="dc"
	SIMULATIONS="cutoutWorlds"
	CAPACITY_INDIVIDUAL_THETAS="1"
	NUMS_BUNDLED_CAPACITIES="3 4 5 6 7 8 9 10 15 20"
	LEARNINGS="structured"
	OPTIMIZERS="multicommodityflow"
	MODELS="NN"
elif [ ${experiment} == diffCapacities ]; then
	ABBREVIATION="cap"
	CAPACITY_MULTIPLICATORS="0 5 10 20 50 100 200 500 1000 2000"
	LEARNINGS="structured"
	OPTIMIZERS="multicommodityflow"
	MODELS="NN"
elif [ ${experiment} == testcutout ]; then
	ABBREVIATION="tcut"
	SIMULATIONS="cutoutWorlds" # cutoutWorlds_5
	SIMULATORS="matsim"
	TIME_VARIANT_THETAS="0"
	TRIP_INDIVIDUAL_THETAS="0"
	CAPACITY_INDIVIDUAL_THETAS="1"
	OPTIMIZERS="multicommodityflow"
	LEARNINGS="structured"
	MODELS="NN"
fi


for SIMULATION in ${SIMULATIONS}
do
	for SIMULATOR in ${SIMULATORS}
	do

		if [[ ${SIMULATION} == "cutoutWorlds" && ${experiment} != "numBundledCapacities" ]]; then
			echo "reset NUMS_BUNDLED_CAPACITIES"
			NUMS_BUNDLED_CAPACITIES="15"
		elif [[ ${SIMULATION} == "districtWorldsArt" && ${experiment} != "numBundledCapacities" ]]; then
			echo "reset NUMS_BUNDLED_CAPACITIES"
			NUMS_BUNDLED_CAPACITIES="10"
		else
			NUMS_BUNDLED_CAPACITIES="3"
		fi
		
		if [[ ${SIMULATION} == "districtWorlds" || ${SIMULATION} == "districtWorldsArt" ]]; then
			echo "reset PERCENTAGE_NEW"
			PERCENTAGE_NEW="1.0"
		fi

		if [[ ${ACTION} == "generate" ]]; then
			if [[ ${SIMULATION} == "cutoutWorlds" ]]; then

				sbatch --job-name="cu" --export=PERCENTAGE_ORIGINAL=$PERCENTAGE_ORIGINAL,PERCENTAGE_NEW=$PERCENTAGE_NEW,SEED_NEW=$SEED_NEW,SEEDS="${SEEDS}",MATSIM_ITERATIONS=$MATSIM_ITERATIONS ./runnables/cluster/run_generate_training_data_cutout.sh
				
			elif [[ ${SIMULATION} == "sparseWorlds" ]]; then
				
				for SEED in ${SEEDS}
				do
					sbatch --job-name="sp" --export=PERCENTAGE_ORIGINAL=$PERCENTAGE_ORIGINAL,SEED=$SEED,MATSIM_ITERATIONS=$MATSIM_ITERATIONS ./runnables/cluster/run_generate_training_data_sparse.sh
				done
				
			elif [[ ${SIMULATION} == "smallWorlds" ]]; then
			
				for SEED in ${SEEDS}
				do
					sbatch --job-name="sm" --export=SEED=$SEED,MATSIM_ITERATIONS=$MATSIM_ITERATIONS ./runnables/cluster/run_generate_training_data_small.sh
				done
				
			fi
		fi

		if [[ ${ACTION} == "training" ]]; then
			for MODEL in ${MODELS}
			do
				for LEARNING in ${LEARNINGS}
				do
					for PERTURBATION in ${PERTURBATIONS}
					do
						for SD_PERTURBATION in ${SD_PERTURBATIONS}
						do
							for TIME_VARIANT_THETA in ${TIME_VARIANT_THETAS}
							do
								for TRIP_INDIVIDUAL_THETA in ${TRIP_INDIVIDUAL_THETAS}
								do
									for CAPACITY_INDIVIDUAL_THETA in ${CAPACITY_INDIVIDUAL_THETAS}
									do
										for NUM_DISCRETE_TIME_STEPS in ${NUMS_DISCRETE_TIME_STEPS}
										do
											for NUM_BUNDLED_CAPACITIES in ${NUMS_BUNDLED_CAPACITIES}
											do
												for CAPACITY_MULTIPLICATOR in ${CAPACITY_MULTIPLICATORS}
												do
													for FEATURE in ${FEATURES}
													do
														for OPTIMIZER in ${OPTIMIZERS}
														do
															if [[ ${LEARNING} == "structured" && (${MODEL} == "NN" || ${MODEL} == "GNN" || ${MODEL} == "Linear") ]]; then
																if [[ ${RUNNING_CLUSTER} == "BAIS" ]]; then
																	CPUS=16
																	MEMORY=48000
																else
																	if [[ ${SIMULATION} == "cutoutWorlds" || ${SIMULATION} == "cutoutWorlds_5" ]]; then
																		CPUS=8
																		MEMORY=48000
																	elif [[ ${SIMULATION} == "squareWorlds_short" && ${TIME_VARIANT_THETA} == "1" && ${CAPACITY_INDIVIDUAL_THETA} == "1" ]]; then
																		CPUS=8
																		MEMORY=48000
																	elif [[ ${SIMULATION} == "squareWorlds_short" && ${TIME_VARIANT_THETA} == "1" && ${OPTIMIZER} == "wardropequilibria" ]]; then
																		CPUS=8
																		MEMORY=48000
																	else
																		CPUS=16
																		MEMORY=48000
																	fi
																fi
															else
																CPUS=1
																MEMORY=10000
															fi
															if [[ ${OPTIMIZER} == "addedshortestpaths" && ${LEARNING} == "structured" && (${TIME_VARIANT_THETA} == "1" || ${CAPACITY_INDIVIDUAL_THETA} == "1") ]]; then
																echo "Addedshortestpaths does not contain time variant / capacity edges"
																continue
															fi
															if [[ $((${TRIP_INDIVIDUAL_THETA} + ${TIME_VARIANT_THETA} + ${CAPACITY_INDIVIDUAL_THETA})) > 1 ]]; then
																if ! [[ ${TIME_VARIANT_THETA} == 1 && ${CAPACITY_INDIVIDUAL_THETA} == 1 && ${TRIP_INDIVIDUAL_THETA} == 0 ]]; then
																	echo "We only investigate trip_individual_theta or time_variant_theta or capacity_individual_theta"
																	continue
																fi
															fi
															if [[ ${LEARNING} == "structured" && ${MODEL} == "Base" ]]; then
																echo "Base can not learn structured"
																continue
															fi
															if [[ ${OPTIMIZER} == "wardropequilibria"  && ${LEARNING} == "structured" && ${CAPACITY_INDIVIDUAL_THETA} == "1" ]]; then
																echo "Wardropequilibria can not consider capacity individual theta"
																continue
															fi
															if [[ ${OPTIMIZER} == "wardropequilibriaRegularized"  && ${LEARNING} == "structured" && ${CAPACITY_INDIVIDUAL_THETA} == "1" ]]; then
																echo "Wardropequilibria can not consider capacity individual theta"
																continue
															fi
															if [[ ${LEARNING} == "structured" && ${MODEL} == "RandomForest" ]]; then
																echo "RandomForest can not learn structured"
																continue
															fi
															sbatch --job-name=tr$ABBREVIATION --cpus-per-task=$CPUS --mem="${MEMORY}mb" --export=PERCENTAGE_ORIGINAL=$PERCENTAGE_ORIGINAL,PERCENTAGE_NEW=$PERCENTAGE_NEW,SEED_NEW=$SEED_NEW,SIMULATION=$SIMULATION,MODEL=${MODEL},LEARNING=${LEARNING},NUM_TRAINING_EPOCHS=${NUM_TRAINING_EPOCHS},NUM_PERTURBATIONS=$NUM_PERTURBATIONS,PERTURBATION=$PERTURBATION,STANDARDIZE=$STANDARDIZE,TIME_VARIANT_THETA=${TIME_VARIANT_THETA},TRIP_INDIVIDUAL_THETA=${TRIP_INDIVIDUAL_THETA},OPTIMIZER=${OPTIMIZER},NUM_DISCRETE_TIME_STEPS=${NUM_DISCRETE_TIME_STEPS},CAPACITY_INDIVIDUAL_THETA=$CAPACITY_INDIVIDUAL_THETA,NUM_BUNDLED_CAPACITIES=${NUM_BUNDLED_CAPACITIES},CAPACITY_MULTIPLICATOR=${CAPACITY_MULTIPLICATOR},RUNNING_CLUSTER=${RUNNING_CLUSTER},VALIDATION_ITERATRIONS="${VALIDATION_ITERATRIONS}",EXPERIMENT=$experiment,SIMULATOR=$SIMULATOR,SD_PERTURBATION=$SD_PERTURBATION,MAX_SUCCESSIVE_AVERAGES_ITERATION=${MAX_SUCCESSIVE_AVERAGES_ITERATION},NUM_LINE_SEARCH_IT=${NUM_LINE_SEARCH_IT},EVALUATION_METRIC=${EVALUATION_METRIC},FEATURE=${FEATURE} ./runnables/cluster/run_training.sh
															if [[ ${LEARNING} == "supervised" ]]; then
																echo "supervised learning does not consider different co-layers"
																break
															fi
														done
													done
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done

		fi

		if [[ ${ACTION} == "evaluation" ]]; then
			for MODEL in ${MODELS}
			do
				for LEARNING in ${LEARNINGS}
				do
					for PERTURBATION in ${PERTURBATIONS}
					do
						for SD_PERTURBATION in ${SD_PERTURBATIONS}
						do
							for SD_PERTURBATION_EVALUATION in ${SD_PERTURBATIONS_EVALUATION}
							do
								for TIME_VARIANT_THETA in ${TIME_VARIANT_THETAS}
								do
									for TRIP_INDIVIDUAL_THETA in ${TRIP_INDIVIDUAL_THETAS}
									do
										for CAPACITY_INDIVIDUAL_THETA in ${CAPACITY_INDIVIDUAL_THETAS}
										do
											for NUM_DISCRETE_TIME_STEPS in ${NUMS_DISCRETE_TIME_STEPS}
											do
												for NUM_BUNDLED_CAPACITIES in ${NUMS_BUNDLED_CAPACITIES}
												do
													for CAPACITY_MULTIPLICATOR in ${CAPACITY_MULTIPLICATORS}
													do
														for FEATURE in ${FEATURES}
														do
															for OPTIMIZER in ${OPTIMIZERS}
															do
																for EVALUATION_PROCEDURE in ${EVALUATION_PROCEDURES}
																do
																	if [[ ${OPTIMIZER} == "addedshortestpaths" && ${LEARNING} == "structured" && (${TIME_VARIANT_THETA} == "1" || ${CAPACITY_INDIVIDUAL_THETA} == "1") ]]; then
																		echo "Addedshortestpaths does not contain time variant / capacity edges"
																		continue
																	fi
																	if [[ $((${TRIP_INDIVIDUAL_THETA} + ${TIME_VARIANT_THETA} + ${CAPACITY_INDIVIDUAL_THETA})) > 1 ]]; then
																		if ! [[ ${TIME_VARIANT_THETA} == 1 && ${CAPACITY_INDIVIDUAL_THETA} == 1 && ${TRIP_INDIVIDUAL_THETA} == 0 ]]; then
																			echo "We only investigate trip_individual_theta or time_variant_theta or capacity_individual_theta"
																			continue
																		fi
																	fi
																	if [[ ${LEARNING} == "structured" && ${MODEL} == "RandomForest" ]]; then
																		echo "RandomForest can not learn structured"
																		continue
																	fi
																	if [[ ${LEARNING} == "supervised" && ${EVALUATION_PROCEDURE} == "perturbed" ]]; then
																		echo "Perturbed evaluation not feasible for supervised learned model"
																		continue
																	fi
																	if [[ ${OPTIMIZER} == "wardropequilibria"  && ${LEARNING} == "structured" && ${CAPACITY_INDIVIDUAL_THETA} == "1" ]]; then
																		echo "Wardropequilibria can not consider capacity individual theta"
																		continue
																	fi
																	if [[ ${OPTIMIZER} == "wardropequilibriaRegularized"  && ${LEARNING} == "structured" && ${CAPACITY_INDIVIDUAL_THETA} == "1" ]]; then
																		echo "Wardropequilibria can not consider capacity individual theta"
																		continue
																	fi
																	if [[ ${LEARNING} == "structured" && ${EVALUATION_PROCEDURE} == "perturbed" ]]; then
																		if [[ ${SIMULATION} == "cutoutWorlds" ]]; then
																			CPUS=8
																			MEMORY=48000
																		else
																			CPUS=16
																			MEMORY=48000
																		fi
																	else
																		CPUS=8
																		MEMORY=48000
																	fi
																	sbatch --job-name=ev$ABBREVIATION --cpus-per-task=$CPUS --mem="${MEMORY}mb" --export=PERCENTAGE_ORIGINAL=$PERCENTAGE_ORIGINAL,PERCENTAGE_NEW=$PERCENTAGE_NEW,SEED_NEW=$SEED_NEW,SIMULATION=$SIMULATION,MODEL=${MODEL},LEARNING=${LEARNING},NUM_TRAINING_EPOCHS=${NUM_TRAINING_EPOCHS},NUM_PERTURBATIONS=$NUM_PERTURBATIONS,PERTURBATION=$PERTURBATION,STANDARDIZE=$STANDARDIZE,TIME_VARIANT_THETA=${TIME_VARIANT_THETA},TRIP_INDIVIDUAL_THETA=${TRIP_INDIVIDUAL_THETA},OPTIMIZER=${OPTIMIZER},NUM_DISCRETE_TIME_STEPS=${NUM_DISCRETE_TIME_STEPS},CAPACITY_INDIVIDUAL_THETA=$CAPACITY_INDIVIDUAL_THETA,NUM_BUNDLED_CAPACITIES=${NUM_BUNDLED_CAPACITIES},CAPACITY_MULTIPLICATOR=${CAPACITY_MULTIPLICATOR},RUNNING_CLUSTER=${RUNNING_CLUSTER},VALIDATION_ITERATRIONS="${VALIDATION_ITERATRIONS}",EXPERIMENT=$experiment,SIMULATOR=$SIMULATOR,SD_PERTURBATION=$SD_PERTURBATION,EVALUATION_PROCEDURE=${EVALUATION_PROCEDURE},MAX_SUCCESSIVE_AVERAGES_ITERATION=${MAX_SUCCESSIVE_AVERAGES_ITERATION},NUM_LINE_SEARCH_IT=${NUM_LINE_SEARCH_IT},SD_PERTURBATION_EVALUATION=${SD_PERTURBATION_EVALUATION},EVALUATION_METRIC=${EVALUATION_METRIC},FEATURE=${FEATURE} ./runnables/cluster/run_evaluation.sh
																	if [[ ${LEARNING} == "supervised" ]]; then
																		echo "supervised learning does not consider different co-layers"
																		break
																	fi
																done
															done
														done
													done
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		fi
	done
done


