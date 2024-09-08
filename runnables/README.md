# Runnables


This directory contains all scripts to run the code.

- Scripts to create data points: Each data point derives from a combination of a scenario and an oracle. The scenario defines the contextual information including the street network and the population, and the oracle defines the mapping from the contextual information to the true traffic flow. 
    - **run_againstsupervisedSimulation_squareWorlds_short.sh:** scenario: squareWorlds_short | oracle: againstsupervisedSimulation
    - **run_easySimulation_squareWorlds_short.sh:** scenario: squareWorlds_short | oracle: easySimulation
    - **run_easySimulationEquilibrium_squareWorlds_short.sh:** scenario: squareWorlds_short | oracle: easySimulationEquilibrium
    - **run_entropy:** possible scenarios.sh: low_entropy, high_entropy, low_entropy_street, high_entropy_street | possible oracles: easySimulation, againstsupervisedSimulation, easySimulationEquilibrium, randomSimulationEquilibrium --> comment the scenarios / oracles that you want to ignore
    - **run_matsim_againstsupervisedWorld.sh:** scenario: againstsupervisedWorld | oracle: MATSim
    - **run_matsim_cutoutWorlds.sh**: scenario: cutoutWorlds | oracle: MATSim
    - **run_matsim_cutoutWorldsCap.sh**: scenario: cutoutWorldsCap | oracle: MATSim
    - **run_matsim_cutoutWorldsSpeed.sh**: scenario: cutoutWorldsSpeed | oracle: MATSim
    - **run_matsim_districtWorlds.sh**: scenario: districtWorlds | oracle: MATSim
    - **run_matsim_districtWorldsArt.sh**: scenario: districtWorldsArt | oracle: MATSim
    - **run_matsim_smallWorlds.sh**: scenario: smallWorlds | oracle: MATSim
    - **run_matsim_smallWorlds_pricing**: scenario: smallWorlds_pricing | oracle: MATSim
    - **run_matsim_smallWorlds_short**: scenario: smallWorlds_short | oracle: MATSim
    - **run_matsim_sparseWorlds**: scenario: sparseWorlds | oracle: MATSim
    - **run_matsim_squareWorlds_short**: scenario: squareWorlds_short | oracle: MATsim
    - **run_matsim_squareWorlds_short_capacitated**: scenario: squareWorlds_short_capacitated | oracle: MATsim
    - **run_randomSimulationEquilibrium_squareWorlds_short**: scenario: squareWorlds_short | oracle: randomSimulationEquilibrium

- Scripts to train a COAML pipeline
  - **run_training.sh**: trains a COAML pipeline. Define the parameters for training in the script.

- Scripts to evaluate a COAML pipeline
  - **run_evaluation.sh**: evaluates a trained COAML pipeline. Define the parameters for evaluation in the script.

- Scripts to run code on the cluster
  - **master.sh:** Master file to bash on the cluster. 1) Define the parameters, e.g., outcomment the experiment and define the hyperparameters. 2) Outcomment if you want to train or to evaluate.
  - **run_evaluation.sh:** Script called from master.sh -> evaluating the COAML pipeline
  - **run_training.sh:** Script called from master.sh -> train the COAML pipeline
  - **srun_test.sh:** Testing script for testing in interactive model
