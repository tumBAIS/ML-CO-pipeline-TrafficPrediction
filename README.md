# WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning

This repository comprises the code to learn a Combinatorial Optimization Augmented Machine Learning (COAML) pipeline that predicts traffic flow from contextual information.

This method is proposed in:
> Kai Jungel, Dario Paccagnan, Axel Parmentier, and Maximilian Schiffer. WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning. arXiv preprint: [arxiv:2410.06656](https://arxiv.org/abs/2410.06656), 2024.

This repository contains all relevant scripts and data sets to reproduce the results from the paper.  
To run the code for reproducing the results we assume using *slurm*, and require a Gurobi license.  
We used Gurobi version 10.0.0.  
We used Python version 3.8.10.  
We run the code on a Linux system.  

The structure of this repository is as follows:  
*./matsim-berlin:* Multi Agent Transport Simulation to simulate oracle scenarios  
*./runnables:* comprises all scripts to execute the code  
*./surrogate:* code of the Combinatorial Optimization Augmented Machine Learning (COAML) pipeline  
*./toySimulation:* code to construct toy simulations that create toy scenarios  

Each of these folders comprises respective README.md files.

To reproduce the results from the paper follow these steps:
## 1. Install MATSim
In the matsim-berlin directory, install MATSim: https://github.com/matsim-scenarios/matsim-berlin

## 2. Install python dependencies
We recommend to install the packages in a virtual environment called `venv`.
```bash 
pip install -r requirements.txt
```

## 3. Create scenarios
1. Install the random road network creator in `./surrogate/generating`: https://github.com/arun1729/road-network
2. To create a scenario run
```bash 
bash runnables/scenario_name.sh
```

## 4. Train and evaluate the COAML pipeline
1. Recommendation: Run the training on a cluster using `slurm`.
2. We provide pre-caculated instances in `surrogate/data`. Please unpack them, if you did not run step *3. Create scenarios*
3. On the cluster, define the parameters in the `runnables/cluster/master.sh` file.
4. On the cluster, in `runnables/cluster/master.sh` uncomment the training or evaluation, depending on what you want to run.
5. On the cluster, run 
```bash 
bash runnables/cluster/master.sh
```

## 5. Visualization of results
1. Unpack `./surrogate/results_paper.zip` and `./surrogate/data/read_in_best_learning_iteration.zip`
1. Move to `./surrogate/visualization` and run:
   - `learning_results.py` to compare the benchmark models
   - `input_data.py` to visualize the scenarios
   - `output_data.py` to visualize the traffic flow

Please note that the benchmark names in the paper were updated: multicommodityflow / MCFP -> CL, wardropequilibria / WE -> PL, wardropequilibriaRegularized / WE-reg -> ER
