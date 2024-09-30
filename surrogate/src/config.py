import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--seed_algorithm", type=int, default=1)
parser.add_argument("--model", type=str, choices=["GNN", "NN", "Base", "RandomForest", "Linear"], default="NN")
parser.add_argument("--learning", type=str, choices=["supervised", "structured", "structured_bfgs"], default="supervised")
parser.add_argument("--dir_models", type=str, default="./pipelines/prediction/models/trained_models/")
parser.add_argument("--dir_standardization", type=str, default="./pipelines/prediction/models/standardizations/")
parser.add_argument("--dir_learning", type=str, default="./pipelines/prediction/models/learning_files/")
parser.add_argument("--dir_instances", type=str, default="./data")
parser.add_argument("--simulation", type=str, choices=["cutoutWorlds", "cutoutWorlds_mixed", "cutoutWorldsCap", "cutoutWorldsSpeed", "cutoutWorldsWithoutBoundary",
                                                       "smallWorlds", "sparseWorlds", "districtWorlds", "districtWorldsArt", "smallWorlds_pricing",
                                                       "smallWorlds_short", "squareWorlds_short", "squareWorlds_short_capacitated", "againstsupervisedWorld",
                                                       "squareWorlds_short", "low_entropy", "high_entropy", "low_entropy_street", "high_entropy_street"],
                    default="districtWorldsArt")
parser.add_argument("--dir_results", type=str, default="./results/")
parser.add_argument("--dir_visualization", type=str, default="./visualizing/visualization/")
parser.add_argument("--read_in_iteration", type=int, default=3)
parser.add_argument("--num_training_epochs", type=int, default=50)  # 50
parser.add_argument("--sd_perturbation", type=float, default=1)  # 1
parser.add_argument("--num_perturbations", type=int, default=2)  # 50
parser.add_argument("--standardize", type=int, default=1)
parser.add_argument("--mode", type=str, choices=["Train", "Test", "Validate", "Bayesian"], default="Validate")
parser.add_argument("--perturbation", type=str, choices=["additive", "multiplicative"], default="additive")
parser.add_argument("--verbose", type=int, choices=[0, 1], default=0)
parser.add_argument("--num_cpus", type=int, default=8)

# SIMULATION
parser.add_argument("--simulator", type=str, choices=["matsim", "againstsupervisedSimulation", "easySimulation", "easySimulationEquilibrium", "randomSimulationEquilibrium"],
                    default="matsim")


# GENERATE TRAINING DATA
parser.add_argument("--percentage_original", type=int, default=1)
parser.add_argument("--percentage_new", type=float, default=1.0)
parser.add_argument("--seed_new", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
# cutout
parser.add_argument("--cutout_seeds", type=int, nargs='*', default=[1])
# small world
parser.add_argument("--population_size", type=int, default=100)
parser.add_argument("--num_nodes", type=int, default=100)
# entropy
parser.add_argument("--dimension_entropy", type=int, default=6)


# EVALUATION
parser.add_argument("--dir_best_iteration_saver", type=str, default="./data/read_in_best_learning_iteration/")
parser.add_argument("--experiment_name", type=str, default="numDiscreteTimeStepsWith")
parser.add_argument("--evaluation_procedure", type=str, choices=["original", "perturbed"], default="original")
parser.add_argument("--evaluation_metric", type=str, choices=["original", "timedist"], default="original")
parser.add_argument("--sd_perturbation_evaluation", type=float, default=1)  # 1

#LEARNING
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--max_time", type=lambda s: datetime.datetime.strptime(s, '%H:%M:%S'), default="20:00:00")
parser.add_argument("--feature", type=str, default="''")


# OPTIMIZATION
parser.add_argument("--time_variant_theta", type=int, choices=[0, 1], default=0)
parser.add_argument("--trip_individual_theta", type=int, choices=[0, 1], default=0)
parser.add_argument("--capacity_individual_theta", type=int, choices=[0, 1], default=0)
parser.add_argument("--num_discrete_time_steps", help="number of discrete time steps for MCFP time-expansion", type=int, default=40)
parser.add_argument("--num_bundled_capacities", help="number of max capacities on one arc", type=int, default=3)  #3
parser.add_argument("--capacity_multiplicator", help="max capacity per arc improvement comparison to true traffic count in %", type=str, default="inf")
parser.add_argument("--co_optimizer", type=str, choices=["addedshortestpaths", "multicommodityflow", "multicommodityflow_uncapacitated",
                                                         "wardropequilibria", "wardropequilibriaRegularized"],
                    default="multicommodityflow")
parser.add_argument("--obtimization_gap", type=str, default="default")


# SUCCESSIVEAVERAGES
parser.add_argument("--max_successive_averages_iteration", type=int, default=20)
# FRANKWOLFE
parser.add_argument("--num_line_search_it", type=int, default=5)

# BAYESIAN OPTIMIZATION
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--max_evals", type=int, default=3)
parser.add_argument("--num_matsim_iterations", type=int, default=2)
parser.add_argument("--num_cand", type=int, default=100)
parser.add_argument("--num_new_points", type=int, default=1)

parser.add_argument("--experimental_design", type=str, choices=["latin_hypercube", "default"], default="default")
parser.add_argument("--problem", type=str, choices=["ackley", "pricing"], default="pricing")
parser.add_argument("--surrogate", type=str, choices=["gp_regressor", "ml-co"], default="ml-co")
parser.add_argument("--strategy", type=str, choices=["srbf_strategy", "default", "sop_strategy"], default="default")
parser.add_argument("--bayesian_verbose", type=int, choices=[0, 1], default=0)
parser.add_argument("--dir_logs", type=str, default="./logs/")
parser.add_argument("--dir_bayesian_results", type=str, default="./results/")





