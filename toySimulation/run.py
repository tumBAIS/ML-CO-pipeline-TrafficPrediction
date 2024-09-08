import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-1])
sys.path.append(main_directory)
import random
import numpy as np
from toySimulation.src.util import read_in_scenario
from toySimulation.againstsupervisedSimulation.run_simulation import run_simulation as againstsupervisedSimulation
from toySimulation.easySimulation.run_simulation import run_simulation as easySimulation
from toySimulation.easySimulationEquilibrium.run_simulation import run_simulation as easySimulationEquilibrium
from toySimulation.randomSimulationEquilibrium.run_simulation import run_simulation as randomSimulationEquilibrium
from surrogate.src.util import get_directory_instances, create_directory, save_json
from surrogate.src import config

SIMULATIONS = {"againstsupervisedSimulation": againstsupervisedSimulation, "easySimulation": easySimulation,
               "easySimulationEquilibrium": easySimulationEquilibrium, "randomSimulationEquilibrium": randomSimulationEquilibrium}

if __name__ == '__main__':
    args = config.parser.parse_args()
    # set seeds
    random.seed(args.seed_algorithm)
    np.random.seed(args.seed_algorithm)

    print("Read in scenario")
    scenario = read_in_scenario(args)
    scenario["o_d_pairs_starting_time"] = [scenario["minimum_start"] for o_d_pair in scenario["o_d_pairs"]]

    print("Run simulation")
    results = SIMULATIONS[args.simulator](args, scenario)

    print("Save simulation results")
    dir_instances = "../surrogate/" + get_directory_instances(args).split("./")[1]
    create_directory(f"{dir_instances}/")
    save_json(scenario, f"{dir_instances}/s-{args.seed}.json")
