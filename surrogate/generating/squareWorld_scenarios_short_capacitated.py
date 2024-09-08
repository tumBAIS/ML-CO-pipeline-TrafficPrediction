import os
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
from surrogate.src import config
from surrogate.generating.squareWorld_scenarios_short import generate_random_graph, generate_random_population
from surrogate.src.util import save_json, get_scenario_acronym, create_directory


def get_link_capacities(args, scenario):
    if args.verbose:
        plt.hist(scenario["link_length"], bins=10)
        plt.show()

    capacities = np.full(len(scenario["links_id"]), 100.0)
    capacities[scenario["link_length"] < 2000] = 60.0
    capacities[scenario["link_length"] < 1000] = 20.0
    capacities[scenario["link_length"] < 500] = 5.0
    return capacities



def save_random_scenario(args, graph, population, minimum_start, maximum_end):

    scenario = {"work_x": population.work_distribution[0],
                   "work_y": population.work_distribution[1],
                   "home_x": population.home_distribution[0],
                   "home_y": population.home_distribution[1],
                   "go_to_work": population.go_to_work,
                   "go_to_home": population.go_to_home,
                   "o_d_pairs": population.o_d_pairs}

    scenario.update(graph)
    scenario["minimum_start"] = minimum_start
    scenario["maximum_end"] = maximum_end
    freespeed = 13.88
    scenario["link_freespeed"] = np.array([freespeed] * len(scenario["links_id"]))
    scenario["link_permlanes"] = np.array([1.0] * len(scenario["links_id"]))

    maximum_length = 0.05 * (maximum_end - minimum_start) * freespeed
    distance_converter = maximum_length / max(np.sqrt(
        (np.array(scenario["nodes_x"])[scenario["link_from"]] - np.array(scenario["nodes_x"])[scenario["link_to"]]) ** 2 +
        (np.array(scenario["nodes_y"])[scenario["link_from"]] - np.array(scenario["nodes_y"])[scenario["link_to"]]) ** 2))

    scenario["link_length"] = distance_converter * np.sqrt(    # 4000.0
        (np.array(scenario["nodes_x"])[scenario["link_from"]] - np.array(scenario["nodes_x"])[scenario["link_to"]]) ** 2 +
        (np.array(scenario["nodes_y"])[scenario["link_from"]] - np.array(scenario["nodes_y"])[scenario["link_to"]]) ** 2)

    scenario["link_capacity"] = get_link_capacities(args=args, scenario=scenario)

    create_directory(f"{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}")
    save_json(scenario, f'{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/scenario.json')



if __name__ == '__main__':
    args = config.parser.parse_args()
    args.simulation = "squareWorlds_short_capacitated"
    np.random.seed(args.seed)
    random.seed(args.seed)

    minimum_start = 8 * 3600
    maximum_end = 9 * 3600

    fig, ax = plt.subplots(figsize=(10, 10))
    graph = generate_random_graph(args=args)
    population = generate_random_population(args=args, G=graph, minimum_start=minimum_start)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()
    #plt.show()

    save_random_scenario(args=args, graph=graph, population=population, minimum_start=minimum_start, maximum_end=maximum_end)