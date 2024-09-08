import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
import numpy as np
import random
from surrogate.src.util import load_json, save_json, get_directory_instances, create_directory, get_scenario_acronym
from surrogate.generating.cutoutWorld_training_data import read_in_scenario, read_in_population, read_in_o_d_pairs


def prepare_training_data(args, matsim_scenario, event_reader, trip_reader, plan_reader, od_calculator_name="plans"):

    # RESET IDs
    matsim_scenario.nodes = matsim_scenario.nodes.rename(columns={"node_id": "node_id_OLD"})
    matsim_scenario.nodes["node_id"] = matsim_scenario.nodes.reset_index().index
    matsim_scenario.links = matsim_scenario.links.merge(matsim_scenario.nodes[["node_id", "node_id_OLD"]], how="left", left_on="from_node", right_on="node_id_OLD").rename(columns={"node_id": "from_node_id"})
    matsim_scenario.links = matsim_scenario.links.merge(matsim_scenario.nodes[["node_id", "node_id_OLD"]], how="left", left_on="to_node", right_on="node_id_OLD").rename(columns={"node_id": "to_node_id"})
    matsim_scenario.links = matsim_scenario.links.drop(columns=["from_node", "to_node", "node_id_OLD_x", "node_id_OLD_y"]).rename(columns={"from_node_id": "from_node", "to_node_id": "to_node"})
    matsim_scenario.links = matsim_scenario.links.rename(columns={"link_id": "link_id_OLD"})
    matsim_scenario.links["link_id"] = matsim_scenario.links.reset_index().index
    mapping_link_old_id = dict(zip(matsim_scenario.links["link_id_OLD"], matsim_scenario.links["link_id"]))

    scenario = {}
    scenario["nodes_x"] = np.array(matsim_scenario.nodes["x"])
    scenario["nodes_y"] = np.array(matsim_scenario.nodes["y"])
    scenario["nodes_id"] = np.array(matsim_scenario.nodes["node_id"])
    scenario["link_from"] = np.array(matsim_scenario.links["from_node"])
    scenario["link_to"] = np.array(matsim_scenario.links["to_node"])
    scenario["links_id"] = np.array(matsim_scenario.links["link_id"])

    scenario["link_length"] = np.array(matsim_scenario.links["length"])
    scenario["link_freespeed"] = np.array(matsim_scenario.links["freespeed"])
    scenario["link_capacity"] = np.array(matsim_scenario.links["capacity"])
    scenario["link_permlanes"] = np.array(matsim_scenario.links["permlanes"])

    vehicle_trips = event_reader.events.groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()

    # READ IN O-D PAIRS
    route_data, links_in_solutions = read_in_o_d_pairs(event_reader, matsim_scenario.links, scenario)
    scenario.update(route_data)

    # READ IN POPULATION
    population = read_in_population(trip_reader.trips)
    scenario["work_x"] = population["work_x"]
    scenario["work_y"] = population["work_y"]
    scenario["home_x"] = population["home_x"]
    scenario["home_y"] = population["home_y"]
    scenario["go_to_work"] = population["go_to_work"]
    scenario["go_to_home"] = population["go_to_home"]
    scenario["minimum_start"] = min(scenario["solution_time"])
    scenario["maximum_end"] = max(scenario["solution_time"]) + 1000
    return scenario


def generate_training_data(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Read in scenario
    matsim_network, event_reader, trip_reader, plan_reader = read_in_scenario(network_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/network.xml',
                                                                 event_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_events.xml.gz',
                                                                 trip_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_trips.csv.gz',
                                                                 plan_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/plans.xml')
    scenario = prepare_training_data(args, matsim_scenario=matsim_network, event_reader=event_reader, trip_reader=trip_reader, plan_reader=plan_reader, od_calculator_name="plans")

    dir_instances = get_directory_instances(args)
    create_directory(f"{dir_instances}/")
    save_json(scenario, f"{dir_instances}/s-{args.seed}.json")
