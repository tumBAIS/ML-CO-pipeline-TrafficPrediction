import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
import numpy as np
import random
import pandas as pd
from surrogate.src.util import load_json, save_json, get_directory_instances, create_directory, get_scenario_acronym
from surrogate.generating.cutoutWorld_training_data import read_in_scenario


def read_in_population(trips):
    work_locations = trips[trips["end_activity_type"].astype(str).str.startswith('w')]
    work_x = work_locations["end_x"].to_numpy()
    work_y = work_locations["end_y"].to_numpy()
    home_locations = trips[trips["end_activity_type"].astype(str).str.startswith('h')]
    home_x = home_locations["end_x"].to_numpy()
    home_y = home_locations["end_y"].to_numpy()
    go_to_work = pd.to_timedelta(work_locations["dep_time"]).dt.seconds.to_numpy()
    go_to_home = pd.to_timedelta(home_locations["dep_time"]).dt.seconds.to_numpy()
    return {"work_x": work_x, "work_y": work_y, "home_x": home_x, "home_y": home_y, "go_to_work": go_to_work, "go_to_home": go_to_home}


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

    minimum_start = 8 * 3600
    maximum_end = 20 * 3600
    half_time = 15 * 3600

    vehicle_trips_to_work = event_reader.events[(event_reader.events["time"] > minimum_start) & (event_reader.events["time"] < half_time)].groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()
    vehicle_trips_to_home = event_reader.events[(event_reader.events["time"] > half_time) & (event_reader.events["time"] < maximum_end)].groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()

    mapping = []
    o_d_pairs = []
    commodities = []
    o_d_pairs_starting_time = []
    o_d_pairs_ending_time = []

    edge_flow_times = np.array(scenario["link_length"]) / np.array(scenario["link_freespeed"])

    commodity = 0
    for vehicle_trips in [vehicle_trips_to_work, vehicle_trips_to_home]:
        for vehicle_trip in vehicle_trips.values():
            for (link, time) in vehicle_trip:
                mapping.append({"time": time, "link": mapping_link_old_id[link], "commodity": commodity})
            start_node = scenario["link_from"][mapping_link_old_id[int(vehicle_trip[0][0])]]
            end_node = scenario["link_to"][mapping_link_old_id[int(vehicle_trip[-1][0])]]
            o_d_pairs.append((start_node, end_node))
            o_d_pairs_starting_time.append(vehicle_trip[0][1])
            o_d_pairs_ending_time.append(vehicle_trip[-1][1] + edge_flow_times[mapping_link_old_id[int(link)]])
            commodities.append(commodity)
            commodity += 1

    mapping = pd.DataFrame(mapping)

    scenario["o_d_pairs"] = o_d_pairs
    scenario["commodities"] = commodities
    scenario["o_d_pairs_starting_time"] = o_d_pairs_starting_time
    scenario["o_d_pairs_ending_time"] = o_d_pairs_ending_time
    scenario["solution_time"] = np.array(mapping.time)
    scenario["solution_link"] = np.array(mapping.link)
    scenario["solution_commodity"] = np.array(mapping.commodity)

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