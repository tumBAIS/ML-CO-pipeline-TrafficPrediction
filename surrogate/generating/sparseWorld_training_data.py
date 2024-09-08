import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
from surrogate.src.util import save_json, Graph, get_directory_instances, create_directory, get_scenario_acronym, get_scenario_directory_acronym
import random
import numpy as np
import collections
from surrogate.generating.cutoutWorld_training_data import read_in_scenario, read_in_population


def read_in_o_d_pairs(event_reader, relevant_links, scenario):
    graph = Graph()
    graph.add_nodes_from_instance(scenario)
    graph.add_links_from_instance(scenario)

    # RETRIEVE VEHILCE LINK CONNECTION
    vehicle_trips = event_reader.events.groupby('vehicle')[["link", "time"]].apply(lambda x: x.values.tolist()).to_dict()
    mappling_OLDID_ID = dict(zip(relevant_links["link_id_OLD"], relevant_links["link_id"]))

    # RETRIEVE O-D PAIRS
    routes = []
    for vehicle, route_links in vehicle_trips.items():
        links = np.array([mappling_OLDID_ID[link] for link in np.array(route_links)[:, 0].astype(int)])
        times = np.array(route_links)[:, 1].astype(float)

        if len(links) > 1:
            # Always splitting when links are not connected
            boolean_split = np.array([graph.links[links[idx]].to_node != graph.links[links[idx+1]].from_node for idx, link in enumerate(links[:-1])])
            # Always splitting when time above
            time_comparison = (times[1:] - times[:-1])
            boolean_split = boolean_split | (time_comparison > 60 * 5)
            # Always splitting when same links next to each other
            boolean_split = boolean_split | (links[1:] == links[:-1])
            # Always splitting when start node and end node are the same
            nodes_path_from_node = np.array([graph.links[link].from_node for link in links] + [graph.links[links[-1]].to_node])
            nodes_path_to_node = np.array([graph.links[links[-1]].from_node] + [graph.links[link].to_node for link in links])
            nodes_from_node_identical = np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1) == np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1).T
            nodes_to_node_identical = np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1) == np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1).T
            nodes_from_to_node_identical = np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1) == np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1).T
            nodes_to_from_node_identical = np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1) == np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1).T
            nodes_identical_idxs = np.where(nodes_from_node_identical | nodes_to_node_identical | nodes_from_to_node_identical | nodes_to_from_node_identical)
            for (idxs_x, idxs_y) in zip(nodes_identical_idxs[0], nodes_identical_idxs[1]):
                if sum(boolean_split[idxs_x: idxs_y-1]) == 0 and idxs_y-1 > idxs_x:
                    boolean_split[idxs_x + np.argmax(time_comparison[idxs_x: idxs_y-1])] = True
        else:
            boolean_split = []
        # We split routes
        vehicle_route = []
        for split, link in zip(boolean_split, links[:-1]):
            vehicle_route.append(link)
            if split:
                routes.append(vehicle_route)
                vehicle_route = []
        vehicle_route.append(links[-1])
        routes.append(vehicle_route)

    o_d_pairs = [(graph.links[route[0]].from_node, graph.links[route[-1]].to_node) for route in routes]
    print(f"Number of od pairs: {len(o_d_pairs)}")
    return o_d_pairs, routes


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

    # READ IN O-D PAIRS
    o_d_pairs_trips = trip_reader.get_o_d_pairs(mapping_link_old_id, Graph(scenario))
    o_d_pairs_plans = plan_reader.get_o_d_pairs(mapping_link_old_id, Graph(scenario))
    o_d_pairs_events, links_in_solutions = read_in_o_d_pairs(event_reader, matsim_scenario.links, scenario)

    scenario["o_d_pairs"] = {"plans": o_d_pairs_plans, "trips": o_d_pairs_trips, "events": o_d_pairs_events}[od_calculator_name]

    # READ IN POPULATION
    population = read_in_population(trip_reader.trips)
    scenario["work_x"] = population["work_x"]
    scenario["work_y"] = population["work_y"]
    scenario["home_x"] = population["home_x"]
    scenario["home_y"] = population["home_y"]
    scenario["go_to_work"] = population["go_to_work"]
    scenario["go_to_home"] = population["go_to_home"]
    return scenario


def generate_training_data(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Read in scenario
    matsim_network, event_reader, trip_reader, plan_reader = read_in_scenario(network_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/network.xml',
                                                                 event_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_events.xml.gz',
                                                                 trip_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_trips.csv.gz',
                                                                 plan_path=f'../matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/plans.xml')
    scenario = prepare_training_data(args, matsim_scenario=matsim_network, event_reader=event_reader, trip_reader=trip_reader, plan_reader=plan_reader)

    dir_instances = get_directory_instances(args)
    create_directory(f"{dir_instances}/")
    save_json(scenario, f"{dir_instances}/s-{args.seed}.json")
