import copy
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
from surrogate.visualizing.event_reader import EventReader, TripReader, PlanReader
from surrogate.src.util import save_json, Graph, get_directory_instances, create_directory, get_scenario_acronym, get_scenario_directory_acronym
import matsim
import random
import numpy as np
import pandas as pd
import collections


def read_in_scenario(network_path, event_path, trip_path, plan_path=None):
    # READ IN EVENTS
    event_reader = EventReader(path_input=event_path)
    # READ IN TRIPS
    trip_reader = TripReader(path_input=trip_path)
    # READ IN NETWORK
    matsim_network = matsim.read_network(network_path)
    # 1. Delete all public transport links
    matsim_network.links = matsim_network.links[~matsim_network.links['link_id'].astype(str).str.startswith('pt_')]
    # 2. Delete all public transport nodes
    matsim_network.nodes = matsim_network.nodes[~matsim_network.nodes['node_id'].astype(str).str.startswith('pt_')]
    # 3. Set the link count
    matsim_network.links["link_id"] = pd.to_numeric(matsim_network.links["link_id"])
    if plan_path:
        plan_reader = PlanReader(path_input=plan_path)
    else:
        plan_reader = None
    return matsim_network, event_reader, trip_reader, plan_reader


def print_route_network(route, graph, x, y):
    for link in graph.links.values():
        if link.link_id in route:
            link.weight = 1
        else:
            link.weight = 0
    graph.draw(x, y)


def read_in_o_d_pairs(event_reader, relevant_links, scenario):
    graph = Graph()
    graph.add_nodes_from_instance(scenario)
    graph.add_links_from_instance(scenario)

    # RETRIEVE VEHILCE LINK CONNECTION
    events_in_area_of_interest = event_reader.events[["vehicle", "link", "time"]].merge(relevant_links[["link_id_OLD", "link_id", "permlanes"]], how="left", left_on="link", right_on="link_id_OLD")
    events_in_area_of_interest = events_in_area_of_interest.dropna(axis=0).drop(columns=["link_id_OLD"])
    vehicle_trips_all = events_in_area_of_interest.groupby('vehicle')[["link_id", "time", "permlanes"]].apply(lambda x: x.values.tolist()).to_dict()

    vehicle_trips = {}
    for vehicle, route_links in vehicle_trips_all.items():
        permlanes_checker = []
        for link in route_links:
            permlanes_checker.append(link[2] > 1)
        if sum(permlanes_checker) / len(permlanes_checker) < 0.4:
            vehicle_trips[vehicle] = route_links
        print(f"Original num. vehicles: {len(vehicle_trips_all)} | New num. vehicles: {len(vehicle_trips)}")


    routes = []
    routes_time = []
    for vehicle, route_links in vehicle_trips.items():
        links = np.array(route_links)[:, 0].astype(int)
        times = np.array(route_links)[:, 1].astype(float)
        times_ext = np.array([times[0]] + list(times) + [times[-1]])

        if len(links) > 1:
            nodes_path_from_node = np.array([graph.links[link].from_node for link in links] + [graph.links[links[-1]].to_node])
            nodes_path_to_node = np.array([graph.links[links[0]].from_node] + [graph.links[link].to_node for link in links])
            time_comparison = (times_ext[1:] - times_ext[:-1])

            # Always splitting when links are not connected
            boolean_split = (nodes_path_from_node != nodes_path_to_node)
            # Always splitting when start node and end node are the same
            nodes_from_node_identical = np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1) == np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1).T
            nodes_to_node_identical = np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1) == np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1).T
            nodes_from_to_node_identical = np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1) == np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1).T
            nodes_to_from_node_identical = np.repeat(nodes_path_to_node[:, np.newaxis], len(nodes_path_to_node), axis=1) == np.repeat(nodes_path_from_node[:, np.newaxis], len(nodes_path_from_node), axis=1).T
            nodes_identical = nodes_from_node_identical | nodes_to_node_identical | nodes_from_to_node_identical | nodes_to_from_node_identical
            path = np.array([(graph.links[link].from_node, graph.links[link].to_node) for link in links])
            nodes_identical_idxs = np.where(nodes_from_node_identical | nodes_to_node_identical | nodes_from_to_node_identical | nodes_to_from_node_identical)
            nodes_identical_idxs = [(x, y) for x, y in zip(nodes_identical_idxs[0], nodes_identical_idxs[1])]
            nodes_identical_idxs.sort(key=lambda x: np.abs(x[1] - x[0]))

            for (idxs_x, idxs_y) in nodes_identical_idxs:
                if idxs_y > idxs_x+1:
                    if sum(boolean_split[idxs_x+1: idxs_y]) == 0:
                        boolean_split[idxs_x+1 + np.argmax(time_comparison[idxs_x+1: idxs_y])] = True

            for idx in np.where(time_comparison > 60 * 5)[0]:
                if boolean_split[idx] + boolean_split[idx-1] + boolean_split[idx+1] == 0:
                    boolean_split[idx] = True
        else:
            boolean_split = []
        # We split routes
        vehicle_route = []
        vehicle_route_time = []
        for split, link, time in zip(boolean_split[1:], links, times):
            vehicle_route.append(link)
            vehicle_route_time.append(time)
            if split:
                routes.append(vehicle_route)
                routes_time.append(vehicle_route_time)
                vehicle_route = []
                vehicle_route_time = []
        if len(vehicle_route) > 0:
            routes.append(vehicle_route)
            routes_time.append(vehicle_route_time)

    edge_flow_times = np.array(scenario["link_length"]) / np.array(scenario["link_freespeed"])

    o_d_pairs = [(graph.links[route[0]].from_node, graph.links[route[-1]].to_node) for route in routes]
    o_d_pairs_starting_time = [route_time[0] for route_time in routes_time]
    o_d_pairs_ending_time = [route_time[-1] + edge_flow_times[int(route[-1])] for route, route_time in zip(routes, routes_time)]
    routes_commodity = [[commodity_idx for links in route] for route, commodity_idx in zip(routes, np.arange(len(routes)))]
    print(f"Number of od pairs: {len(o_d_pairs)}")

    route_data = {"commodities": np.arange(len(o_d_pairs)),
                  "o_d_pairs": o_d_pairs,
                  "solution_link": np.array([x for xs in routes for x in xs]),
                  "solution_time": np.array([x for xs in routes_time for x in xs]),
                  "solution_commodity": np.array([x for xs in routes_commodity for x in xs]),
                  "o_d_pairs_starting_time": o_d_pairs_starting_time,
                  "o_d_pairs_ending_time": o_d_pairs_ending_time}

    return route_data, routes


def read_in_population(trips):
    work_locations = trips[trips["end_activity_type"].astype(str).str.startswith('work_')]
    work_x = work_locations["end_x"].to_numpy()
    work_y = work_locations["end_y"].to_numpy()
    home_locations = trips[trips["end_activity_type"].astype(str).str.startswith('home_')]
    home_x = home_locations["end_x"].to_numpy()
    home_y = home_locations["end_y"].to_numpy()
    go_to_work = pd.to_timedelta(work_locations["dep_time"]).dt.seconds.to_numpy()
    go_to_home = pd.to_timedelta(home_locations["dep_time"]).dt.seconds.to_numpy()
    return {"work_x": work_x, "work_y": work_y, "home_x": home_x, "home_y": home_y, "go_to_work": go_to_work, "go_to_home": go_to_home}


def split_scenario(args, matsim_scenario, event_reader, trip_reader):
    x = np.array(matsim_scenario.nodes["x"])
    y = np.array(matsim_scenario.nodes["y"])
    scenarios = []
    for seed in args.cutout_seeds:
        random.seed(seed)
        center = random.sample(list(matsim_scenario.nodes.to_dict("records")), 1)[0]
        distance = np.sqrt((x - center["x"])**2 + (y - center["y"])**2)
        relevant_nodes = matsim_scenario.nodes[distance < np.quantile(distance, 0.01)]
        relevant_nodes_id = np.array(relevant_nodes["node_id"])
        relevant_links = matsim_scenario.links[[((link["from_node"] in relevant_nodes_id) and (link["to_node"] in relevant_nodes_id)) for link in matsim_scenario.links.to_dict("records")]]
        # 1. Set new node id
        relevant_nodes = relevant_nodes.rename(columns={"node_id": "node_id_OLD"})
        relevant_nodes["node_id"] = relevant_nodes.reset_index().index

        # 2. Set the link id
        relevant_links = relevant_links.merge(relevant_nodes[["node_id", "node_id_OLD"]], how="left", left_on="from_node", right_on="node_id_OLD").rename(columns={"node_id": "from_node_id"})
        relevant_links = relevant_links.merge(relevant_nodes[["node_id", "node_id_OLD"]], how="left", left_on="to_node", right_on="node_id_OLD").rename(columns={"node_id": "to_node_id"})
        relevant_links = relevant_links.drop(columns=["from_node", "to_node", "node_id_OLD_x", "node_id_OLD_y"]).rename(columns={"from_node_id": "from_node", "to_node_id": "to_node"})
        relevant_links = relevant_links.rename(columns={"link_id": "link_id_OLD"})
        relevant_links["link_id"] = relevant_links.index

        scenario = {}
        scenario["nodes_x"] = np.array(relevant_nodes["x"])
        scenario["nodes_y"] = np.array(relevant_nodes["y"])
        scenario["nodes_id"] = np.array(relevant_nodes["node_id"])
        scenario["link_from"] = np.array(relevant_links["from_node"])
        scenario["link_to"] = np.array(relevant_links["to_node"])
        scenario["links_id"] = np.array(relevant_links["link_id"])

        scenario["link_length"] = np.array(relevant_links["length"])
        scenario["link_freespeed"] = np.array(relevant_links["freespeed"])
        scenario["link_capacity"] = np.array(relevant_links["capacity"])
        scenario["link_permlanes"] = np.array(relevant_links["permlanes"])

        # READ IN O-D PAIRS
        route_data, links_in_solutions = read_in_o_d_pairs(event_reader, relevant_links, scenario)
        scenario.update(route_data)

        # CHECK THAT IT IS VALID SOLUTION
        for od_pair, links_in_solution in zip(scenario["o_d_pairs"], links_in_solutions):
            graph = Graph()
            graph.set_nodes_from_instance(scenario)
            graph.set_links_from_instance(scenario)
            for link in graph.links.values():
                if link.link_id in links_in_solution:
                    link.weight = 1
                else:
                    link.weight = 100
            graph.reset_fast_calculation_files()
            shortest_path = graph.calculate_shortest_path(origin=od_pair[0], destination=od_pair[1])
            try:
                assert collections.Counter(shortest_path) == collections.Counter(links_in_solution)
            except:
                raise Exception

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
        scenarios.append(scenario)
    return scenarios


def generate_training_data(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Read in scenario
    matsim_network, event_reader, trip_reader, _ = read_in_scenario(network_path=f"../matsim-berlin/scenarios/cutoutWorlds/{get_scenario_acronym(args)}/network.xml",
                                                                 event_path=f'../matsim-berlin/scenarios/cutoutWorlds/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_events.xml.gz',
                                                                 trip_path=f'../matsim-berlin/scenarios/cutoutWorlds/{get_scenario_acronym(args)}/output/{get_scenario_acronym(args)}.output_trips.csv.gz')

    scenarios = split_scenario(args, matsim_scenario=matsim_network, event_reader=event_reader, trip_reader=trip_reader)

    for scenario_idx, (scenario, seed) in enumerate(zip(scenarios, args.cutout_seeds)):
        if seed < 10:
            args.mode = "Train"
        elif seed < 15:
            args.mode = "Validate"
        elif seed < 20:
            args.mode = "Test"

        dir_instances = get_directory_instances(args)
        create_directory(f"{dir_instances}/")
        save_json(scenario, f"{dir_instances}/s-{seed}.json")