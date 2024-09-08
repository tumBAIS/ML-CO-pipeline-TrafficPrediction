import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matsim
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-2])
sys.path.append(main_directory)
from surrogate.src.util import save_json, get_scenario_acronym, create_directory
from surrogate.src import config
from surrogate.src.util import Graph
from surrogate.generating.square_network_master.rng.network_gen import main as generate_square_network


class Population:
    def __init__(self, home_distribution, work_distribution, go_to_work, go_to_home, o_d_pairs, o_d_person_ids):
        self.home_distribution = home_distribution
        self.work_distribution = work_distribution
        self.go_to_work = go_to_work
        self.go_to_home = go_to_home
        self.o_d_pairs = o_d_pairs
        self.o_d_person_ids = o_d_person_ids


def generate_random_graph(args):
    nodes_home, edges_home = generate_square_network(30, 15, seed=args.seed) #generate_square_network(20, 200, seed=args.seed) #generate_square_network(3, 100, seed=args.seed) # #
    nodes_work, edges_work = generate_square_network(30, 15, seed=args.seed + 3) #generate_square_network(20, 200, seed=args.seed) #generate_square_network(3, 100, seed=args.seed) # #

    nodes_home = pd.DataFrame(nodes_home)
    nodes_home["node_area"] = "home"
    nodes_work = pd.DataFrame(nodes_work)
    nodes_work["node_area"] = "work"
    nodes_work["y"] = nodes_work["y"] + 100
    edges_home = pd.DataFrame(edges_home)
    edges_home["node_area"] = "home"
    edges_work = pd.DataFrame(edges_work)
    edges_work["node_area"] = "work"

    nodes = pd.concat([nodes_home, nodes_work])
    edges = pd.concat([edges_home, edges_work])
    nodes["new_id"] = np.arange(len(nodes))

    edges = edges.merge(nodes[["node_area", "id", "new_id"]].add_suffix("_i"), how="left", left_on=["node_area", "id_i"], right_on=["node_area_i", "id_i"])
    edges = edges.merge(nodes[["node_area", "id", "new_id"]].add_suffix("_j"), how="left", left_on=["node_area", "id_j"], right_on=["node_area_j", "id_j"])
    edges = edges.drop(columns=["node_area_i", "node_area_j"])

    node_home_start = sorted(nodes[nodes["node_area"] == "home"].to_dict("records"), key=lambda x: x["y"])[-1]
    node_work_end = sorted(nodes[nodes["node_area"] == "work"].to_dict("records"), key=lambda x: x["y"])[0]

    nodes_connection = []
    edges_connection = []

    start_id = node_home_start["new_id"]
    id_counter = max(nodes["new_id"].values)
    end_id = id_counter + 1
    for counter, (x, y) in enumerate(zip(np.linspace(node_home_start["x"], node_work_end["x"], 10)[1:-1],
                              np.linspace(node_home_start["y"], node_work_end["y"], 10)[1:-1])):
        end_id = id_counter + counter + 1
        nodes_connection.append({"new_id": end_id, "x": x, "y": y})
        edges_connection.append({"new_id_i": start_id, "new_id_j": end_id})
        edges_connection.append({"new_id_i": end_id, "new_id_j": start_id})
        start_id = end_id
    edges_connection.append({"new_id_i": end_id, "new_id_j": node_work_end["new_id"]})
    edges_connection.append({"new_id_i": node_work_end["new_id"], "new_id_j": end_id})

    nodes = pd.concat([nodes, pd.DataFrame(nodes_connection)])
    edges = pd.concat([edges, pd.DataFrame(edges_connection)])

    nodes["id"] = nodes["new_id"]
    nodes = nodes.drop(columns=["new_id"])
    edges["id_i"] = edges["new_id_i"]
    edges["id_j"] = edges["new_id_j"]
    edges = edges.drop(columns=["new_id_i", "new_id_j"])

    nodes = nodes.to_dict("records")
    edges = edges.to_dict("records")

    for edge in edges:
        if {"id_i": edge["id_j"], "id_j": edge["id_i"]} not in edges:
            edges.append({"id_i": edge["id_j"], "id_j": edge["id_i"]})

    graph = {}
    graph["nodes_x"] = [node["x"] for node in nodes]
    graph["nodes_y"] = [node["y"] for node in nodes]
    graph["nodes_id"] = [node["id"] for node in nodes]
    graph["link_from"] = [edge["id_i"] for edge in edges]
    graph["link_to"] = [edge["id_j"] for edge in edges]
    graph["links_id"] = np.arange(len(graph["link_to"]))
    weights = [1] * len(graph["link_to"])
    g = Graph(graph, weights)
    g.draw()
    home_area = {"mean_x": np.mean(nodes_home["x"]), "max_x": np.max(nodes_home["x"]), "min_x": np.min(nodes_home["x"]),
                 "mean_y": np.mean(nodes_home["y"]), "max_y": np.max(nodes_home["y"]), "min_y": np.min(nodes_home["y"])}
    work_area = {"mean_x": np.mean(nodes_work["x"]), "max_x": np.max(nodes_work["x"]), "min_x": np.min(nodes_work["x"]),
                 "mean_y": np.mean(nodes_work["y"]), "max_y": np.max(nodes_work["y"]), "min_y": np.min(nodes_work["y"])}
    return graph, home_area, work_area


def generate_random_population(args, G, minimum_start, home_area, work_area):
    def sample_locations(area, sample_size):
        x = [random.uniform(area["min_x"], area["max_x"]) for _ in range(sample_size)]
        y = [random.uniform(area["min_y"], area["max_y"]) for _ in range(sample_size)]
        return x, y

    def generate_trips(home_distribution, work_distribution):
        def get_nearest_node(x, y, nodes_x, nodes_y):
            return np.argmin(np.sqrt((nodes_x - x) ** 2 + (nodes_y - y) ** 2))

        trips = []
        person_ids = []
        for person_id, (person_home_x, person_home_y, person_work_x, person_work_y) in enumerate(zip(home_distribution[0], home_distribution[1], work_distribution[0], work_distribution[1])):
            location_home = get_nearest_node(person_home_x, person_home_y, G["nodes_x"], G["nodes_y"])
            location_work = get_nearest_node(person_work_x, person_work_y, G["nodes_x"], G["nodes_y"])
            trips += [(location_home, location_work), (location_work, location_home)]
            person_ids += [person_id, person_id]
        return trips, person_ids

    work_distribution = sample_locations(area=work_area, sample_size=args.population_size)
    home_distribution = sample_locations(area=home_area, sample_size=args.population_size)
    plt.scatter(work_distribution[0], work_distribution[1], c="red", label="work")
    plt.scatter(home_distribution[0], home_distribution[1], c="green", label="home")
    plt.legend()
    go_to_work = [(datetime.timedelta(seconds=minimum_start)).total_seconds() for person in range(args.population_size)]
    go_to_home = [(datetime.timedelta(seconds=minimum_start, minutes=30)).total_seconds() for person in range(args.population_size)]
    o_d_pairs, o_d_person_ids = generate_trips(home_distribution, work_distribution)
    return Population(home_distribution, work_distribution, go_to_work, go_to_home, o_d_pairs, o_d_person_ids)



def write_plans(graph, population):
    with open(f"{main_directory}/matsim-berlin/scenarios/{args.simulation}/sd_{args.seed}/plans_{args.seed}.xml", 'wb+') as f_write:
        writer = matsim.writers.PopulationWriter(f_write)
        writer.start_population()

        for person_id, (home, work, go_to_work_time, go_to_home_time) in enumerate(zip(np.array(population.home_distribution).T,
                                                                                       np.array(population.work_distribution).T,
                                                                                       population.go_to_work,
                                                                                       population.go_to_home)):
            writer.start_person(person_id)
            writer.start_attributes()
            writer.add_attribute(name="subpopulation", value="person", typ="java.lang.String")
            writer.end_attributes()
            writer.start_plan(selected=True)
            writer.add_activity(type='h', x=home[0], y=home[1], end_time=go_to_work_time)
            writer.add_leg(mode='car')
            writer.add_activity(type='w', x=work[0], y=work[1], end_time=go_to_home_time)
            writer.add_leg(mode='car')
            writer.add_activity(type='h', x=home[0], y=home[1])
            writer.end_plan()
            writer.end_person()

        writer.end_population()



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
    scenario["link_capacity"] = np.array([100.0] * len(scenario["links_id"]))
    scenario["link_permlanes"] = np.array([1.0] * len(scenario["links_id"]))

    maximum_length = 0.05 * (maximum_end - minimum_start) * freespeed
    distance_converter = maximum_length / max(np.sqrt(
        (np.array(scenario["nodes_x"])[scenario["link_from"]] - np.array(scenario["nodes_x"])[scenario["link_to"]]) ** 2 +
        (np.array(scenario["nodes_y"])[scenario["link_from"]] - np.array(scenario["nodes_y"])[scenario["link_to"]]) ** 2))

    scenario["link_length"] = distance_converter * np.sqrt(    # 4000.0
        (np.array(scenario["nodes_x"])[scenario["link_from"]] - np.array(scenario["nodes_x"])[scenario["link_to"]]) ** 2 +
        (np.array(scenario["nodes_y"])[scenario["link_from"]] - np.array(scenario["nodes_y"])[scenario["link_to"]]) ** 2)


    create_directory(f"{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}")
    save_json(scenario, f'{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/scenario.json')


if __name__ == '__main__':
    args = config.parser.parse_args()
    args.simulation = "againstsupervisedWorld"
    np.random.seed(args.seed)
    random.seed(args.seed)

    minimum_start = 8 * 3600
    maximum_end = 9 * 3600

    fig, ax = plt.subplots(figsize=(10, 10))
    graph, home_area, work_area = generate_random_graph(args=args)
    population = generate_random_population(args=args, G=graph, minimum_start=minimum_start, home_area=home_area, work_area=work_area)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()
    #plt.show()

    save_random_scenario(args=args, graph=graph, population=population, minimum_start=minimum_start, maximum_end=maximum_end)