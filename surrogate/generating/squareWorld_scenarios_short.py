import os
import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
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
    nodes, edges = generate_square_network(100, 50, seed=args.seed) #generate_square_network(20, 200, seed=args.seed) #generate_square_network(3, 100, seed=args.seed) # #
    nodes_converter = {node["id"]: new_id for new_id, node in enumerate(nodes)}

    for edge in edges:
        if {"id_i": edge["id_j"], "id_j": edge["id_i"]} not in edges:
            edges.append({"id_i": edge["id_j"], "id_j": edge["id_i"]})


    graph = {}
    graph["nodes_x"] = [node["x"] for node in nodes]
    graph["nodes_y"] = [node["y"] for node in nodes]
    graph["nodes_id"] = [nodes_converter[node["id"]] for node in nodes]
    graph["link_from"] = [nodes_converter[edge["id_i"]] for edge in edges]
    graph["link_to"] = [nodes_converter[edge["id_j"]] for edge in edges]
    graph["links_id"] = np.arange(len(graph["link_to"]))
    weights = [1] * len(graph["link_to"])
    g = Graph(graph, weights)
    g.draw()
    return graph

def generate_random_population(args, G, minimum_start):

    def get_distribution(G):
        hist, x_bins, y_bins = np.histogram2d(G["nodes_x"], G["nodes_y"], bins=55)
        hist = convolve(hist, Gaussian2DKernel(x_stddev=2))
        hist = np.rot90(hist)  # we have to rotate to be consistent with x and y-axis
        hist = hist / hist.sum()  # the probability has to sum to 1
        hist_copy = copy.deepcopy(hist)
        hist_copy[hist_copy < 0.0001] = "nan"
        plt.imshow(hist_copy, extent=(min(G["nodes_x"]),
                                 max(G["nodes_x"]),
                                 min(G["nodes_y"]),
                                 max(G["nodes_y"])), alpha=0.5)
        mesh = np.meshgrid(x_bins[:-1], np.flip(y_bins[:-1]))  # we have to flip the y-axis so that the y-axis increases its values correctly
        return hist, mesh

    def sample_locations(distribution, sample_size):
        flat = distribution.flatten()
        sample_index = np.random.choice(a=flat.size, p=flat, size=sample_size)
        return np.unravel_index(sample_index, distribution.shape)

    def convert_to_coords(distribution, mesh):
        (mesh_x, mesh_y) = mesh
        return mesh_x[distribution[0], distribution[1]], mesh_y[distribution[0], distribution[1]]

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

    pop_distribution, pop_mesh = get_distribution(G)
    work_distribution = sample_locations(distribution=pop_distribution, sample_size=args.population_size)
    home_distribution = sample_locations(distribution=pop_distribution, sample_size=args.population_size)
    work_distribution = convert_to_coords(work_distribution, pop_mesh)
    home_distribution = convert_to_coords(home_distribution, pop_mesh)
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
    args.simulation = "squareWorlds_short"
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
