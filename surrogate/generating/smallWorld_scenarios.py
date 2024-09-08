import os
import copy
import datetime
import networkx as nx
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


class Graph:
    def __init__(self, networkxG, nodePos):
        self.networkxG = networkxG
        self.nodePos = nodePos

    def get_nodes_x(self):
        return np.array(list(self.nodePos.values()))[:, 0]

    def get_nodes_y(self):
        return np.array(list(self.nodePos.values()))[:, 1]

    def get_graph_in_json(self):
        return {"nodes_x": self.get_nodes_x(),
                   "nodes_y": self.get_nodes_y(),
                   "nodes_id": [str(idx) for idx in self.networkxG.nodes],
                   "link_from": np.concatenate((np.array(self.networkxG.edges)[:, 0], np.array(self.networkxG.edges)[:, 1])),
                   "link_to": np.concatenate((np.array(self.networkxG.edges)[:, 1], np.array(self.networkxG.edges)[:, 0])),
                   "links_id": [str(idx) for idx, edge in enumerate(np.concatenate((np.array(self.networkxG.edges), np.array(self.networkxG.edges))))]}

class Population:
    def __init__(self, home_distribution, work_distribution, go_to_work, go_to_home, o_d_pairs):
        self.home_distribution = home_distribution
        self.work_distribution = work_distribution
        self.go_to_work = go_to_work
        self.go_to_home = go_to_home
        self.o_d_pairs = o_d_pairs

def generate_random_graph():
    G = nx.newman_watts_strogatz_graph(n=args.num_nodes, k=3, p=0.5, seed=args.seed)
    nodePos = nx.spring_layout(G, seed=1)
    nx.draw_networkx(G, pos=nodePos, node_size=30, with_labels=False, ax=ax)
    return Graph(G, nodePos)

def generate_random_population(G):

    def get_distribution(G):
        hist, x_bins, y_bins = np.histogram2d(np.array(list(G.nodePos.values()))[:, 0], np.array(list(G.nodePos.values()))[:, 1], bins=55)
        hist = convolve(hist, Gaussian2DKernel(x_stddev=2))
        hist = np.rot90(hist)  # we have to rotate to be consistent with x and y-axis
        hist = hist / hist.sum()  # the probability has to sum to 1
        hist_copy = copy.deepcopy(hist)
        hist_copy[hist_copy < 0.0001] = "nan"
        plt.imshow(hist_copy, extent=(min(np.array(list(G.nodePos.values()))[:, 0]),
                                 max(np.array(list(G.nodePos.values()))[:, 0]),
                                 min(np.array(list(G.nodePos.values()))[:, 1]),
                                 max(np.array(list(G.nodePos.values()))[:, 1])), alpha=0.5)
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
        for person_home_x, person_home_y, person_work_x, person_work_y in zip(home_distribution[0], home_distribution[1], work_distribution[0], work_distribution[1]):
            location_home = get_nearest_node(person_home_x, person_home_y, np.array(G.get_nodes_x()), np.array(G.get_nodes_y()))
            location_work = get_nearest_node(person_work_x, person_work_y, np.array(G.get_nodes_x()), np.array(G.get_nodes_y()))
            trips += [(location_home, location_work), (location_work, location_home)]
        return trips

    pop_distribution, pop_mesh = get_distribution(G)
    work_distribution = sample_locations(distribution=pop_distribution, sample_size=args.population_size)
    home_distribution = sample_locations(distribution=pop_distribution, sample_size=args.population_size)
    work_distribution = convert_to_coords(work_distribution, pop_mesh)
    home_distribution = convert_to_coords(home_distribution, pop_mesh)
    plt.scatter(work_distribution[0], work_distribution[1], c="red")
    plt.scatter(home_distribution[0], home_distribution[1], c="green")
    go_to_work = [(datetime.timedelta(hours=5) + (datetime.timedelta(hours=5) * random.random())).total_seconds() for person in range(args.population_size)]
    go_to_home = [(datetime.timedelta(hours=15) + (datetime.timedelta(hours=7) * random.random())).total_seconds() for person in range(args.population_size)]
    o_d_pairs = generate_trips(home_distribution, work_distribution)
    return Population(home_distribution, work_distribution, go_to_work, go_to_home, o_d_pairs)



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



def save_random_scenario(graph, population):

    scenario = {"nodes_x": np.array(list(graph.nodePos.values()))[:, 0],
                   "nodes_y": np.array(list(graph.nodePos.values()))[:, 1],
                   "nodes_id": [int(idx) for idx in graph.networkxG.nodes],  # TODO is this correct when we apply the spring_layout?
                   "link_from": np.concatenate((np.array(graph.networkxG.edges)[:, 0], np.array(graph.networkxG.edges)[:, 1])),
                   "link_to": np.concatenate((np.array(graph.networkxG.edges)[:, 1], np.array(graph.networkxG.edges)[:, 0])),
                   "links_id": [int(idx) for idx, edge in enumerate(np.concatenate((np.array(graph.networkxG.edges), np.array(graph.networkxG.edges))))],
                   "work_x": population.work_distribution[0],
                   "work_y": population.work_distribution[1],
                   "home_x": population.home_distribution[0],
                   "home_y": population.home_distribution[1],
                   "go_to_work": population.go_to_work,
                   "go_to_home": population.go_to_home,
                   "o_d_pairs": population.o_d_pairs}

    scenario["link_length"] = 5000.0 * np.sqrt(
        (np.array(scenario["nodes_x"])[scenario["link_from"]] - np.array(scenario["nodes_x"])[scenario["link_to"]]) ** 2 +
        (np.array(scenario["nodes_y"])[scenario["link_from"]] - np.array(scenario["nodes_y"])[scenario["link_to"]]) ** 2)
    scenario["link_freespeed"] = np.array([13.88] * len(scenario["links_id"]))
    scenario["link_capacity"] = np.array([100.0] * len(scenario["links_id"]))
    scenario["link_permlanes"] = np.array([1.0] * len(scenario["links_id"]))

    create_directory(f"{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}")
    save_json(scenario, f'{main_directory}/matsim-berlin/scenarios/{args.simulation}/{get_scenario_acronym(args)}/scenario.json')



if __name__ == '__main__':
    args = config.parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    fig, ax = plt.subplots(figsize=(10, 10))
    graph = generate_random_graph()
    population = generate_random_population(graph)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()
    #plt.show()

    save_random_scenario(graph=graph, population=population)

