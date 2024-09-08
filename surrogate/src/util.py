import json
import time
import sys
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from operator import attrgetter
import multiprocessing as mp
import pandas as pd
import datetime
import tikzplotlib


class Link:
    def __init__(self, from_node, to_node, link_id, weight):
        self.link_id = link_id
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    def to_dict(self):
        return {
            'link_id': self.link_id,
            'from_node': self.from_node,
            'to_node': self.to_node,
            'weight': self.weight
        }


class Node:
    def __init__(self, x, y, node_id):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)


class Graph:
    def __init__(self, instance=None, weights=None):
        self.nodes = {}
        self.links = {}
        self.links_pandas = None
        if instance:
            self.set_nodes_from_instance(instance)
            self.set_links_from_instance(instance, weights)

    def set_nodes_from_instance(self, instance):
        self.nodes = {}
        self.add_nodes_from_instance(instance)

    def add_nodes_from_instance(self, instance):
        for x, y, node_id in zip(instance["nodes_x"], instance["nodes_y"], instance["nodes_id"]):
            self.add_node(x, y, node_id)

    def set_links_from_instance(self, instance, weights=None):
        self.links = {}
        self.add_links_from_instance(instance, weights)

    def add_links_from_instance(self, instance, weights=None):
        if weights is None:
            weights = [0] * len(instance["links_id"])  #instance["link_counts"]
        if isinstance(weights, dict):
            weights = list(weights.values())
        for from_node, to_node, link_id, weight in zip(instance["link_from"], instance["link_to"], instance["links_id"], weights):
            self.add_link(from_node, to_node, link_id, weight)
        self.reset_fast_calculation_files()

    def reset_fast_calculation_files(self):
        self.links_pandas = pd.DataFrame.from_records([link.to_dict() for link in self.links.values()])
        self.links_from_nodes = np.array(self.links_pandas["from_node"])
        self.links_to_nodes = np.array(self.links_pandas["to_node"])
        self.links_weights = np.array(self.links_pandas["weight"])

    def add_node(self, x, y, node_id):
        self.nodes[node_id] = Node(x, y, node_id)

    def add_link(self, from_node, to_node, link_id, weight):
        self.links[link_id] = Link(from_node, to_node, link_id, weight)

    def get_nx(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(list(self.nodes.keys()))
        G.add_weighted_edges_from([(link.from_node, link.to_node, link.weight) for _, link in self.links.items()], weight="theta")
        return G

    def draw(self, x=None, y=None, ax=None, weights_label="", cmap=plt.cm.hot_r, vmin=None, vmax=None, colormap=False, fig=None, name="", args=None):
        g_nx = self.get_nx()
        nodePos = {node.node_id: [node.x, node.y] for _, node in self.nodes.items()}
        #cmap = plt.cm.viridis #plt.cm.Greys # plt.cm.viridis
        if vmin is None or vmax is None:
            edge_vmin = min([link.weight for _, link in self.links.items()])
            edge_vmax = max([link.weight for _, link in self.links.items()])
        else:
            edge_vmin = vmin
            edge_vmax = vmax
        print(f"VMIN: {edge_vmin} | VMAX: {edge_vmax}")
        nx.draw_networkx_edges(g_nx, nodePos, edgelist=[(link.from_node, link.to_node) for _, link in self.links.items()], node_size=30,
                               edge_color=[link.weight for _, link in self.links.items()], edge_cmap=cmap,
                               edge_vmin=edge_vmin, edge_vmax=edge_vmax, width=2, ax=ax, arrows=False)  #, arrows=False, ax=ax
        plt.xlim((min(np.array(list(nodePos.values()))[:, 0]), max(np.array(list(nodePos.values()))[:, 0])))
        plt.ylim((min(np.array(list(nodePos.values()))[:, 1]), max(np.array(list(nodePos.values()))[:, 1])))
        # TODO They changed the edge order
        # TODO alternatively we could use: https://github.com/matsim-vsp/matsim-python-tools/blob/master/matsim/Network.py
        if weights_label is not None and colormap:
            axes = fig.add_axes([1.05, 0.15, 0.05, 0.7])
            fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax), cmap=cmap), cax=axes) # label=weights_label
        if x and y:
            plt.scatter(x[0], y[0], c="red")
            plt.scatter(x[1], y[1], c="blue")


    def get_link_a(self, origin, destination):
        relevant_from_node = (self.links_from_nodes == origin)
        relevant_to_node = (self.links_to_nodes == destination)
        weights_inf = np.full(relevant_from_node.shape, np.inf)
        relevant = relevant_from_node & relevant_to_node
        weights_inf[relevant] = self.links_weights[relevant]
        return np.argmin(weights_inf)

    def get_link_b(self, origin, destination):
        relevant_links = []
        for link_id, link in self.links.items():   # Caution: it is a digraph: we need to return the link with the smallest weight
            if link.from_node == origin and link.to_node == destination:
                relevant_links.append(link)
        if len(relevant_links) > 0:
            return min(relevant_links, key=attrgetter('weight')).link_id
        else:
            return False

    def calculate_shortest_path(self, origin, destination, networkx_graph=None):
        if networkx_graph is None:
            networkx_graph = self.get_nx()
        solution_nodes = nx.shortest_path(networkx_graph, source=origin, target=destination, weight="theta")
        solution_path_a = [self.get_link_a(solution_nodes[i], solution_nodes[i+1]) for i, node in enumerate(solution_nodes[:-1])]
        return solution_path_a


def test_run_on_usefulness(args, write_table=False):
    if (args.co_optimizer == "addedshortestpaths") and (args.learning == "structured") and (args.time_variant_theta or args.capacity_individual_theta):
        if not write_table:
            print("We can not run ADDEDSHORTESTPATH with TIME VARIANT THETA / CAPACITY INDIVIDUAL THETA.")
        return False
    if (args.time_variant_theta + args.trip_individual_theta + args.capacity_individual_theta) > 1:
        if not (args.time_variant_theta == 1 and args.capacity_individual_theta == 1 and args.trip_individual_theta == 0):
            if not write_table:
                print("We can not run consider two from TIME VARIANT THETA / TRIP INDIVIDUAL THETA / CAPACITY INDIVIDUAL THETA simultaneously.")
            return False
    if args.model == "Base":
        args.read_in_iteration = -1
        if args.learning in ["structured"] and args.mode == "Train":
            if not write_table:
                print("We can not learn BASE model with STRUCTURED LEARNING.")
            return False
    if args.model == "RandomForest":
        args.read_in_iteration = -1
        if args.learning == "structured":
            if not write_table:
                print("We can not learn RandomForest with structured learning")
            return False
    if args.learning == "supervised" and args.capacity_individual_theta == 1:
        if not write_table:
            print("Supervised can not consider capacity expansion")
        return False
    if args.mode in ["Test", "Validate"] and args.learning == "supervised" and args.evaluation_procedure == "perturbed":
        if not write_table:
            print("We can not apply perturbed evaluation procedure to supervised learning")
        return False
    if (args.model not in ["Linear"]) and (args.learning in ["structured_bfgs"]):
        if not write_table:
            print("Structured learning BFGS is only working with Linear ML-layer")
        return False
    return True


def update_args(args):
    print(f"INFO:: Number of CPUs in system before setting CPU_COUNT: {mp.cpu_count()}")
    #os.environ["CPU_COUNT"] = str(len(os.sched_getaffinity(0)))
    os.environ["CPU_COUNT"] = str(args.num_cpus)
    print(f"INFO:: Number of CPUs in system: {mp.cpu_count()}")
    print(f"INFO:: Number of CPUs available to process: {len(os.sched_getaffinity(0))}")
    print(f"INFO:: Number of communicated CPUs: {args.num_cpus}")

    useful = test_run_on_usefulness(args)
    if not useful:
        sys.exit()

    if args.time_variant_theta == 1:
        if args.evaluation_metric != "timedist":
            print("We set the evaluation metric to 'timedist' as time_variant_theta is 1.")
            args.evaluation_metric = "timedist"

    # Adapt capacity_multiplicator
    if args.capacity_multiplicator == "-1" or args.capacity_multiplicator == "inf":
        args.capacity_multiplicator = np.inf
    elif args.capacity_multiplicator.isdigit():
        args.capacity_multiplicator = int(args.capacity_multiplicator)
    elif "supervised" in args.capacity_multiplicator:
        args.capacity_multiplicator = ("supervised", int(args.capacity_multiplicator.split("supervised")[1]))


    # When we consider regularized wardrop version we do not consider any perturbation
    if args.co_optimizer == "wardropequilibriaRegularized":
        args.num_perturbations = 0
    if args.co_optimizer == "wardropequilibria" and args.time_variant_theta == 1:
        args.num_perturbations = 0

    # set time limit
    args.max_time = datetime.timedelta(hours=args.max_time.hour, minutes=args.max_time.minute, seconds=args.max_time.second)
    args.max_training_time = args.max_time #- datetime.timedelta(hours=1)
    if args.max_training_time.total_seconds() < 0:
        raise Exception("There is too few training time assigned to training.")

    # set perturbation for evaluation
    if args.mode in ["Test", "Validate"]:
        args.sd_perturbation = args.sd_perturbation_evaluation

    print(f"INFO:: co_optimizer: {args.co_optimizer}")
    print(f"INFO:: time_variant_theta: {args.time_variant_theta}")
    print(f"INFO:: trip_individual_theta: {args.trip_individual_theta}")
    print(f"INFO:: capacity_individual_theta: {args.capacity_individual_theta}")
    print(f"INFO:: capacity_multiplicator: {args.capacity_multiplicator}")
    print(f"INFO:: num_bundled_capacities: {args.num_bundled_capacities}")
    print(f"INFO:: learning: {args.learning}")
    print(f"INFO:: model: {args.model}")
    return args


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_scenario_directory_acronym(args):
    if args.simulation in ["cutoutWorlds", "cutoutWorlds_1", "cutoutWorlds_2", "cutoutWorlds_3", "cutoutWorlds_4", "cutoutWorlds_5",
                           "cutoutWorlds_6", "cutoutWorlds_7", "cutoutWorlds_8", "cutoutWorlds_9", "cutoutWorlds_mixed", "cutoutWorldsCap", "cutoutWorldsSpeed",
                           "cutoutWorldsWithoutBoundary", "districtWorlds", "districtWorldsArt"]:
        return "po-" + str(args.percentage_original) + "_pn-" + str(args.percentage_new) + "_sn-" + str(args.seed_new)
    elif args.simulation == "sparseWorlds":
        return "po-" + str(args.percentage_original)
    elif args.simulation in ["smallWorlds", "smallWorlds_pricing", "smallWorlds_short", "squareWorlds_short",
                             "squareWorlds_short_capacitated", "againstsupervisedWorld", "squareWorlds_short_againstsupervisedSimulation",
                             "low_entropy", "high_entropy", "low_entropy_street", "high_entropy_street"]:
        return "s"
    else:
        raise Exception("Unknown simulation")


def get_scenario_acronym(args):
    if args.simulation in ["cutoutWorlds", "cutoutWorlds_1", "cutoutWorlds_2", "cutoutWorlds_3", "cutoutWorlds_4", "cutoutWorlds_5",
                           "cutoutWorlds_6", "cutoutWorlds_7", "cutoutWorlds_8", "cutoutWorlds_9", "cutoutWorlds_mixed", "cutoutWorldsCap", "cutoutWorldsSpeed",
                           "cutoutWorldsWithoutBoundary"]:
        return get_scenario_directory_acronym(args)
    elif args.simulation in ["sparseWorlds", "districtWorlds", "districtWorldsArt"]:
        return get_scenario_directory_acronym(args) + "_s-" + str(args.seed)
    elif args.simulation in ["smallWorlds", "smallWorlds_pricing", "smallWorlds_short", "squareWorlds_short",
                             "squareWorlds_short_capacitated", "againstsupervisedWorld", "squareWorlds_short_againstsupervisedSimulation",
                             "low_entropy", "high_entropy", "low_entropy_street", "high_entropy_street"]:
        return get_scenario_directory_acronym(args) + "-" + str(args.seed)
    else:
        raise Exception("Unknown simulation")


# CONFIGS ####################################################
def get_model_name(args):
    return f"m-{args.model}"


def get_learning_name(args):
    return f"l-{args.learning}"


def get_iteration(args):
    return f"it-{args.read_in_iteration}"


def get_mode(args):
    return f"mo-{args.mode}"


def get_simulation(args):
    if args.simulator == "matsim":
        return f"si-{args.simulation}"
    else:
        return f"si-{args.simulation}_{args.simulator}"


def get_perturbation(args):
    return ((f"pert-{args.perturbation}" if ((args.learning in ["structured"]) and (args.model not in ["Base"])) else "pert-None") +
            (f"_nump-{args.num_perturbations}" if ((args.learning in ["structured"]) and (args.model not in ["Base"]) and args.num_perturbations != 50) else ""))


def get_standardization(args):
    return f"st-{args.standardize}" if (args.mode not in ["Base"]) else "st-None"


def get_co_optimizer(args):
    co_optimizer_specification = f"co-{args.co_optimizer}"
    if args.co_optimizer == "multicommodityflow":
        co_optimizer_specification += f"_{get_num_bundled_capacities(args)}_{get_capacity_multiplicator(args)}"
    return co_optimizer_specification


def get_capacity_multiplicator(args):
    return f"cm-{args.capacity_multiplicator}"


def get_num_discrete_time_steps(args):
    if args.co_optimizer in ["addedshortestpaths"] and args.learning in ["structured"]:
        return "dts-None"
    else:
        return f"dts-{args.num_discrete_time_steps}"


def get_time_variant_theta(args):
    return f"tvt-{args.time_variant_theta}"


def get_trip_individual_theta(args):
    return f"tit-{args.trip_individual_theta}"


def get_capacity_individual_theta(args):
    return f"cit-{args.capacity_individual_theta}"


def get_experimental_design(args):
    return f"ed-{args.experimental_design}"


def get_problem(args):
    return f"p-{args.problem}"


def get_evaluation_metric(args):
    if args.evaluation_metric == "original":
        return ""
    else:
        return f"_em-{args.evaluation_metric}"


def get_surrogate(args):
    return f"su-{args.surrogate}"


def get_strategy(args):
    return f"str-{args.strategy}"


def get_feature(args):
    if (args.feature == "") or (args.feature == "''"):
        return ""
    else:
        return f"_fe-{args.feature}"


def get_evaluation_procedure(args):
    if args.evaluation_procedure == "original":
        return f"evp-{args.evaluation_procedure}"
    elif args.evaluation_procedure == "perturbed":
        return f"evp-{args.evaluation_procedure}_sde-{args.sd_perturbation_evaluation}"
    else:
        raise Exception("Wrong evaluation procedure defined.")


def get_num_bundled_capacities(args):
    return f"nbc-{args.num_bundled_capacities}"


# DIRECTORIES ####################################################
def get_directory_instances(args):
    if args.simulator == "matsim":
        return f"{args.dir_instances}/{args.simulation}/{args.mode}/{get_scenario_directory_acronym(args)}"
    else:
        return f"{args.dir_instances}/{args.simulation}_{args.simulator}/{args.mode}/{get_scenario_directory_acronym(args)}"



# FILE NAMES ####################################################
def get_result_file(args):
    return args.dir_results + get_result_file_name(args)


def get_model_file(args):
    return args.dir_models + get_model_file_name(args)


def get_learning_file(args):
    return args.dir_learning + get_model_file_name(args)


def get_standardization_file(args):
    return args.dir_standardization + get_standardization_file_name(args)


def get_result_file_name(args):
    return f"{get_result_name(args)}_{get_iteration(args)}"


def get_result_name(args):
    return f"result{args.experiment_name}_{get_model_file_name_specification(args)}_{get_evaluation_procedure(args)}{get_evaluation_metric(args)}_{get_mode(args)}"


def get_standardization_file_name(args):
    return f"standardization_{get_model_file_name_specification(args)}"


def get_ml_layer(args):
    return f"{get_model_name(args)}_{get_perturbation(args)}_{get_standardization(args)}{get_feature(args)}_{get_learning_name(args)}"


def get_co_layer(args):
    co_layer_specification = ""
    if args.learning in ["structured", "structured_wardrop"]:
        co_layer_specification += f"{get_co_optimizer(args)}_"
    co_layer_specification += f"{get_num_discrete_time_steps(args)}_{get_time_variant_theta(args)}_{get_trip_individual_theta(args)}_{get_capacity_individual_theta(args)}"
    return co_layer_specification


def get_pipeline_name(args):
    return f"{get_ml_layer(args)}_{get_co_layer(args)}"


def get_model_file_name(args):
    if args.mode in ["Train", "Test", "Validate"]:
        return f"model_{args.experiment_name}_{get_model_file_name_specification(args)}"
    elif args.mode == "Bayesian":
        return f"bayesian_{get_bayesian_specification(args)}"


def get_model_file_name_specification(args):
    return f"{get_pipeline_name(args)}_{get_simulation(args)}_{get_scenario_directory_acronym(args)}"


def get_bayesian_specification(args):
    return (f"{get_pipeline_name(args)}_{get_simulation(args)}_{get_scenario_acronym(args)}_"
            f"{get_experimental_design(args)}_{get_problem(args)}_{get_surrogate(args)}_{get_strategy(args)}")


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(json_object, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_object, f, cls=NpEncoder)


def remove_directory(directory):
    if os.path.exists(directory):
        os.remove(directory)
        print("Dir {} ... deleted".format(directory))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("New dir {} ... created".format(directory))


def create_directories(list_directories):
    for directory in list_directories:
        create_directory(directory)


def convert_x_str(x):
    return str([float(value) for value in x])


def get_iteration_file(args):
    return f"{args.dir_best_iteration_saver}{get_model_file_name_specification(args)}_{get_evaluation_procedure(args)}{get_evaluation_metric(args)}.txt"


def get_read_in_iteration(args, prefix=""):
    if args.model == "Base" and args.learning == "structured":
        return -1
    if args.model == "RandomForest":
        return -1
    else:
        f = open(f"{prefix}{get_iteration_file(args)}", "r")
        read_in_iteration = int(f.read())
        return read_in_iteration
