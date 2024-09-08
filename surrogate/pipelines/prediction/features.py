import os
import numpy as np
import pandas as pd
from surrogate.src import util
from collections import Counter, defaultdict


def load_training_instances(args, prefix=""):
    training_instances_collecter = []
    for training_instance in os.listdir(f"{prefix}{util.get_directory_instances(args)}/"):
        training_instance_data = load_training_instance(args, training_instance, prefix)
        training_instance_data["training_instance_name"] = training_instance
        training_instances_collecter.append(training_instance_data)
    return training_instances_collecter


def load_training_instance(args, training_instance, prefix=""):
    return util.load_json(f"{prefix}{util.get_directory_instances(args)}/" + training_instance)


def get_num_streets_ingoingoutgoing(data):
    count_from = {i: 0 for i in range(len(data["nodes_id"]))}
    for key, value in Counter(data["link_from"]).items():
        count_from[key] = value

    count_to = {i: 0 for i in range(len(data["nodes_id"]))}
    for key, value in Counter(data["link_to"]).items():
        count_to[key] = value

    node_counter = pd.DataFrame({"nodes_id": data["nodes_id"], "count_from": list(count_from.values()), "count_to": list(count_to.values())})
    links = pd.DataFrame({"link_id": data["links_id"], "link_from": data["link_from"], "link_to": data["link_to"]})
    links = links.merge(node_counter.add_suffix("_from"), how="left", left_on="link_from", right_on="nodes_id_from")
    links = links.merge(node_counter.add_suffix("_to"), how="left", left_on="link_to", right_on="nodes_id_to")
    return links["count_to_from"].values, links["count_from_to"].values, links["count_from_from"].values, links["count_to_to"].values


def get_num_items_around_location(args, data, location, item, divisor, attribute=None, threshold=None):
    location_x = np.repeat(np.array(data[location + "_x"])[:, np.newaxis], len(data[item + "_x"]), axis=1)
    location_y = np.repeat(np.array(data[location + "_y"])[:, np.newaxis], len(data[item + "_y"]), axis=1)
    item_x = np.repeat(np.array(data[item + "_x"])[:, np.newaxis], len(data[location + "_x"]), axis=1).T
    item_y = np.repeat(np.array(data[item + "_y"])[:, np.newaxis], len(data[location + "_y"]), axis=1).T
    distances = np.sqrt((item_x - location_x)**2 + (item_y - location_y)**2)
    max_distances = np.max(distances) / divisor
    if attribute and threshold:
        attribute_rep = np.repeat(np.array(data[attribute])[:, np.newaxis], len(data[location + "_x"]), axis=1).T
        return np.sum((distances < max_distances) & (attribute_rep < threshold), axis=1)
    else:
        return np.sum(distances < max_distances, axis=1)


def get_link_coordinates(data):
    data_nodes_x = np.array(data["nodes_x"])
    data_nodes_y = np.array(data["nodes_y"])
    links_x = (data_nodes_x[data["link_to"]] + data_nodes_x[data["link_from"]]) / 2
    links_y = (data_nodes_y[data["link_to"]] + data_nodes_y[data["link_from"]]) / 2
    return links_x, links_y


