import copy
import numpy as np
import pandas as pd
import math

def load_capacities(args, training_instances):
    if args.learning == "structured" and args.co_optimizer == "multicommodityflow":
        if isinstance(args.capacity_multiplicator, tuple) and args.capacity_multiplicator[0] == "supervised":
            raise Exception("Not implemented")
        else:
            for training_instance in training_instances:
                solution_representation = training_instance["solution_representation"]
                capacity = pd.DataFrame({"target": 1, "link_id": training_instance["solution_link"]})
                capacity["time"] = [solution_representation.solution_scheme_time[(np.abs(t - solution_representation.solution_scheme_time)).argmin()] for t in training_instance["solution_time"]]
                capacity = capacity.set_index(["time", "link_id"])
                capacity = capacity.groupby(["link_id", "time"]).sum()
                if not args.time_variant_theta:
                    capacity = capacity.groupby(["link_id"]).sum()
                if args.capacity_individual_theta:
                    capacity = solution_representation.add_capacity_groups(solution=capacity, capacity_per_entry=capacity.target.values)
                training_instance["capacity_prediction"] = capacity
    return training_instances


def get_capacity_expanded_graph(args, arcs, max_capacity):

    #GET NEW EDGES
    new_arcs = []
    for original_arc in arcs.to_dict("records"):
        for new_arc_cap_idx in range(math.ceil(max_capacity / args.num_bundled_capacities)):
            new_arc = copy.deepcopy(original_arc)
            new_arc["capacity"] = args.num_bundled_capacities
            new_arc["index_capacity_group"] = new_arc_cap_idx
            new_arcs.append(new_arc)
    new_arcs = pd.DataFrame(new_arcs)
    new_arcs["arc_idx"] = np.arange(len(new_arcs))
    return new_arcs


def set_capacity(args, arcs, instance):
    if isinstance(args.capacity_multiplicator, tuple) and args.capacity_multiplicator[0] == "supervised":
        capacity_multiplicator = args.capacity_multiplicator[1]
    else:
        capacity_multiplicator = args.capacity_multiplicator

    capacity_prediction = instance["capacity_prediction"]
    arcs = arcs.merge(capacity_prediction, how="left", on=capacity_prediction.index.names)
    arcs["target"] = arcs["target"].fillna(0)
    new_capacity = arcs["target"].values.astype(float)
    new_capacity[arcs["capacity"] == np.inf] = np.inf
    if capacity_multiplicator > 0:
        new_capacity[new_capacity == 0] = 1
    new_capacity[new_capacity != np.inf] = np.ceil(((capacity_multiplicator / 100) * new_capacity[new_capacity != np.inf]) + new_capacity[new_capacity != np.inf])   # Increase in percentage
    # We set the new capacity
    arcs["capacity"] = new_capacity
    arcs = arcs.drop(columns=["target"])
    return arcs
