import numpy as np
import pandas as pd
import gurobipy as gp
import math


class Solution:
    def __init__(self, args):
        self.args = args
        self.solution_scheme = None
        self.solution_scheme_nodes = None
        self.solution_scheme_metrics = []
        self.solution_scheme_metrics_nodes = []
        self.time_steps = None
        self.solution_scheme_time = None
        self.solution_scheme_commodities = None
        self.solution_scheme_linksId = None
        self.solution_scheme_nodesId = None
        self.maximum_capacity = None
        self.num_max_capacity_groups = None
        self.solution_scheme_capacity_group = None

    def initialize_solution_scheme(self, instance):
        assert isinstance(instance, dict)
        time_step_length = math.ceil((instance["maximum_end"]-instance["minimum_start"]) / self.args.num_discrete_time_steps)
        puffer_time = 0
        self.time_steps = np.arange(start=0, stop=instance["maximum_end"]-instance["minimum_start"] + puffer_time, step=time_step_length)
        self.solution_scheme_time = np.arange(start=instance["minimum_start"], stop=instance["maximum_end"] + puffer_time, step=time_step_length)
        self.solution_scheme_commodities = instance["commodities"]
        self.solution_scheme_linksId = instance["links_id"]
        self.solution_scheme_nodesId = instance["nodes_id"]

    def load_y(self, sol, multiindex=None, y_column_name="target"):
        if isinstance(sol, dict):
            y = self.load_y_from_instance(instance=sol)
        elif isinstance(sol, gp.tupledict):
            y = self.load_y_from_tupledict(instance=sol, multiindex=multiindex)
        elif isinstance(sol, np.ndarray):
            y = self.load_y_from_array(instance=sol)
        elif isinstance(sol, pd.DataFrame):
            y = self.load_y_from_pandas(instance=sol)
        elif isinstance(sol, list):
            y = self.load_y_from_solutionpaths(solution_paths=sol)
        else:
            raise Exception("Unrecognized solution type.")

        y = y.set_index(self.solution_scheme_metrics)
        y = self.shrink_dimensions(y, y_column_name)

        if self.args.capacity_individual_theta and ("index_capacity_group" not in y.index.names):
            y = self.add_capacity_groups(solution=y, capacity_per_entry=y.target)

        y = self.solution_scheme.merge(y, how="left", on=self.solution_scheme.index.names)[list(y.columns)]
        return y[y_column_name].fillna(0)

    def load_y_from_tupledict(self, instance, multiindex):
        if multiindex is None:
            raise Exception("Multiindex is missing to load solution from gp.tupledict.")
        y = pd.DataFrame(instance)
        return y

    def load_y_from_instance(self, instance):
        y = pd.DataFrame()
        y["commodity"] = instance["solution_commodity"]
        y["time"] = [self.solution_scheme_time[(np.abs(t - self.solution_scheme_time)).argmin()] for t in instance["solution_time"]]  # discretize starting times to T
        y["link_id"] = instance["solution_link"]
        if "solution_value" in instance.keys():
            y["target"] = instance["solution_value"]
        else:
            y["target"] = [1] * len(instance["solution_link"])

        if self.args.time_variant_theta and (self.args.feature not in ["withoutedges", "withoutedgeslogic", "withoutedgessquare"]):
            for commodity, od_pair, od_pair_ending_time in zip(instance["commodities"], instance["o_d_pairs"], instance["o_d_pairs_ending_time"]):
                time_end = self.solution_scheme_time[(np.abs(od_pair_ending_time - self.solution_scheme_time)).argmin()]
                remaining_times = self.solution_scheme_time[list(self.solution_scheme_time).index(time_end):]
                standby_solution = pd.DataFrame({"time": remaining_times, "commodity": commodity, "link_id": -od_pair[1]-1, "target": 1})
                y = pd.concat([y, standby_solution], ignore_index=True)
        return y

    def load_y_from_array(self, instance):
        y = pd.DataFrame(index=self.solution_scheme.index)
        y = y.reset_index()
        if len(instance.shape) == 2:
            y["target"] = list(map(tuple, instance.T))
        else:
            y["target"] = instance
        return y

    def load_y_from_solutionpaths(self, solution_paths):
        solution = []
        for commodity, sol in enumerate(solution_paths):
            for link in sol:
                solution.append({"commodity": commodity, "time": 0, "link_id": link, "target": 1})
        y = pd.DataFrame(solution)
        return y

    def load_y_from_pandas(self, instance):
        return instance

    def shrink_dimensions(self, solution, y_column_name):
        solution = solution.groupby(self.solution_scheme_metrics)[y_column_name].sum()
        return pd.DataFrame(solution)

    def load_solution_scheme(self, instance, y_column_name="target"):
        def adding_trip_dimension(solution_scheme):
            updated_solution_scheme = solution_scheme.merge(pd.DataFrame({"commodity": self.solution_scheme_commodities}), how="cross")
            updated_solution_scheme["new_link_from"] = updated_solution_scheme["link_from"]
            updated_solution_scheme["new_link_to"] = updated_solution_scheme["link_to"]
            return updated_solution_scheme

        def adding_time_dimension(solution_scheme, solution_scheme_nodes):
            # Discretize times to traverse edge
            edge_flow_times = np.array(instance["link_length"]) / np.array(instance["link_freespeed"])
            edge_flow_times_rounded = [(np.abs(t - instance["solution_representation"].time_steps)).argmin() for t in edge_flow_times]  # discretize edge_flow_times to T
            edge_flow_times_rounded = [flow_time if flow_time > 0 else 1 for flow_time in edge_flow_times_rounded]  # We need to ignore that a link costs t=0
            edge_flow_times_rounded = pd.DataFrame({"link_id": instance["links_id"], "edge_flow_times_rounded": edge_flow_times_rounded})
            solution_scheme = solution_scheme.merge(edge_flow_times_rounded, how="left", on="link_id")

            new_nodes = pd.DataFrame(index=pd.MultiIndex.from_product([solution_scheme_nodes["node_id"], np.arange(len(self.solution_scheme_time))],
                                                                      names=["node_id", "time_idx"])).reset_index()
            time_pandas = pd.DataFrame({"time": self.solution_scheme_time, "time_idx": np.arange(len(self.solution_scheme_time))})
            new_nodes = new_nodes.merge(time_pandas, how="left", on="time_idx")
            updated_solution_scheme_nodes = new_nodes.merge(solution_scheme_nodes, how="left", on="node_id")
            updated_solution_scheme_nodes["new_node_id"] = np.arange(len(updated_solution_scheme_nodes))
            # Matrix original_node_idx x time -> new node index
            matrix_node_time = np.array(updated_solution_scheme_nodes["new_node_id"]).reshape(len(solution_scheme_nodes), len(self.solution_scheme_time))

            updated_solution_scheme = []

            def add_edge(i, j, t, edge_idx, i_original, j_original):
                updated_solution_scheme.append({"new_link_from": i, "new_link_to": j, "time": t, "link_id": edge_idx, "link_from": i_original, "link_to": j_original})

            for t_idx in range(len(self.solution_scheme_time)):

                # Add edges between same node in time
                if self.args.feature not in ["withoutedges", "withoutedgeslogic", "withoutedgessquare"]:
                    for node_id in instance["nodes_id"]:
                        if t_idx + 1 < len(self.solution_scheme_time):
                            i_new = matrix_node_time[node_id, t_idx]
                            j_new = matrix_node_time[node_id, t_idx + 1]
                            add_edge(i_new, j_new, self.solution_scheme_time[t_idx], -node_id - 1, node_id, node_id)

                # Add edges between different nodes in time
                for arc in solution_scheme.to_dict("records"):
                    i, j = arc["link_from"], arc["link_to"]
                    travel_time_ij = arc["edge_flow_times_rounded"]
                    i_new = matrix_node_time[i, t_idx]
                    if t_idx + travel_time_ij < len(self.solution_scheme_time):
                        j_new = matrix_node_time[j, t_idx + travel_time_ij]
                        add_edge(i_new, j_new, self.solution_scheme_time[t_idx], arc["link_id"], i, j)

            updated_solution_scheme = pd.DataFrame(updated_solution_scheme)
            return updated_solution_scheme, updated_solution_scheme_nodes

        def adding_capacity_dimension(solution_scheme, y):
            self.maximum_capacity = int(np.ceil(max(y[y.index.get_level_values("link_id") >= 0].target)))
            self.num_max_capacity_groups = math.ceil(self.maximum_capacity / self.args.num_bundled_capacities)
            self.solution_scheme_capacity_group = list(range(self.num_max_capacity_groups))
            updated_solution_scheme = self.add_capacity_groups(solution=solution_scheme.set_index(self.solution_scheme_metrics),
                                                               capacity_per_entry=[self.maximum_capacity] * len(solution_scheme))
            updated_solution_scheme = updated_solution_scheme.reset_index().merge(solution_scheme, how="left", on=self.solution_scheme_metrics)
            y = self.add_capacity_groups(solution=y, capacity_per_entry=y.target)
            return updated_solution_scheme, y

        assert isinstance(instance, dict)

        self.initialize_solution_scheme(instance)

        solution_scheme = pd.DataFrame({"link_id": self.solution_scheme_linksId,
                                        "link_from": instance["link_from"], "link_to": instance["link_to"],
                                        "new_link_from": instance["link_from"], "new_link_to": instance["link_to"]})
        self.solution_scheme_metrics.append("link_id")
        solution_scheme_nodes = pd.DataFrame({"node_id": instance["nodes_id"], "x": instance["nodes_x"], "y": instance["nodes_y"]})
        self.solution_scheme_metrics_nodes.append("node_id")
        if self.args.trip_individual_theta:
            self.solution_scheme_metrics.append("commodity")
            solution_scheme = adding_trip_dimension(solution_scheme)
        if self.args.time_variant_theta:
            self.solution_scheme_metrics.append("time")
            self.solution_scheme_metrics_nodes.append("time")
            solution_scheme, solution_scheme_nodes = adding_time_dimension(solution_scheme, solution_scheme_nodes)

        y = self.load_y_from_instance(instance=instance)
        y = self.shrink_dimensions(y, y_column_name)
        if self.args.feature in ["withoutedges", "withoutedgeslogic", "withoutedgessquare"]:
            y = y[y.index.get_level_values("link_id") >= 0]

        if self.args.capacity_individual_theta:
            solution_scheme, y = adding_capacity_dimension(solution_scheme, y)
            self.solution_scheme_metrics.append("index_capacity_group")

        self.solution_scheme = solution_scheme.set_index(self.solution_scheme_metrics)
        self.solution_scheme_nodes = solution_scheme_nodes.set_index(self.solution_scheme_metrics_nodes)
        y = self.solution_scheme.merge(y, how="left", on=self.solution_scheme.index.names)[list(y.columns)]
        return y[y_column_name].fillna(0)

    def add_capacity_groups(self, solution, capacity_per_entry):
        y_new = pd.DataFrame()
        y_new.index = pd.MultiIndex.from_tuples([idx if isinstance(idx, tuple) else (idx,) for idx_list in [[i] * math.ceil(t) for i, t in zip(solution.index, capacity_per_entry)] for idx in idx_list])
        y_new.index = y_new.index.set_names(solution.index.names)
        y_new["index_counter"] = [idx_counter for idx_counter_list in [list(np.arange(math.ceil(t))) if i >= 0 else list(np.zeros(math.ceil(t))) for i, t in zip(solution.index.get_level_values("link_id"), capacity_per_entry)] for idx_counter in idx_counter_list]

        if "target" in solution.columns:
            new_target = []
            for e in solution["target"]:
                target_value, remainder = divmod(e, 1)
                a = [1] * int(target_value)
                if remainder > 0:
                    a.append(remainder)
                new_target += a
            y_new["target"] = new_target

        num_groups = math.ceil(max(capacity_per_entry) / self.args.num_bundled_capacities)
        index_counter = np.arange(num_groups * self.args.num_bundled_capacities)
        index_capacity_group = np.meshgrid(np.arange(self.args.num_bundled_capacities), np.arange(num_groups))[1].flatten()
        index_capacity_group_merger = pd.DataFrame({"index_counter": index_counter, "index_capacity_group": index_capacity_group})

        index_names = y_new.index.names
        y_new = y_new.reset_index().merge(index_capacity_group_merger, how="left", on="index_counter").set_index(index_names)
        y_new = y_new.set_index(["index_capacity_group"], append=True)
        y_new = y_new.drop(columns=["index_counter"])
        y_new = y_new.groupby(y_new.index.names).sum()
        return y_new

