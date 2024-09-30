import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp
import scipy


def plot_time_expanded_graph(arcs, nodes, T, weights=None):
    matplotlib.use('TkAgg')
    cmap = matplotlib.cm.get_cmap('viridis')
    colors = {t: cmap(t / len(T)) for t in range(len(T))}

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')

    # defining all 3 axis
    for arc in arcs.to_dict("records"):
        i, j = arc["i"], arc["j"]
        t_start = nodes.iloc[i]["time"]
        t_end = nodes.iloc[j]["time"]
        z_y = np.array([nodes.iloc[i]["y"], nodes.iloc[j]["y"]])
        x_time = np.array([t_start, t_end])
        y_x = np.array([nodes.iloc[i]["x"], nodes.iloc[j]["x"]])

        # plotting
        if weights is None:
            c = colors[t_end]
        else:
            c = cmap(weights[arc["arc_idx"]])
        ax.plot3D(x_time, y_x, z_y, color=c, marker="o")

    ax.set_title('Time expanded street network')
    ax.set_xlabel('Time')
    ax.set_ylabel('x coordinate')
    ax.set_zlabel('y coordinate')
    plt.show()


def get_time_expanded_graph(nodes, arcs, commodities, inflow, instance):

    """
    We expand the graph with a time dimension. Therefore, a flow solution is still x_{commodity, time_expanded_edge}
    """
    T = instance["solution_representation"].solution_scheme_time

    # Discretize times to traverse edge
    edge_flow_times = np.array(instance["link_length"]) / np.array(instance["link_freespeed"])
    edge_flow_times_rounded = [(np.abs(t - instance["solution_representation"].time_steps)).argmin() for t in edge_flow_times]  # discretize edge_flow_times to T

    edge_flow_times_rounded = [flow_time if flow_time > 0 else 1 for flow_time in edge_flow_times_rounded]  # We need to ignore that a link costs t=0
    edge_flow_times_rounded = pd.DataFrame({"link_id": instance["links_id"], "edge_flow_times_rounded": edge_flow_times_rounded})
    arcs = arcs.merge(edge_flow_times_rounded, how="left", on="link_id")


    # GET NEW NODES
    # New node index
    new_nodes = pd.DataFrame(index=pd.MultiIndex.from_product([nodes["node_id"], np.arange(len(T))], names=["original_node_id", "time_idx"])).reset_index()
    new_nodes = new_nodes.merge(nodes, how="left", left_on="original_node_id", right_on="node_id")
    new_nodes["node_id"] = np.arange(len(new_nodes))
    # Matrix original_node_idx x time -> new node index
    matrix_node_time = np.array(new_nodes["node_id"]).reshape(len(nodes), len(T))

    # ASSIGN STARTING TIME TO COMMODITIES
    commodities["time_idx"] = [(np.abs(t - T)).argmin() for t in commodities["starting_time"]]

    # GET NEW EDGES
    new_edges = []

    def add_edge(i, j, t, edge_idx, i_original, j_original, cap, index_capacity_group):
        new_edges.append({"i": i, "j": j, "time": t, "link_id": edge_idx, "i_original": i_original,
                          "j_original": j_original, "capacity": cap, "index_capacity_group": index_capacity_group})

    for t_idx in range(len(T)):

        # Add edges between same node in time
        for node in nodes.to_dict("records"):
            if t_idx+1 < len(T):
                i_new = matrix_node_time[node["node_id"], t_idx]
                j_new = matrix_node_time[node["node_id"], t_idx+1]
                add_edge(i_new, j_new, T[t_idx], -node["node_id"]-1, node["node_id"], node["node_id"], cap=np.inf, index_capacity_group=0)

        # Add edges between different nodes in time
        for arc in arcs.to_dict("records"):
            i, j = arc["i"], arc["j"]
            travel_time_ij = arc["edge_flow_times_rounded"]
            i_new = matrix_node_time[i, t_idx]
            if t_idx+travel_time_ij < len(T):
                j_new = matrix_node_time[j, t_idx+travel_time_ij]
                add_edge(i_new, j_new, T[t_idx], arc["arc_idx"], i, j, cap=arc["capacity"], index_capacity_group=arc["index_capacity_group"])

    new_edges = pd.DataFrame(new_edges)
    new_edges["arc_idx"] = np.arange(len(new_edges))

    # GET NEW INFLOW
        # commodity, node
    # Convert original inflow into pandas
    inflow = inflow.rename(columns={"node_id": "original_node_id"})
    # Merge new inflow pandas with original inflow values
    new_inflow = pd.DataFrame(index=pd.MultiIndex.from_product([commodities["commodity"], new_nodes["node_id"]], names=["commodity", "node_id"])).reset_index()
    new_inflow = new_inflow.merge(new_nodes, how="left", on="node_id")
    new_inflow = new_inflow.merge(inflow, how="left", on=["commodity", "original_node_id"])

    copy_commodities = copy.deepcopy(commodities)
    new_inflow = new_inflow.merge(copy_commodities[["commodity", "origin", "supply", "time_idx"]], how="left", left_on=["commodity", "original_node_id", "time_idx"], right_on=["commodity", "origin", "time_idx"])
    copy_commodities["time_idx"] = max(new_inflow["time_idx"])
    new_inflow = new_inflow.merge(copy_commodities[["commodity", "destination", "demand", "time_idx"]], how="left", left_on=["commodity", "original_node_id", "time_idx"], right_on=["commodity", "destination", "time_idx"])
    new_inflow = new_inflow.fillna(0)
    new_inflow["inflow_value"] = new_inflow["supply"] + new_inflow["demand"]
    new_inflow = new_inflow.drop(columns=["origin", "supply", "destination", "demand"])

    return new_nodes, new_edges, new_inflow
