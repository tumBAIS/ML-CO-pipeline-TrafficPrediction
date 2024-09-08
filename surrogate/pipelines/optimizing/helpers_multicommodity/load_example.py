import numpy as np
import gurobipy as gp
import pandas as pd


def get_example():
    """
                    Boston
        Detroit                 Denver
                    New York

        """

    T = np.arange(8)

    # Create commodities
    commodity_names = ["Pencils", "Pens"]
    commodities = pd.DataFrame({"commodity": commodity_names})

    # Create nodes
    nodes_names = ["Detroit", "Denver", "Boston", "New York"]
    nodes_x = [0, 2, 1, 1]
    nodes_y = [1, 1, 0, 3]
    nodes = pd.DataFrame({"node_id": nodes_names, "x": nodes_x, "y": nodes_y})

    # Create arcs
    i = ["Detroit", "Detroit", "Boston", "New York"]
    j = ["Boston", "New York", "Denver", "Denver"]
    capacity = [20, 30, 40, 50]
    edge_flow_times_rounded = [1, 2, 3, 4]
    arc_idx = np.arange(len(i))
    arcs = pd.DataFrame({"arc_idx": arc_idx, "i": i, "j": j, "capacity": capacity, "edge_flow_times_rounded": edge_flow_times_rounded})

    # Create costs
    commodity = ["Pencils", "Pencils", "Pencils", "Pencils", "Pens", "Pens", "Pens", "Pens"]
    link_id = [0, 1, 2, 3, 0, 1, 2, 3]
    target = [10, 20, 30, 40, 50, 60, 70, 80]
    cost = pd.DataFrame({"commodity": commodity, "link_id": link_id, "target": target})

    # Create inflow
    commodity = ["Pencils", "Pencils", "Pencils", "Pencils", "Pens", "Pens", "Pens", "Pens"]
    node_id = ["Detroit", "Denver", "Boston", "New York", "Detroit", "Denver", "Boston", "New York"]
    inflow_value = [50, -50, 0, 0, 30, -30, 0, 0]
    inflow = pd.DataFrame({"commodity": commodity, "node_id": node_id, "inflow_value": inflow_value})

    commodities_converter = {commodity: commodity_id for commodity_id, commodity in enumerate(commodity_names)}
    nodes_converter = {node: node_id for node_id, node in enumerate(nodes_names)}

    commodities["commodity"] = [commodities_converter[com] for com in commodities["commodity"]]
    nodes["node_id"] = [nodes_converter[i] for i in nodes["node_id"]]
    arcs["i"] = [nodes_converter[i] for i in arcs["i"]]
    arcs["j"] = [nodes_converter[i] for i in arcs["j"]]
    cost["commodity"] = [commodities_converter[com] for com in cost["commodity"]]
    inflow["commodity"] = [commodities_converter[com] for com in inflow["commodity"]]
    inflow["node_id"] = [nodes_converter[i] for i in inflow["node_id"]]

    cost = cost.set_index(["commodity", "link_id"])

    return nodes, arcs, commodities, inflow, cost, T
