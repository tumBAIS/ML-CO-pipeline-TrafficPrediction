import numpy as np
import pandas as pd


def from_instance_to_multicommodityflow_components(instance):

    commodities = pd.DataFrame({"commodity": instance["commodities"],
                                "origin": np.array(instance["o_d_pairs"])[:, 0], "destination": np.array(instance["o_d_pairs"])[:, 1],
                                "supply": [1] * len(instance["o_d_pairs"]), "demand": [-1] * len(instance["o_d_pairs"]),
                                "starting_time": instance["o_d_pairs_starting_time"] if "o_d_pairs_starting_time" in instance else None})

    nodes = pd.DataFrame({"node_id": instance["nodes_id"], "x": instance["nodes_x"], "y": instance["nodes_y"]})


    # Arcs and capacities per arc
    arcs = pd.DataFrame({"arc_idx": instance["links_id"], "i": instance["link_from"], "j": instance["link_to"],
                         "capacity": instance["link_capacity"], "link_id": instance["links_id"], "index_capacity_group": 0, "time": 0})

    # Supply (> 0) and demand (< 0) for pairs of commodity-city
    inflow = pd.DataFrame(index=pd.MultiIndex.from_product([commodities["commodity"], nodes["node_id"]])).reset_index()
    inflow = inflow.merge(commodities[["commodity", "origin", "supply"]], how="left", left_on=["commodity", "node_id"], right_on=["commodity", "origin"])
    inflow = inflow.merge(commodities[["commodity", "destination", "demand"]], how="left", left_on=["commodity", "node_id"], right_on=["commodity", "destination"])
    inflow = inflow.fillna(0)
    inflow["inflow_value"] = inflow["supply"] + inflow["demand"]
    inflow = inflow.drop(columns=["origin", "supply", "destination", "demand"])

    return nodes, arcs, commodities, inflow

