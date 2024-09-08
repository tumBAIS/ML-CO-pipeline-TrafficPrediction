import random
import numpy as np
import pandas as pd
from surrogate.pipelines.optimizing.helpers_multicommodity.get_multicommodityflow_components import from_instance_to_multicommodityflow_components
from surrogate.pipelines.optimizing.helpers_multicommodity.get_timeexpanded_components import get_time_expanded_graph
from surrogate.pipelines.optimizing.helpers_multicommodity.run_multicommodityflow import run_matrix_model
from surrogate.pipelines.optimizing.multicommodityflow import get_cost
from surrogate.pipelines.solution_representation import Solution


def run_co_multicommodityflow(args, thetas, instance):

    cost = pd.DataFrame({"link_id": instance["links_id"], "target": thetas}).set_index("link_id")

    # Get multi-commodity flow components
    nodes, arcs, commodities, inflow = from_instance_to_multicommodityflow_components(instance=instance)

    # Get time-expansion of multi-commodity flow
    nodes, arcs, inflow = get_time_expanded_graph(nodes=nodes, arcs=arcs, commodities=commodities, inflow=inflow, instance=instance)

    cost = get_cost(commodities, arcs, cost)

    # Create optimization model
    solution = run_matrix_model(args, nodes, arcs, inflow, cost, commodities)

    return solution


def run_simulation(args, scenario):
    start = 1
    end = 100
    thetas = [random.uniform(start, end) for _ in range(len(scenario["links_id"]))]

    scenario["commodities"] = np.arange(len(scenario["o_d_pairs"]))
    solution_representation = Solution(args)
    solution_representation.initialize_solution_scheme(instance=scenario)
    scenario["solution_representation"] = solution_representation

    solution = run_co_multicommodityflow(args, thetas=thetas, instance=scenario)

    solution = solution[(solution["value"] > 0) & (~solution["link_id"].isna())]
    scenario["solution_time"] = np.array(solution.time)
    scenario["solution_link"] = np.array(solution.link_id)
    scenario["solution_commodity"] = np.array(solution.commodity)

    del scenario["solution_representation"]

    return scenario
