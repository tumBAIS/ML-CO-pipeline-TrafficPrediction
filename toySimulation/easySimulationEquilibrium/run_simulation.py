import random
import numpy as np
import pandas as pd
from surrogate.pipelines.optimizing.helpers_multicommodity.get_multicommodityflow_components import from_instance_to_multicommodityflow_components
from surrogate.pipelines.optimizing.helpers_multicommodity.get_timeexpanded_components import get_time_expanded_graph
from surrogate.pipelines.optimizing.helpers_multicommodity.run_wardropequilibrium import find_wardrop_equilibria
from surrogate.pipelines.optimizing.multicommodityflow import get_cost
from surrogate.pipelines.solution_representation import Solution
from surrogate.pipelines.optimizing.wardropequilibrium import get_latency_parameters


def run_co_wardropequilibrium(args, thetas, instance):

    cost = pd.DataFrame({"link_id": instance["links_id"], "target": list(map(tuple, thetas.T))}).set_index("link_id")

    nodes, arcs, commodities, inflow = from_instance_to_multicommodityflow_components(instance=instance)

    # Get time-expansion of multi-commodity flow
    nodes, arcs, inflow = get_time_expanded_graph(nodes=nodes, arcs=arcs, commodities=commodities, inflow=inflow, instance=instance)

    latency_parameters = get_latency_parameters(arcs, cost)
    cost = get_cost(commodities, arcs, cost)
    latency = {"latency_intercept": latency_parameters[0], "latency_x": latency_parameters[1]}
    latency["latency_intercept"][latency["latency_intercept"] < 0] = 0
    latency["latency_x"][latency["latency_x"] < 0] = 0
    latency = pd.DataFrame(latency)

    # Create optimization model
    solution, aggregated_flow, objective_value = find_wardrop_equilibria(args, nodes, arcs, inflow, cost, commodities, latency)

    return solution


def run_simulation(args, scenario):
    # set seeds
    random.seed(args.seed_algorithm)
    np.random.seed(args.seed_algorithm)

    thetas = np.array([np.array(scenario["link_length"])/100, np.ones(len(scenario["link_length"]))])

    scenario["commodities"] = np.arange(len(scenario["o_d_pairs"]))
    solution_representation = Solution(args)
    solution_representation.initialize_solution_scheme(instance=scenario)
    scenario["solution_representation"] = solution_representation

    solution = run_co_wardropequilibrium(args, thetas=thetas, instance=scenario)

    solution = solution[(solution["value"] > 0) & (~solution["link_id"].isna())]
    scenario["solution_time"] = np.array(solution.time)
    scenario["solution_link"] = np.array(solution.link_id)
    scenario["solution_commodity"] = np.array(solution.commodity)
    scenario["solution_value"] = np.array(solution.value)

    del scenario["solution_representation"]

    return scenario
