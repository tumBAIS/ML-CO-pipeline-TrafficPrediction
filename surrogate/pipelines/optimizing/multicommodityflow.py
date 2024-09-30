import numpy as np
import time
import pandas as pd
from surrogate.pipelines.optimizing.helpers_multicommodity.get_timeexpanded_components import get_time_expanded_graph, plot_time_expanded_graph
from surrogate.pipelines.optimizing.helpers_multicommodity.get_capacity_expanded_components import get_capacity_expanded_graph, set_capacity
from surrogate.pipelines.optimizing.helpers_multicommodity.run_multicommodityflow import run_conventional_model, run_matrix_model
from surrogate.pipelines.optimizing.helpers_multicommodity.get_multicommodityflow_components import from_instance_to_multicommodityflow_components


def get_multicommodity_flow_components(instance, thetas):
    nodes, arcs, commodities, inflow = from_instance_to_multicommodityflow_components(instance)
    cost = instance["solution_representation"].load_y(thetas)
    cost = pd.DataFrame(cost)
    return nodes, arcs, commodities, inflow, cost


def get_capacityexpanded_components(args, arcs, instance):
    time_start_capacity_expansion = time.time()
    arcs = get_capacity_expanded_graph(args=args, arcs=arcs, max_capacity=instance["solution_representation"].maximum_capacity)
    time_end_capacity_expansion = time.time()
    print(f"TIME capacity-expansion graph: {time_end_capacity_expansion - time_start_capacity_expansion} seconds")
    return arcs


def get_timeexpaned_components(args, verbose, nodes, arcs, commodities, inflow, instance):
    time_start_time_expansion = time.time()
    nodes, arcs, inflow = get_time_expanded_graph(nodes=nodes, arcs=arcs, commodities=commodities, inflow=inflow, instance=instance)
    time_end_time_expansion = time.time()
    if verbose:
        plot_time_expanded_graph(arcs=arcs, nodes=nodes, T=instance["solution_representation"].solution_scheme_time)
    print(f"TIME time-expansion graph: {time_end_time_expansion - time_start_time_expansion} seconds")
    return nodes, arcs, inflow


def get_cost(commodities, edges, cost):
    # GET NEW COST
    # commodity, i, j
    new_cost = pd.DataFrame(index=pd.MultiIndex.from_product([commodities["commodity"], edges["arc_idx"]], names=["commodity", "arc_idx"])).reset_index()
    new_cost = new_cost.merge(edges, how="left", on="arc_idx")
    new_cost = new_cost.merge(cost, how="left", on=cost.index.names)
    new_cost = new_cost.drop(columns=new_cost.columns.difference(["commodity", "time", "link_id", "arc_idx", "target", "i", "j", "index_capacity_group"]))
    new_cost["target"] = new_cost["target"].fillna(0)
    new_cost = new_cost.set_index(cost.index.names)
    return new_cost


def optimize(args, thetas, instance, verbose=False, latency_function=None):

    print(f"INSTANCE NAME: {instance['training_instance_name']}")

    thetas = -1 * thetas  # we need a maximization problem - but the shortest path is a minimization problem - so we negate the thetas.
    thetas[thetas < 0] = 0


    nodes, arcs, commodities, inflow, cost = get_multicommodity_flow_components(instance, thetas)

    # Get capacity-expansion of multi-commodity flow
    if args.capacity_individual_theta:
        arcs = get_capacityexpanded_components(args, arcs, instance)

    # Get time-expansion of multi-commodity flow
    if args.time_variant_theta:
        nodes, arcs, inflow = get_timeexpaned_components(args, verbose, nodes, arcs, commodities, inflow, instance)

    # Here we must set the capacity to the true capacity
    if not args.capacity_individual_theta:
        arcs = set_capacity(args, arcs, instance)

    cost = get_cost(commodities, arcs, cost)

    # Create optimization model
    solution = run_matrix_model(args, nodes, arcs, inflow, cost, commodities)

    if solution is np.nan:
        return np.nan

    # Merge solution with solution scheme
    y = instance["solution_representation"].load_y(solution, y_column_name="value")

    if verbose:
        _ = run_conventional_model(commodities, arcs, nodes, cost, inflow)
        if not run_conventional_model(commodities, arcs, nodes, cost, inflow, lb=solution, ub=solution):
            raise Exception("Found solution not valid")

        for com in np.array(commodities["commodity"]):
            flow = np.array(solution[solution["commodity"] == com]["value"])
            plot_time_expanded_graph(arcs=arcs, nodes=nodes, T=T, weights=[(flow_i / max(flow)) for flow_i in flow])

    return np.array(y)
