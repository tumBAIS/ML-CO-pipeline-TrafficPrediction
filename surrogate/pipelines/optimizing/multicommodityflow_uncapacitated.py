import numpy as np
from surrogate.pipelines.optimizing.helpers_multicommodity.get_timeexpanded_components import plot_time_expanded_graph
from surrogate.pipelines.optimizing.helpers_multicommodity.run_multicommodityflow import run_conventional_model, run_matrix_model, run_matrix_model_uncapacitated
from surrogate.pipelines.optimizing.multicommodityflow import get_multicommodity_flow_components, get_timeexpaned_components, set_capacity, get_cost


def optimize(args, thetas, instance, verbose=False, latency_function=None):

    thetas = -1 * thetas  # we need a maximization problem - but the shortest path is a minimization problem - so we negate the thetas.
    thetas[thetas < 0] = 0

    # Get multi-commodity flow components
    nodes, arcs, commodities, inflow, cost = get_multicommodity_flow_components(instance, thetas)

    # Get capacity-expansion of multi-commodity flow
    if args.capacity_individual_theta:
        raise Exception("Capacity individual theta not possible in uncapacitated multicommodityflow")

    # Get time-expansion of multi-commodity flow
    if args.time_variant_theta:
        nodes, arcs, inflow = get_timeexpaned_components(args, verbose, nodes, arcs, commodities, inflow, instance)

    cost = get_cost(commodities, arcs, cost)

    # Create optimization model
    solution = run_matrix_model_uncapacitated(args, nodes, arcs, inflow, cost, commodities)

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