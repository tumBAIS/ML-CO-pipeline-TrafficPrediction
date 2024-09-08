import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surrogate.pipelines.optimizing.helpers_multicommodity.run_wardropequilibrium import find_wardrop_equilibria
from surrogate.pipelines.optimizing.multicommodityflow import get_multicommodity_flow_components, get_cost, get_timeexpaned_components
from surrogate.pipelines.optimizing.successiveaverages import solve_added_shortest_path


def get_travel_time(args, latency_function, y, instance):
    theta = latency_function.predict([instance])
    return theta[0] + theta[1] * y


def check(args, latency_function, y, instance):
    travel_time = get_travel_time(args, latency_function, y, instance)
    y_old = solve_added_shortest_path(travel_time, instance).values
    travel_time = get_travel_time(args, latency_function, y_old, instance)
    y_new = solve_added_shortest_path(travel_time, instance).values
    plt.plot(np.abs(y_old - y_new), label="diff")
    plt.title("Check")
    plt.show()


def check_solution(y, edge_flow, aggregated_flow, objective_value, nodes, arcs, inflow, cost, commodities, latency):
    # Check aggregation constraint
    aggregated_y = y.groupby("link_id").sum()
    assert np.allclose(aggregated_y.values, aggregated_flow)
    # Check flow conservation constraint
    for n_counter, n in enumerate(nodes.to_dict("records")):
        print(f"{n} : {n_counter}/{len(nodes)}")
        for c in commodities.to_dict("records"):
            #try:
            sum_ingoing = edge_flow.groupby(["j", "commodity"]).sum().loc[(n["node_id"], c["commodity"]), "value"]
            sum_outgoing = edge_flow.groupby(["i", "commodity"]).sum().loc[(n["node_id"], c["commodity"]), "value"]
            inflow_ij = inflow[(inflow["node_id"] == n["node_id"]) & (inflow["commodity"] == c["commodity"])].iloc[0, -1]
            assert math.isclose(sum_outgoing - sum_ingoing, inflow_ij, abs_tol=1e-09)
            #except:
            #    print("Hier")
    # Check objective value
    objective_value_post = np.sum(latency["latency_intercept"].values * aggregated_flow + (1/2) * latency["latency_x"].values * aggregated_flow**2)
    assert math.isclose(objective_value_post, objective_value, abs_tol=1e-09)


def get_latency_parameters(arcs, cost):
    # arcs and thetas might have different length as arcs contain arcs connecting equal nodes over time, which are not considered in theta
    latency_parameters = arcs.merge(cost, how="left", on=cost.index.names)
    if isinstance(cost["target"].values[0], tuple):
        latency_parameters["target"] = latency_parameters["target"].apply(lambda x: (0, 0) if x is np.nan else x)
    else:
        latency_parameters["target"] = latency_parameters["target"].fillna(0)
    return np.asarray(latency_parameters["target"].to_list()).T


def optimize(args, thetas, instance, latency_function, verbose=False):

    verbose = False

    # Get multi-commodity flow components
    nodes, arcs, commodities, inflow, cost = get_multicommodity_flow_components(instance, thetas)

    # Get time-expansion of multi-commodity flow
    if args.time_variant_theta:
        nodes, arcs, inflow = get_timeexpaned_components(args, verbose, nodes, arcs, commodities, inflow, instance)

    latency_parameters = get_latency_parameters(arcs, cost)
    cost = get_cost(commodities, arcs, cost)
    if args.co_optimizer == "wardropequilibriaRegularized":
        latency = {"latency_intercept": latency_parameters, "latency_x": np.repeat(-1, len(latency_parameters))}
    elif args.co_optimizer == "wardropequilibria":
        latency = {"latency_intercept": latency_parameters[0], "latency_x": latency_parameters[1]}
    latency["latency_intercept"] = -1 * latency["latency_intercept"]
    latency["latency_intercept"][latency["latency_intercept"] < 0] = 0
    latency["latency_x"] = -1 * latency["latency_x"]
    latency["latency_x"][latency["latency_x"] < 0] = 0
    latency = pd.DataFrame(latency)

    # Create optimization model
    solution, aggregated_flow, objective_value = find_wardrop_equilibria(args, nodes, arcs, inflow, cost, commodities, latency)

    # Merge solution with solution scheme
    y = instance["solution_representation"].load_y(solution, y_column_name="value")

    if verbose:
        check_solution(y, solution, aggregated_flow, objective_value, nodes, arcs, inflow, cost, commodities, latency)
        check(args, latency_function, y.values, instance)

    return np.array(y)
