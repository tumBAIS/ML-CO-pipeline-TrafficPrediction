import time
import gurobipy as gp
import numpy as np
from gurobipy import GRB
import scipy


def get_flow_conservation_constraint(arcs, nodes, inflow, commodities):
    inflow_matrix = np.zeros((len(nodes), len(commodities)))
    inflow_matrix[np.array(inflow["node_id"]).astype(int), np.array(inflow["commodity"]).astype(int)] = np.array(inflow["inflow_value"])

    commodity_edge_matrix = np.arange((len(commodities) * len(arcs)))
    commodity_edge_matrix = commodity_edge_matrix.reshape((len(commodities), len(arcs)))
    arcs_ending_nodes = np.array(arcs["j"])
    arcs_starting_nodes = np.array(arcs["i"])
    sparse_matrix_row = []
    sparse_matrix_col = []
    sparse_matrix_val = []
    counter = 0
    num_commodities = len(commodities)
    for node in nodes.to_dict("records"):
        cols_in = commodity_edge_matrix[:, arcs_ending_nodes == node["node_id"]]
        sparse_matrix_col += list(cols_in.flatten())
        sparse_matrix_row += list(np.repeat(range(counter, counter + num_commodities), cols_in.shape[1]))
        sparse_matrix_val += list(np.full(cols_in.shape, -1).flatten())
        cols_out = commodity_edge_matrix[:, arcs_starting_nodes == node["node_id"]]
        sparse_matrix_col += list(cols_out.flatten())
        sparse_matrix_row += list(np.repeat(range(counter, counter + num_commodities), cols_out.shape[1]))
        sparse_matrix_val += list(np.full(cols_out.shape, 1).flatten())
        counter += num_commodities
    rhs_flow_conservation = inflow_matrix.flatten()
    A_flow_conservation = scipy.sparse.csr_array((sparse_matrix_val, (sparse_matrix_row, sparse_matrix_col)))
    return A_flow_conservation, rhs_flow_conservation


def get_aggregation_constraint(arcs, commodities):
    # Capacity constraints
    commodity_edge_matrix = np.arange((len(commodities) * len(arcs)))
    commodity_edge_matrix = commodity_edge_matrix.reshape((len(commodities), len(arcs)))
    sparse_matrix_row = []
    sparse_matrix_col = []
    sparse_matrix_val = []
    for arc in arcs.to_dict("records"):
        cols = commodity_edge_matrix[:, arc["arc_idx"]]
        sparse_matrix_col += list(cols)
        sparse_matrix_row += [arc["arc_idx"]] * len(cols)
        sparse_matrix_val += [1] * len(cols)

    A_capacity_constraints = scipy.sparse.csr_array((sparse_matrix_val, (sparse_matrix_row, sparse_matrix_col)))
    return A_capacity_constraints


def find_wardrop_equilibria(args, nodes, arcs, inflow, cost, commodities, latency):

    # Create optimization model
    time_start_gurobi_model = time.time()
    with gp.Env() as env, gp.Model(env=env) as m_matrix:
        m_matrix.Params.Threads = 1
        m_matrix.setParam("OutputFlag", 0)
        m_matrix.Params.Seed = 1

        if args.obtimization_gap != "default":
            m_matrix.setParam('MIPGap', args.obtimization_gap)

        # Decision variables
        matrix_flow = m_matrix.addMVar(shape=len(arcs) * len(commodities), name="matrixFlow") #, vtype=GRB.INTEGER
        aggregated_matrix_flow = m_matrix.addMVar(shape=len(arcs), name="aggregatedMatrixFlow") #, vtype=GRB.INTEGER
        m_matrix.setObjective(np.array(latency["latency_intercept"]) @ aggregated_matrix_flow +
                              aggregated_matrix_flow @ ((1/2) * np.diag(latency["latency_x"])) @ aggregated_matrix_flow)

        # Aggregation constraint
        A_aggregation_constraint = get_aggregation_constraint(arcs, commodities)
        m_matrix.addConstr(A_aggregation_constraint @ matrix_flow == aggregated_matrix_flow, name="aggregation_constraint")

        # Flow-conservation constraints
        A_flow_conservation, rhs_flow_conservation = get_flow_conservation_constraint(arcs, nodes, inflow, commodities)
        m_matrix.addConstr(A_flow_conservation @ matrix_flow == rhs_flow_conservation, name="flow_conservation")

        time_end_gurobi_model = time.time()
        print(f"TIME initializing gurobi model: {time_end_gurobi_model - time_start_gurobi_model} seconds")

        # Compute optimal solution
        time_start_optimization = time.time()

        m_matrix.optimize()
        time_end_optimization = time.time()
        print(f"TIME optimizing gurobi model: {time_end_optimization - time_start_optimization} seconds")

        if m_matrix.Status == GRB.OPTIMAL:
            obj_original = m_matrix.getObjective()
            print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m_matrix.Params.Threads}")
            cost["value"] = matrix_flow.X
            aggregated_flow = aggregated_matrix_flow.X
            objective_value = obj_original.getValue()
        else:
            raise Exception(f"The CO-layer did not find a solution, with exit code: {m_matrix.Status}")

    return cost.reset_index(), aggregated_flow, objective_value
