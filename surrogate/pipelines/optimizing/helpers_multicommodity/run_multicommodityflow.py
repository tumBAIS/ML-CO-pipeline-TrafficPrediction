import time
import gurobipy as gp
import numpy as np
import scipy
from gurobipy import GRB
import multiprocessing as mp
import matplotlib.pyplot as plt


def run_conventional_model(commodities, arcs, nodes, cost, inflow, lb=0.0, ub=float('inf')):
    time_start_gurobi_model = time.time()
    m = gp.Model("netflow")
    m.Params.Threads = 1
    m.setParam("OutputFlag", 0)
    if not isinstance(lb, float):
        lb = {(cost["commodity"], cost["i"], cost["j"]): cost["value"] for cost in cost.to_dict("records")}
    if not isinstance(ub, float):
        ub = {(cost["commodity"], cost["i"], cost["j"]): cost["value"] for cost in cost.to_dict("records")}
    commodities_gurobi = np.array(commodities["commodity"])
    nodes_gurobi = np.array(nodes["node_id"])
    arcs_gurobi, capacity_gurobi = gp.multidict({(arc["i"], arc["j"]): arc["capacity"] for arc in arcs.to_dict("records")})

    cost_gurobi = {(cost["commodity"], cost["i"], cost["j"]): cost["target"] for cost in cost.to_dict("records")}

    inflow_gurobi = {(inflow_i["commodity"], inflow_i["node_id"]): inflow_i["inflow_value"] for inflow_i in inflow.to_dict("records")}

    # Create variables
    flow = m.addVars(commodities_gurobi, arcs_gurobi, obj=cost_gurobi, name="flow", lb=lb, ub=ub)
    # Arc-capacity constraints
    m.addConstrs((flow.sum("*", i, j) <= capacity_gurobi[i, j] for i, j in arcs_gurobi), "cap")
    # Flow-conservation constraints
    m.addConstrs(
        (
            flow.sum(h, "*", j) + inflow_gurobi[h, j] == flow.sum(h, j, "*")
            for h in commodities_gurobi
            for j in nodes_gurobi
        ),
        "node",
    )
    time_end_gurobi_model = time.time()
    print(f"TIME initializing gurobi model: {time_end_gurobi_model - time_start_gurobi_model} seconds")

    # Compute optimal solution
    time_start_optimization = time.time()
    m.optimize()
    time_end_optimization = time.time()
    print(f"TIME optimizing gurobi model: {time_end_optimization - time_start_optimization} seconds")

    if m.Status == GRB.OPTIMAL:
        obj_original = m.getObjective()
        print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m.Params.Threads}")
        return m.getAttr("X", flow)
    else:
        return False


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


def get_capacity_constraint(arcs, commodities):
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
    rhs_capacity_constraints = np.array(arcs["capacity"])
    return A_capacity_constraints, rhs_capacity_constraints


def run_matrix_model(args, nodes, arcs, inflow, cost, commodities):
    # Create optimization model
    time_start_gurobi_model = time.time()
    with gp.Env() as env, gp.Model(env=env) as m_matrix:
        m_matrix.Params.Threads = 1
        m_matrix.setParam("OutputFlag", 0)
        m_matrix.Params.Seed = 1

        # Initialize costs
        cost_variable = np.array(cost["target"])

        matrix_flow = m_matrix.addMVar(shape=len(cost_variable), name="matrix_flow", vtype=GRB.CONTINUOUS)
        m_matrix.setObjective(cost_variable @ matrix_flow)

        # Capacity constraints
        A_capacity_constraints, rhs_capacity_constraints = get_capacity_constraint(arcs, commodities)
        m_matrix.addConstr(A_capacity_constraints @ matrix_flow <= rhs_capacity_constraints, name="capacity constraints")

        # Flow-conservation constraints
        A_flow_conservation, rhs_flow_conservation = get_flow_conservation_constraint(arcs, nodes, inflow, commodities)
        m_matrix.addConstr(A_flow_conservation @ matrix_flow == rhs_flow_conservation, name="flow_conservation")

        time_end_gurobi_model = time.time()
        print(f"TIME initializing gurobi model: {time_end_gurobi_model - time_start_gurobi_model} seconds")

        # Compute optimal solution
        time_start_optimization = time.time()

        m_matrix.optimize()   #callback=cb
        time_end_optimization = time.time()
        print(f"TIME optimizing gurobi model: {time_end_optimization - time_start_optimization} seconds")

        if m_matrix.Status == GRB.OPTIMAL or m_matrix.Status == GRB.INTERRUPTED:
            obj_original = m_matrix.getObjective()
            print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m_matrix.Params.Threads}")
            solution = matrix_flow.X
            cost["value"] = solution
        elif m_matrix.Status == GRB.TIME_LIMIT:
            print(f"OPTIMIZATION exceeded TIME LIMIT")
            obj_original = m_matrix.getObjective()
            print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m_matrix.Params.Threads}")
            solution = matrix_flow.X
            cost["value"] = solution
        else:
            raise Exception(f"The CO-layer did not find a solution, with exit code: {m_matrix.Status}")

    return cost.reset_index()


def run_matrix_model_uncapacitated(args, nodes, arcs, inflow, cost, commodities):
    # Create optimization model
    time_start_gurobi_model = time.time()
    with gp.Env() as env, gp.Model(env=env) as m_matrix:
        m_matrix.Params.Threads = 1
        m_matrix.setParam("OutputFlag", 0)
        m_matrix.Params.Seed = 1

        # Initialize costs
        cost_variable = np.array(cost["target"])

        matrix_flow = m_matrix.addMVar(shape=len(cost_variable), name="matrix_flow")
        m_matrix.setObjective(cost_variable @ matrix_flow)

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
            solution = matrix_flow.X
            cost["value"] = solution
        elif m_matrix.Status == GRB.TIME_LIMIT:
            print(f"OPTIMIZATION exceeded TIME LIMIT")
            return np.nan
        else:
            raise Exception(f"The CO-layer did not find a solution, with exit code: {m_matrix.Status}")

    return cost.reset_index()