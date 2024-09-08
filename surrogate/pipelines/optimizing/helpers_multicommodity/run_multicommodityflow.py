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
    # Reset variables to original scope
    if not isinstance(lb, float):
        lb = {(cost["commodity"], cost["i"], cost["j"]): cost["value"] for cost in cost.to_dict("records")}
    if not isinstance(ub, float):
        ub = {(cost["commodity"], cost["i"], cost["j"]): cost["value"] for cost in cost.to_dict("records")}
    commodities_gurobi = np.array(commodities["commodity"])
    nodes_gurobi = np.array(nodes["node_id"])
    arcs_gurobi, capacity_gurobi = gp.multidict({(arc["i"], arc["j"]): arc["capacity"] for arc in arcs.to_dict("records")})
    """arcs, capacity = gp.multidict(
        {
            ("Detroit", "Boston"): 100,
            ("Detroit", "New York"): 80,
            ("Detroit", "Seattle"): 120,
            ("Denver", "Boston"): 120,
            ("Denver", "New York"): 120,
            ("Denver", "Seattle"): 120,
        }
    )"""

    # Cost for triplets commodity-source-destination
    cost_gurobi = {(cost["commodity"], cost["i"], cost["j"]): cost["target"] for cost in cost.to_dict("records")}
    """cost = {
        ("Pencils", "Detroit", "Boston"): 10,
        ("Pencils", "Detroit", "New York"): 20,
        ("Pencils", "Detroit", "Seattle"): 60,
        ("Pencils", "Denver", "Boston"): 40,
        ("Pencils", "Denver", "New York"): 40,
        ("Pencils", "Denver", "Seattle"): 30,
        ("Pens", "Detroit", "Boston"): 20,
        ("Pens", "Detroit", "New York"): 20,
        ("Pens", "Detroit", "Seattle"): 80,
        ("Pens", "Denver", "Boston"): 60,
        ("Pens", "Denver", "New York"): 70,
        ("Pens", "Denver", "Seattle"): 30,
    }"""

    # Supply (> 0) and demand (< 0) for pairs of commodity-city
    inflow_gurobi = {(inflow_i["commodity"], inflow_i["node_id"]): inflow_i["inflow_value"] for inflow_i in inflow.to_dict("records")}
    """inflow = {
        ("Pencils", "Detroit"): 50,
        ("Pencils", "Denver"): 60,
        ("Pencils", "Boston"): -50,
        ("Pencils", "New York"): -50,
        ("Pencils", "Seattle"): -10,
        ("Pens", "Detroit"): 60,
        ("Pens", "Denver"): 40,
        ("Pens", "Boston"): -40,
        ("Pens", "New York"): -30,
        ("Pens", "Seattle"): -30,
    }"""

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


#def run_matrix_model(commodities, arcs, nodes, cost, capacity, inflow):
def run_matrix_model(args, nodes, arcs, inflow, cost, commodities):
    # Create optimization model
    time_start_gurobi_model = time.time()
    with gp.Env() as env, gp.Model(env=env) as m_matrix:
        #m_matrix = gp.Model("matrix")
        m_matrix.Params.Threads = 1
        #m_matrix.setParam("Method", 0)
        #m_matrix.setParam("NodeFileStart", 0.1)
        #m_matrix.setParam("PreSparsify", -1)
        #m_matrix.setParam("MIPGap", 0.1)
        #if args.simulation in ["cutoutWorlds", "cutoutWorlds_8"] and args.co_optimizer == "multicommodityflow" and args.learning == "structured" and args.capacity_individual_theta == 1 and args.model == "NN":
        #    m_matrix.Params.timeLimit = 500
        #m_matrix.Params.SolutionLimit = 10
        m_matrix.setParam("OutputFlag", 0)
        m_matrix.Params.Seed = 1
        #m_matrix.Params.Method = 0
        #m_matrix.Params.SolutionNumber = 10
        #m_matrix.Params.PoolSolutions = 10
        #m_matrix.Params.NoRelHeurTime = 2
        #m_matrix.Params.NumericFocus = 1
        #m_matrix.Params.MIPFocus = 3
        #m_matrix.update()

        #if args.obtimization_gap != "default":
            #m_matrix.setParam('MIPGap', args.obtimization_gap)
        #m_matrix.setParam('MIPGap', 0.9)
        #m_matrix.Params.timeLimit = 1
        #m_matrix.Params.LogFile = "mip.log"

        #def cb(model, where):
        #    if where == GRB.Callback.MIPNODE:
        #        # Get model objective
        #        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

                # Has objective changed?
        #        if abs(obj - model._cur_obj) > 1e-8:
                    # If so, update incumbent and time
        #            model._cur_obj = obj
        #            model._time = time.time()

            # Terminate if objective has not improved in 20s
        #    if time.time() - model._time > 30:
        #        model.terminate()

        # Initialize costs
        cost_variable = np.array(cost["target"])

        matrix_flow = m_matrix.addMVar(shape=len(cost_variable), name="matrix_flow", vtype=GRB.CONTINUOUS)
        m_matrix.setObjective(cost_variable @ matrix_flow)

        #if starting_solution is not None:
        #    m_matrix.update()
        #    for v, value in zip(m_matrix.getVars(), starting_solution):
        #        v.Start = value

        # Capacity constraints
        A_capacity_constraints, rhs_capacity_constraints = get_capacity_constraint(arcs, commodities)
        m_matrix.addConstr(A_capacity_constraints @ matrix_flow <= rhs_capacity_constraints, name="capacity constraints")

        # Flow-conservation constraints
        A_flow_conservation, rhs_flow_conservation = get_flow_conservation_constraint(arcs, nodes, inflow, commodities)
        m_matrix.addConstr(A_flow_conservation @ matrix_flow == rhs_flow_conservation, name="flow_conservation")

        time_end_gurobi_model = time.time()
        print(f"TIME initializing gurobi model: {time_end_gurobi_model - time_start_gurobi_model} seconds")

        #if starting_solution is not None:
            #for i, var in enumerate(m_matrix.getVars()):
            #    var_start = m_matrix.getVarByIndex(i)
            #    if var_start is not None:
            #        var_start.setAttr('Start', starting_solution[i])
            #for var, value in starting_solution.items():
            #    var_start = m_matrix.getVarByName(var)
            #    if var_start is not None:
            #        var_start.setAttr('Start', value)
            #matrix_flow.Start = starting_solution
            #m_matrix.update()

        # Compute optimal solution
        time_start_optimization = time.time()
        #m_matrix.update()
        #m_matrix.read("model.sol")
        #m_matrix.update()

        """value = []
        for var in m_matrix.getVars():
            # Get variable name
            #print({var.Start})
            value.append(var.Start)
            #var.LB = var.Start
            #var.UB = var.Start
        print(sum(value))"""
        #is_mip = m_matrix.getAttr("IsMIP")

        # Print the result
        #if is_mip:
        #    print("The model is a MIP (Mixed-Integer Programming) or MILP (Mixed-Integer Linear Programming).")
        #else:
        #    print("The model is an IP (Integer Programming).")
        #m_matrix._cur_obj = float('inf')
        #m_matrix._time = time.time()
        m_matrix.optimize()   #callback=cb
        time_end_optimization = time.time()
        print(f"TIME optimizing gurobi model: {time_end_optimization - time_start_optimization} seconds")

        if m_matrix.Status == GRB.OPTIMAL or m_matrix.Status == GRB.INTERRUPTED:
            obj_original = m_matrix.getObjective()
            print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m_matrix.Params.Threads}")
            solution = matrix_flow.X
            cost["value"] = solution
            #m_matrix.write(f"model.sol")
        elif m_matrix.Status == GRB.TIME_LIMIT:
            print(f"OPTIMIZATION exceeded TIME LIMIT")
            obj_original = m_matrix.getObjective()
            print(f"SOLUTION HAS OBJECTIVE VALUE: {obj_original.getValue()}, used Threads: {m_matrix.Params.Threads}")
            solution = matrix_flow.X
            cost["value"] = solution
            #return np.nan
        else:
            raise Exception(f"The CO-layer did not find a solution, with exit code: {m_matrix.Status}")
        #del env
        #del m_matrix

    #def parse_log(filename):
    #    mip_gaps = []
    #    with open(filename, "r") as file:
    #        for line in file:
    #            if "MIP gap" in line:
    #                mip_gap = float(line.split()[-1])
    #                mip_gaps.append(mip_gap)
    #    return mip_gaps

    #def plot_mip_gap(mip_gaps):
    #    plt.plot(mip_gaps)
    #    plt.xlabel("Iteration")
    #    plt.ylabel("MIP Gap")
    #    plt.title("MIP Gap Plot")
    #    plt.grid(True)
    #    plt.show()

    #mip_gaps = parse_log("mip.log")

    # Plot the MIP gap values
    #plot_mip_gap(mip_gaps)

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

        # No capacity constraints

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
            #print(m_matrix.solCount)
        elif m_matrix.Status == GRB.TIME_LIMIT:
            print(f"OPTIMIZATION exceeded TIME LIMIT")
            return np.nan
        else:
            raise Exception(f"The CO-layer did not find a solution, with exit code: {m_matrix.Status}")

    return cost.reset_index()