import matsim
import numpy as np
import matplotlib.pyplot as plt
from surrogate.visualizing.event_reader import PlanReader
from surrogate.src.util import load_json, Graph
from surrogate.src import config
from surrogate.pipelines.solution_representation import Solution
import random
import tikzplotlib
from matplotlib import cm


def visualize_solution_from_training_data(args, axis, training_data, heading=None, vmin=None, vmax=None):
    args.time_variant_theta = 0
    args.trip_individual_theta = 0
    args.capacity_individual_theta = 0

    solution_representation = Solution(args)
    y = solution_representation.load_solution_scheme(training_data)
    num_commodities = len(training_data["commodities"])
    print(f"num. commodities: {num_commodities}")

    g = Graph(training_data, y)
    # Visualize network
    g.draw(ax=axis, weights_label="Traffic count", colormap=cm.get_cmap('viridis'), fig=f, vmin=vmin, vmax=vmax)
    save_tikz(figure_name="Output")
    #plt.title(heading)


def visualize_scenario_from_training_data(args, axis, training_data, heading=None):
    args.time_variant_theta = 0
    args.trip_individual_theta = 0

    weights = np.ones(len(training_data["links_id"]))
    g = Graph(training_data, weights)
    # Visualize network
    g.draw(ax=axis, weights_label=None, cmap=plt.cm.gist_gray)
    # Visualize population
    visualize_work_and_home(ax=axis, work_x=training_data["work_x"], work_y=training_data["work_y"], home_x=training_data["home_x"], home_y=training_data["home_y"])
    save_tikz(figure_name=f"Input_{heading}")
    plt.title(heading)


def visualize_network(network_file, start=None, no_start=None, color="blue", ax=None, alpha=0.1):
    net = matsim.read_network(network_file)
    if no_start is not None:
        net.links = net.links[~net.links['link_id'].astype(str).str.startswith(no_start)]
    if start is not None:
        net.links = net.links[net.links['link_id'].astype(str).str.startswith(start)]
    geo = net.as_geo()
    geo.plot(edgecolor=color, alpha=alpha, ax=ax)


def visualize_work_and_home(ax, work_x, work_y, home_x, home_y):
    ax.scatter(work_x, work_y, c="red", s=3, alpha=1, label="work")
    ax.scatter(home_x, home_y, c="blue", s=3, alpha=1, label="home")


def visualize_population(population_file, ax):
    plan_reader = PlanReader(population_file)
    visualize_work_and_home(ax=ax, work_x=np.array(plan_reader.work_activities_location)[:, 0],
                            work_y=np.array(plan_reader.work_activities_location)[:, 1],
                            home_x=np.array(plan_reader.home_activities_location)[:, 0],
                            home_y=np.array(plan_reader.home_activities_location)[:, 1])


def plot():
    plt.tight_layout()
    plt.show()


def save_tikz(figure_name):
    plt.legend()
    tikzplotlib.save(f"../visualizing/visualization/{figure_name}.tex")


if __name__ == '__main__':
    args = config.parser.parse_args()
    random.seed(0)
    """f, ax = plt.subplots(figsize=(8, 8))
    visualize_network(network_file='../../matsim-berlin/scenarios/sparse_worlds/scenario-1po_0.0pct_0sd/network.xml', no_start="pt_", color="green", ax=ax, alpha=1)
    visualize_population(population_file='../../matsim-berlin/scenarios/sparse_worlds/scenario-1po_0.0pct_0sd/plans.xml', ax=ax)
    plt.show()
    """

    """f, ax = plt.subplots()
    visualize_scenario_from_training_data(args=args, axis=ax, training_data=load_json(f"../data/districtWorldsArt/Validate/po-1_pn-1.0_sn-1/s-12.json"), heading="")
    plot()"""

    for i in range(10, 15):
        f, ax = plt.subplots()
        visualize_scenario_from_training_data(args=args, axis=ax, training_data=load_json(f"../data/districtWorlds/Validate/po-1_pn-1.0_sn-1/s-{i}.json"), heading="")
        plot()

    """for i in range(10, 15):
        f, ax = plt.subplots()
        visualize_solution_from_training_data(args=args, axis=ax, training_data=load_json(f"../data/districtWorldsArt/Validate/po-1_pn-1.0_sn-1/s-{i}.json"),
                                              heading="districtWorldsArt")  # , vmin=0, vmax=12)
        plot()

        f, ax = plt.subplots()
        visualize_scenario_from_training_data(args=args, axis=ax, training_data=load_json(f"../data/districtWorldsArt/Validate/po-1_pn-1.0_sn-1/s-{i}.json"), heading="districtWorldsArt")
        plot()"""

