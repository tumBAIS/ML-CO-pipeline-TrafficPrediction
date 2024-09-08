import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from surrogate.src.util import load_json, Graph
from surrogate.src import config, util
from surrogate.visualizing.learning_results import set_args_list
from surrogate.pipelines.prediction.features import load_training_instances
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest
from surrogate.visualizing.input_data import save_tikz
from surrogate.evaluation import Pipeline
from surrogate.pipelines.solution_representation import Solution
MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}


def print_prediction_over_time(args, time_drive_distribution, name, colorbar=True):
    fig, ax = plt.subplots()
    if colorbar:
        axes = fig.add_axes([1.005, 0.2, 0.05, 0.7])
        sns.heatmap(time_drive_distribution, vmin=0, vmax=6, ax=ax, cbar=colorbar, cbar_ax=axes, cmap="hot_r")  #vmin=0, vmax=4,
    else:
        sns.heatmap(time_drive_distribution, vmin=0, vmax=6, ax=ax, cbar=colorbar, cmap="hot_r")  #vmin=0, vmax=4,
    ax.set_xlabel("Time")
    ax.set_ylabel("Links")
    ax.yaxis.set_label_position("left")
    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.savefig(f".{args.dir_visualization}EVALUATION{name}", bbox_inches='tight')
    plt.show()


def visualize_traffic_over_time(args, data_file, name, colorbar=True):
    data = load_json(f"{data_file}")
    time_drive_distribution_original = np.array(data["original"])
    time_drive_distribution_discretized_y_hat = np.array(data["y_hat"])
    print_prediction_over_time(args, time_drive_distribution_original, name=f"Original", colorbar=False)
    print_prediction_over_time(args, time_drive_distribution_discretized_y_hat, name=f"Prediction{name}", colorbar=colorbar)


def visualize_traffic_over_time_from_instance(args, instance, divisor=1):
    instance_data = load_json(instance)
    pip = Pipeline(args=args, mlLayer=None, coLayer=None)
    time_distribution = pip.get_time_drive_distribution(instance_data, solution_links=instance_data["solution_link"],
                                                        solution_times=instance_data["solution_time"],
                                                        solution_amount=np.ones(len(instance_data["solution_link"])),
                                                        divisor=divisor)
    sns.heatmap(time_distribution)
    plt.xlabel("Time")
    plt.ylabel("Links")
    plt.tight_layout()
    plt.show()


def visualize_traffic(args, instance_file):
    args.time_variant_theta = 0
    args.trip_individual_theta = 0
    args.capacity_individual_theta = 0

    fig, ax = plt.subplots()
    scenario = load_json(instance_file)
    graph = Graph()
    graph.add_nodes_from_instance(scenario)
    solution_representation = Solution(args)
    solution = solution_representation.load_solution_scheme(scenario)
    graph.add_links_from_instance(scenario, weights=solution)
    graph.draw(colormap=True, fig=fig, ax=ax)
    plt.show()


def visualize_predicted_traffic(args_original, args_lists, result_directory):

    def get_correct_data(data, training_instance):
        y = training_instance["y"].values
        for idx, (y_in_data, y_in_data_hat) in enumerate(zip(data["y"], data["y_hat"])):
            if (len(y) == len(y_in_data)) and (y == y_in_data).all():
                return y_in_data, y_in_data_hat
        return False

    results_saver = []
    args_containers = [set_args_list(args_original, args_list) for args_list in args_lists]
    args_containers = [x for xs in args_containers for x in xs]
    for args in args_containers:
        try:
            model = MODEL[args.model](args)
            args.read_in_iteration = util.get_read_in_iteration(args, prefix=".")
            print(f"{util.get_result_file_name(args)} -- iteration: {args.read_in_iteration}")
            data = util.load_json("." + result_directory + util.get_result_file_name(args))
            training_instances = load_training_instances(args, ".")
            training_instances = model.get_solution(args, training_instances)
            results_saver.append((data, training_instances))
            # Iterate over test instances
            for training_instance in [training_instances[2]]:  #2
                data_y, data_y_hat = get_correct_data(data, training_instance)
                for data_i, data_name_i in [(data_y, "data_y"), (data_y_hat, "data_y_hat")]:
                    fig, ax = plt.subplots(layout='constrained', squeeze=False)
                    colorbar = False
                    if data_name_i == "data_y":
                        name = "Visualization_data_y"
                    else:
                        if args.learning == "supervised":
                            name = f"Visualization_{args.learning}_{args.model}_{data_name_i}"
                        else:
                            name = f"Visualization_{args.learning}_{args.co_optimizer}_{args.model}_{data_name_i}"
                            colorbar = True
                    g = Graph(training_instance, data_i)
                    # Visualize network
                    g.draw(weights_label="Traffic count", vmin=0, vmax=13, colormap=colorbar, fig=fig, name=name, args=args) #vmin=0, vmax=18,
                    save_tikz(figure_name=f"{name}")
                    plt.savefig(f".{args.dir_visualization}{name}", bbox_inches='tight')
                    plt.title(name)
                    plt.show()
        except:
            args.read_in_iteration = "X"
            print(F"!!!!!! COULD NOT FIND {util.get_result_file_name(args)}")




if __name__ == '__main__':
    args = config.parser.parse_args()
    #visualize_traffic(instance_file="../data/smallWorlds_pricing/Train/s/s-1.json")
    #visualize_traffic(instance_file="../data/smallWorlds/Train/s/s-1.json")
    #visualize_traffic(args, instance_file="../data/high_entropy_easySimulation/Train/s/s-1.json")
    #visualize_traffic(args, instance_file="../data/low_entropy_easySimulation/Train/s/s-1.json")

    #visualize_traffic_over_time(args=args, data_file="../visualizing/evaluation/EVALUATION_DATA_resultnumDiscreteTimeStepsWith_m-NN_pert-None_st-1_l-supervised_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-39_s-17.json", name="Testsupervised", colorbar=False)
    #visualize_traffic_over_time(args=args, data_file="../visualizing/evaluation/EVALUATION_DATA_resultnumDiscreteTimeStepsWith_m-NN_pert-additive_st-1_l-structured_co-multicommodityflow_nbc-3_cm-inf_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-21_s-17.json", name="TestMCFP")
    #visualize_traffic_over_time(args=args, data_file="../visualizing/evaluation/EVALUATION_DATA_resultnumDiscreteTimeStepsWith_m-NN_pert-additive_nump-0_st-1_l-structured_co-wardropequilibria_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-18_s-17.json", name="TestWE")

    #visualize_traffic_over_time(args=args, data_file="../visualizing/visualization/EVALUATION_DATA_resultnumDiscreteTimeStepsWithTest_m-NN_pert-None_st-1_fe-square_l-supervised_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-42_s-17.json", name="supervised", colorbar=False)
    #visualize_traffic_over_time(args=args, data_file="../visualizing/visualization/EVALUATION_DATA_resultnumDiscreteTimeStepsWithTest_m-NN_pert-additive_st-1_fe-square_l-structured_co-multicommodityflow_nbc-3_cm-inf_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-1_s-17.json", name="MCFP")
    #visualize_traffic_over_time(args=args, data_file="../visualizing/visualization/EVALUATION_DATA_resultnumDiscreteTimeStepsWithTest_m-NN_pert-additive_nump-0_st-1_fe-square_l-structured_co-wardropequilibria_dts-40_tvt-1_tit-0_cit-0_si-squareWorlds_short_s_evp-original_em-timedist_mo-Test_it-18_s-17.json", name="TestWE")


    #visualize_traffic_over_time_from_instance(args=args, instance="../data/cutoutWorlds/Test/po-1_pn-0.1_sn-1/s-15.json", divisor=100)
    #visualize_traffic_over_time_from_instance(args=args, instance="../data/squareWorlds_short/Test/s/s-15.json", divisor=1)



    visualize_predicted_traffic(args_original=args, args_lists=[{"mode": "Test", "models": ["NN"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria"],
                                                                   "simulation": "cutoutWorlds",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [3],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]}],
                         result_directory="./results_final/")

