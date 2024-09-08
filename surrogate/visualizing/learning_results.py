import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
import collections
from surrogate.src import util
from surrogate.src import config
import seaborn as sns
import copy


class Experiment:
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        self.simulation = simulation_to_name(simulation, simulator, time_variant_theta)
        self.y_ticks = y_ticks
        self.get_best_categories = [("-Base", "-structured_co", "multicommodity"),  #("-Base", "-supervised_"),
                                    ("-NN_", "-supervised_"), ("-GNN_", "-supervised_"),
                                    ("-NN", "-structured_co", "multicommodity"), #("-GNN", "-structured_co", "multicommodity"),
                                    ("-NN", "-structured_co", "wardropequilibria"), ("-NN", "-structured_wardrop", "wardropequilibria"),
                                    ("-NN", "-structured_co", "wardropequilibriaRegularized")]#,
                                    #("-GNN", "-supervised_"), ("-GNN", "-structured_")]#,
                                    #("-RandomForest", "-supervised_")]

    def get_x_labels(self, args):
        ...

    def get_x_axis_name(self):
        ...


class DefaultExperiment(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'Benchmarks-{self.simulation}'

    def get_x_labels(self, args):
        return ""

    def get_x_axis_name(self):
        return ""


class WardropExperiment(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'Wardrop-{self.simulation}'

    def get_x_labels(self, args):
        return args.co_optimizer


class StructuredWardropExperiment(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'StructuredWardrop-{self.simulation}'

    def get_x_labels(self, args):
        return f"{args.co_optimizer}_{args.learning}"


class Benchmarks(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'Benchmarks-{self.simulation}'

    def get_x_labels(self, args):
        if self.y_ticks is None:
            return f"{util.get_model_file_name_specification(args)}_{util.get_evaluation_procedure(args)}"
        else:
            x_label = ""
            for y_tick in self.y_ticks:
                if y_tick == "trip_individual_thetas":
                    x_label += util.get_trip_individual_theta(args) + "_"
                elif y_tick == "learnings":
                    x_label += util.get_learning_name(args) + "_"
                elif y_tick == "capacity_individual_thetas":
                    x_label += util.get_capacity_individual_theta(args) + "_"
                elif y_tick == "time_variant_thetas":
                    x_label += util.get_time_variant_theta(args) + "_"
                else:
                    raise Exception("x_tick not known.")
            return x_label

    def get_x_axis_name(self):
        return "Model specification"


class PerturbedEvaluation(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'PerturbedEvaluation-{self.simulation}'

    def get_x_labels(self, args):
        if args.time_variant_theta and args.capacity_individual_theta:
            return f"time_capacity_sde-{args.sd_perturbation_evaluation}"
        elif args.time_variant_theta:
            return f"time_sde-{args.sd_perturbation_evaluation}"
        elif args.trip_individual_theta:
            return f"trip_sde-{args.sd_perturbation_evaluation}"
        elif args.capacity_individual_theta:
            return f"capacity_sde-{args.sd_perturbation_evaluation}"
        else:
            return f"sde-{args.sd_perturbation_evaluation}"

    def get_x_axis_name(self):
        return "Model specification"


class NumDiscreteTimeSteps(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'NumDiscreteTimeSteps-{self.simulation}'

    def get_x_labels(self, args):
        return f"{util.get_model_file_name_specification(args)}"

    def get_x_axis_name(self):
        return "Num. discrete time steps"


class SupervisedCapacities(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'supervisedCapacities-{self.simulation}'

    def get_x_labels(self, args):
        return args.capacity_multiplicator

    def get_x_axis_name(self):
        return "% Increase from sup. pred."


class DiffCapacities(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f"DiffCapacities-{self.simulation}"
        self.get_best_categories = [("NN", "-structured_")]

    def get_x_labels(self, args):
        return args.capacity_multiplicator

    def get_x_axis_name(self):
        return "Percentage of capacity increase in \\\\ comparison from true traffic count"


class NumBundledCapacities(Experiment):
    def __init__(self, simulation, simulator, time_variant_theta, y_ticks):
        super().__init__(simulation=simulation, simulator=simulator, time_variant_theta=time_variant_theta, y_ticks=y_ticks)
        self.name = f'NumBundledCapacities-{self.simulation}'

    def get_x_labels(self, args):
        return args.num_bundled_capacities

    def get_x_axis_name(self):
        return "Max capacity per arc"


experiments = {"numBundledCapacities": NumBundledCapacities, "ownsimulation": Benchmarks, "perturbedEvaluation": PerturbedEvaluation,
               "diffCapacities": DiffCapacities,
               "numDiscreteTimeSteps": NumDiscreteTimeSteps, "numDiscreteTimeStepsWith": NumDiscreteTimeSteps, "numDiscreteTimeStepsUncapacitated": NumDiscreteTimeSteps,
               "benchmarks": Benchmarks, "benchmarksSecond": Benchmarks, "": DefaultExperiment,
               "supervisedCapacities": SupervisedCapacities,
               "numBundledCapacities_capacitated": NumBundledCapacities,
               "wardrop": WardropExperiment, "structuredwardrop": StructuredWardropExperiment, "testcutout": Benchmarks,
               "withoutPerturbation": Benchmarks, "entropy": Benchmarks, "numDiscreteTimeStepsWithTest": NumDiscreteTimeSteps}


class ArgsContainer:
    def __init__(self, args_original, mode, model, learning, simulation, perturbation, optimizer, time_variant_theta,
                 trip_individual_theta, capacity_individual_theta, num_discrete_time_steps, experiment_name, num_bundled_capacities,
                 capacity_multiplicator, simulator, percentage_original, percentage_new, seed_new, evaluation_procedure, sd_perturbation_evaluation,
                 evaluation_metric, num_perturbations, feature):
        self.mode = mode
        self.model = model
        self.learning = learning
        self.simulation = simulation
        self.perturbation = perturbation
        self.co_optimizer = optimizer
        self.time_variant_theta = time_variant_theta
        self.trip_individual_theta = trip_individual_theta
        self.capacity_individual_theta = capacity_individual_theta
        self.num_discrete_time_steps = num_discrete_time_steps
        self.dir_best_iteration_saver = args_original.dir_best_iteration_saver
        self.standardize = args_original.standardize
        self.dir_instances = args_original.dir_instances
        self.learning_rate = args_original.learning_rate
        self.seed = args_original.seed
        self.dir_visualization = args_original.dir_visualization
        self.experiment_name = experiment_name
        self.num_bundled_capacities = num_bundled_capacities
        self.capacity_multiplicator = capacity_multiplicator
        self.simulator = simulator
        self.percentage_original = percentage_original
        self.percentage_new = percentage_new
        self.seed_new = seed_new
        self.evaluation_procedure = evaluation_procedure
        self.sd_perturbation_evaluation = sd_perturbation_evaluation
        self.evaluation_metric = evaluation_metric
        self.num_perturbations = num_perturbations
        self.feature = feature


def set_args_list(args_original, args_list, write_table=False):
    args_containers = []
    for learning in args_list["learnings"]:
        for model in args_list["models"]:
            for perturbation in args_list["perturbations"]:
                for time_variant_theta in args_list["time_variant_thetas"]:
                    for trip_individual_theta in args_list["trip_individual_thetas"]:
                        for capacity_individual_theta in args_list["capacity_individual_thetas"]:
                            for optimizer in args_list["optimizers"]:
                                for num_discrete_time_steps in args_list["num_discrete_time_steps"]:
                                    for num_bundled_capacities in args_list["num_bundled_capacities"]:
                                        for capacity_multiplicator in args_list["capacity_multiplicator"]:
                                            for evaluation_procedure in args_list["evaluation_procedure"]:
                                                for sd_perturbation_evaluation in args_list["sd_perturbation_evaluation"]:
                                                    for num_perturbations in args_list["num_perturbations"]:
                                                        for feature in args_list["features"]:

                                                            #if args_list["simulation"] == "cutoutWorlds" and capacity_individual_theta == 1 and learning in ["structured", "supervised"]:
                                                            #    num_bundled_capacities = 15

                                                            args = ArgsContainer(args_original=args_original,
                                                                                 mode=args_list["mode"],
                                                                                 model=model,
                                                                                 learning=learning,
                                                                                 simulation=args_list["simulation"],
                                                                                 perturbation=perturbation,
                                                                                 optimizer=optimizer,
                                                                                 time_variant_theta=time_variant_theta,
                                                                                 trip_individual_theta=trip_individual_theta,
                                                                                 capacity_individual_theta=capacity_individual_theta,
                                                                                 num_discrete_time_steps=num_discrete_time_steps,
                                                                                 experiment_name=args_list["experiment_name"],
                                                                                 num_bundled_capacities=num_bundled_capacities,
                                                                                 capacity_multiplicator=capacity_multiplicator,
                                                                                 simulator=args_list["simulator"],
                                                                                 percentage_original=args_list["percentage_original"],
                                                                                 percentage_new=args_list["percentage_new"],
                                                                                 seed_new=args_list["seed_new"],
                                                                                 evaluation_procedure=evaluation_procedure,
                                                                                 sd_perturbation_evaluation=sd_perturbation_evaluation,
                                                                                 evaluation_metric=args_list["evaluation_metric"],
                                                                                 feature=feature,
                                                                                 num_perturbations=0 if ((optimizer in ["wardropequilibriaRegularized"]) or (optimizer == "wardropequilibria" and time_variant_theta == 1)) else num_perturbations)

                                                            # Combinations to ignore
                                                            useful = util.test_run_on_usefulness(args, write_table=write_table)
                                                            if useful:
                                                                args_containers.append(args)
    return args_containers


def visualize_learning_evolution(args_original, args_lists, learning_directory):
    args_containers = [set_args_list(args_original, args_list, write_table=False) for args_list in args_lists]
    args_containers = [x for xs in args_containers for x in xs]
    for args in args_containers:
        try:
            learning_evolution = util.load_json(learning_directory + util.get_model_file_name(args))
            generate_plot_learning_evolution(args_original, args, learning_evolution, prefix=".")
            tikzplotlib.save(f".{args_original.dir_visualization}LEARNING{util.get_model_file_name(args)}.tex", extra_axis_parameters=["ylabel style={align=center}"])
            plt.show()
        except Exception as e:
            print(e)


def generate_plot_learning_evolution(args_original, args, learning_evolution, suffix="", prefix=""):
    plt.clf()
    learning_evolution = pd.DataFrame(learning_evolution)
    loss_evolution, mse_evolution, mae_evolution = learning_evolution["loss"].values, learning_evolution["mean_squared_error"].values, learning_evolution["mean_absolute_error"].values
    fig, ax1 = plt.subplots()
    # LOSS AXIS
    color = 'tab:red'
    ax1.set_xlabel('Training epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_evolution, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # MSE AXIS
    #ax2 = ax1.twinx()
    #color = 'tab:blue'
    #ax2.set_ylabel('MSE', color=color)
    #ax2.plot(mse_evolution, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)
    # MAE AXIS
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel('MAE', color=color)
    ax3.plot(mae_evolution, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.title("Learning evolution \n " + util.get_model_file_name(args), fontsize=10)
    fig.tight_layout()
    plt.savefig(f"{prefix}{args_original.dir_visualization}LEARNING{util.get_model_file_name(args)}{suffix}.png")


def simulation_to_name(simulation_name, simulator, time_variant_theta):
    name = ""
    if simulation_name == "squareWorlds_short":
        name = "squareWorldShort"
    if simulation_name == "low_entropy":
        name = "LEU"
    if simulation_name == "high_entropy":
        name = "HEU"
    elif simulation_name == "low_entropy_street":
        name = "LE"
    elif simulation_name == "high_entropy_street":
        name = "HE"
    elif simulation_name == "cutoutWorlds":
        name = "cutoutWorlds"
    elif simulation_name == "smallWorlds_short":
        name = "smallWorldShort"
    elif simulation_name == "squareWorlds_short_capacitated":
        name = "squareWorldShortCapacitated"
    elif simulation_name == "againstsupervisedWorld":
        name = "againstsupervisedWorld"
    elif simulation_name == "cutoutWorlds_mixed":
        name = "cutoutWorldsMixed"
    elif simulation_name == "cutoutWorldsCap":
        name = "cutoutWorldsCap"
    elif simulation_name == "cutoutWorldsSpeed":
        name = "cutoutWorldsSpeed"
    elif simulation_name == "cutoutWorldsWithoutBoundary":
        name = "cutoutWorldsWithoutBoundary"
    elif simulation_name == "districtWorlds":
        name = "districtWorlds"
    elif simulation_name == "districtWorldsArt":
        name = "districtWorldsArt"
    if simulator == "againstsupervisedSimulation":
        name += "randomMCFP"
    elif simulator == "easySimulation":
        name += "easyMCFP"
    elif simulator == "easySimulationEquilibrium":
        name += "easyWE"
    elif simulator == "randomSimulationEquilibrium":
        name += "randomWE"
    if time_variant_theta:
        name += "timeDep."
    if name == "":
        raise Exception("Experiment name is missing")

    return name


def get_best(results_saver, areas, metric, write_table=False):
    new_result_saver = {}
    for area in areas:
        area_result_saver = {}
        for key_results_saver in results_saver.keys():
            if all([area_item in key_results_saver for area_item in area]):
                area_result_saver[key_results_saver] = results_saver[key_results_saver]
        if metric == "r2_score":
            new_result_saver[area] = max(area_result_saver.items(), key=lambda x: np.mean(x[1]))[1]
            if write_table is False:
                print(f"BEST RESULT MODEL: {max(area_result_saver.items(), key=lambda x: np.mean(x[1]))[0]}")
        else:
            if len(area_result_saver) > 0:
                new_result_saver[area] = min(area_result_saver.items(), key=lambda x: np.mean(x[1]))[1]
                if write_table is False:
                    print(f"BEST RESULT MODEL: {min(area_result_saver.items(), key=lambda x: np.mean(x[1]))[0]}")
    return new_result_saver


def rename(results_saver):
    keys = copy.deepcopy(list(results_saver.keys()))
    for key_old in keys:
        if key_old == ("-Base", "-supervised_"):
            name = "ML.B"
        elif key_old == ("-Base", "-structured_co", "multicommodity"):
            name = "Base"
        elif key_old == ("-NN_", "-supervised_"):
            name = "FNN"
        elif key_old == ("-GNN_", "-supervised_"):
            name = "GNN"
        elif key_old == ("-NN", "-structured_co", "multicommodity"):
            name = "MCFP"
        elif key_old == ("-GNN", "-structured_co", "multicommodity"):
            name = "MCFP-GNN"
        elif key_old == ("-NN", "-structured_co", "wardropequilibria"):
            name = "WE"
        elif key_old == ("-NN", "-structured_wardrop", "wardropequilibria") or key_old == ("-NN", "-structured_co", "wardropequilibriaRegularized"):
            name = "WE-reg"
        results_saver[name] = results_saver.pop(key_old)

    return results_saver


def get_table_name(original_name):
    new_name = []
    if "cutoutWorlds" in original_name:
        new_name.append("RealWorld")
    elif "squareWorldShort" in original_name:
        new_name.append("SW")
    elif "LEU" in original_name:
        new_name.append("LEU")
    elif "HEU" in original_name:
        new_name.append("HEU")
    elif "LE" in original_name:
        new_name.append("LE")
    elif "HE" in original_name:
        new_name.append("HE")
    if "randomMCFP" in original_name:
        new_name.append("randomMCFP")
    elif "easyMCFP" in original_name:
        new_name.append("easyMCFP")
    elif "easyWE" in original_name:
        new_name.append("easyWE")
    elif "randomWE" in original_name:
        new_name.append("randomWE")
    else:
        new_name.append("MATSim")
    if "timeDep." in original_name:
        new_name.append("timeDep.")
    return "-".join(new_name)


def compare_performances(args_original, args_lists, result_directory, metric, name_suffix="", y_ticks=None, xlim=None, best=False, add_result=None, write_table=False):
    results_saver = {}
    args_containers = [set_args_list(args_original, args_list, write_table=write_table) for args_list in args_lists]
    args_containers = [x for xs in args_containers for x in xs]
    experiment = experiments[args_lists[0]["experiment_name"]](args_lists[0]["simulation"], args_lists[0]["simulator"], args_lists[0]["time_variant_thetas"][0], y_ticks)
    for args in args_containers:
        try:
            args.read_in_iteration = util.get_read_in_iteration(args, prefix=".")
            if write_table is False:
                print(f"{util.get_result_file_name(args)} -- iteration: {args.read_in_iteration}")
            data = util.load_json("." + result_directory + util.get_result_file_name(args))

            if f"{experiment.get_x_labels(args)}" in results_saver.keys():
                if collections.Counter(results_saver[f"{experiment.get_x_labels(args)}"]) != collections.Counter(data[metric]):
                    raise Exception("X label already exists with different values")
            results_saver[f"{experiment.get_x_labels(args)}"] = data[metric]
        except:
            args.read_in_iteration = "X"
            if write_table is False:
                print(F"!!!!!! COULD NOT FIND {util.get_result_file_name(args)}")

    if add_result is not None:
        data = util.load_json("." + result_directory + add_result)
        results_saver[f"GNN_MCFP"] = data[metric]

    best_model, best_model_result_list = min(results_saver.items(), key=lambda x: np.mean(x[1]))
    best_model_median, best_model_result_list_median = min(results_saver.items(), key=lambda x: np.median(x[1]))
    if write_table is False:
        print(f"The best model: {best_model} / value MEAN: {np.mean(best_model_result_list)}")
        print(f"The best model: {best_model_median} / value MEDIAN: {np.median(best_model_result_list_median)}")

    if best:
        results_saver = get_best(results_saver, experiment.get_best_categories, metric, write_table=write_table)
        results_saver = rename(results_saver)

    if write_table:
        print(f"{get_table_name(experiment.simulation)} & mean & {round(np.mean(results_saver['Base']), 3)} & "
              f"{round(np.mean(results_saver['ML']), 3)} & {round(np.mean(results_saver['MCFP']), 3)} & {round(np.mean(results_saver['WE']), 3)} & "
              f"{round(np.mean(results_saver['WE-reg']), 3)}" +
              f" & 25\% perc. & {round(np.percentile(results_saver['Base'], 25), 3)} & "
              f"{round(np.percentile(results_saver['ML'], 25), 3)} & {round(np.percentile(results_saver['MCFP'], 25), 3)} & {round(np.percentile(results_saver['WE'], 25), 3)} & "
              f"{round(np.percentile(results_saver['WE-reg'], 25), 3)} \\\\")
        print(f" & max & {round(np.max(results_saver['Base']), 3)} & "
              f"{round(np.max(results_saver['ML']), 3)} & {round(np.max(results_saver['MCFP']), 3)} & {round(np.max(results_saver['WE']), 3)} & "
              f"{round(np.max(results_saver['WE-reg']), 3)}" +
              f" & 75\% perc. & {round(np.percentile(results_saver['Base'], 75), 3)} & "
              f"{round(np.percentile(results_saver['ML'], 75), 3)} & {round(np.percentile(results_saver['MCFP'], 75), 3)} & {round(np.percentile(results_saver['WE'], 75), 3)} & "
              f"{round(np.percentile(results_saver['WE-reg'], 75), 3)} \\\\")
        print(f" & min & {round(np.min(results_saver['Base']), 3)} & "
              f"{round(np.min(results_saver['ML']), 3)} & {round(np.min(results_saver['MCFP']), 3)} & {round(np.min(results_saver['WE']), 3)} & "
              f"{round(np.min(results_saver['WE-reg']), 3)}" +
              f" & & & & & & \\\\")
    else:
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(results_saver)
        if False:
            if args_lists[0]["simulation"] in ["cutoutWorlds", "cutoutWorlds_mixed"]:
                plt.axvline(x=0.3230570816232625, label="best found median")
            if args_lists[0]["simulation"] == "squareWorlds_short":
                plt.axvline(x=2.246361301369863, label="best found median")
            if args_lists[0]["simulation"] == "squareWorlds_short_capacitated":
                plt.axvline(x=1.7799543951566403, label="best found median")
            if args_lists[0]["simulation"] == "smallWorlds_short":
                plt.axvline(x=1.3643790849673203, label="best found median")
        #plt.axvline(x=2.246361301369863, label="best found median")
        #plt.xlabel(metric)

        #plt.xlabel(metric)
        #plt.xlim(xlim)
        #plt.ylabel(experiment.get_x_axis_name())
        #plt.xticks(ticks=range(1, 1 + len(results_saver)), labels=results_saver.keys(), fontsize="small")

        plt.xticks(range(len(results_saver)), results_saver.keys())
        #plt.xlim(xlim)
        if metric == "mean_absolute_error":
            plt.ylabel("mean absolute error")
        else:
            raise Exception
        plt.yscale("log")
        plt.ylim((0.01, 26))

        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        tikzplotlib.save(f"./visualization/compare_performances_{experiment.name}{name_suffix}-{metric}" + ("_best" if best else "") + ".tex", extra_axis_parameters=["ylabel style={align=center}"])
        plt.title("Results " + simulation_to_name(args_lists[0]["simulation"], args_lists[0]["simulator"], args_lists[0]["time_variant_thetas"][0]))
        plt.tight_layout()

        plt.show()

        results_saver_sorted = sorted(results_saver.items(), key=lambda x: np.mean(x[1]))
        for model, result in results_saver_sorted:
            print(f"MODEL: {model} | {metric}: {np.mean(result)}")


def print_best_validation_iterations(args):
    for best_iteration_saver_file in os.listdir(f".{args.dir_best_iteration_saver}"):
        f = open(f".{args.dir_best_iteration_saver}{best_iteration_saver_file}", "r")
        read_in_iteration = int(f.read())
        if "structured" in best_iteration_saver_file:
            print(f"{best_iteration_saver_file} ====> {read_in_iteration}")


def get_mean(file_name):
    data = util.load_json(file_name)
    print(file_name)
    print(f"Mean: {np.mean(data['mean_absolute_error'])}")
    print(f"Median: {np.median(data['mean_absolute_error'])}")


def get_output_table():




    print("\\begin{table}[h]")
    print("\\caption{Benchmark performances on scenarios}")
    print("\\label{tab:appendix:benchmarkperformances}")
    print("\\tiny")
    print("\\centering")
    print("\\setlength\\tabcolsep{2pt}")
    print("\\begin{tabular}{l l c c c c c l c c c c c}")
    print("\\toprule")
    print("Scenario &  & Base & ML & MCFP & WE & WE-reg &  & Base & ML & MCFP & WE & WE-reg\\\\")
    print("\\midrule")
    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN", "Base"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "cutoutWorlds",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [15],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                  ],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True, write_table=True)
    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN", "Base"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "squareWorlds_short",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [3],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                  ],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True, write_table=True)
    for simulation in ["squareWorlds_short", "low_entropy", "high_entropy", "low_entropy_street", "high_entropy_street"]:
        for simulator in ["easySimulation", "easySimulationEquilibrium", "againstsupervisedSimulation", "randomSimulationEquilibrium"]:
            compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN", "Base"], "learnings": ["structured", "supervised"],
                                                                           "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                           "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                           "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                           "simulation": str(simulation),
                                                                           "num_discrete_time_steps": [20], "num_bundled_capacities": [3],
                                                                           "experiment_name": "entropy", "simulator": str(simulator),
                                                                           "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                           "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                           "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                          ],
                                 result_directory="./results_paper/", metric="mean_absolute_error", best=True, write_table=True)
    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN", "Base"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [1], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "squareWorlds_short",
                                                                   "num_discrete_time_steps": [40], "num_bundled_capacities": [3],
                                                                   "experiment_name": "numDiscreteTimeStepsWithTest", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "timedist", "num_perturbations": [50], "features": ["square"]}],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True, write_table=True)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    args_original = config.parser.parse_args()

    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "squareWorlds_short",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [3],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                  ],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True)


    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "cutoutWorlds",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [15],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                  ],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True)


    compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN"], "learnings": ["structured", "supervised"],
                                                                   "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                   "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                   "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                   "simulation": "districtWorldsArt",
                                                                   "num_discrete_time_steps": [20], "num_bundled_capacities": [10],
                                                                   "experiment_name": "benchmarks", "simulator": "matsim",
                                                                   "percentage_original": 1, "percentage_new": 1.0, "seed_new": 1,
                                                                   "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                   "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                  ],
                         result_directory="./results_paper/", metric="mean_absolute_error", best=True)



    for simulation in ["squareWorlds_short", "low_entropy", "high_entropy", "low_entropy_street", "high_entropy_street"]:
        for simulator in ["easySimulation", "easySimulationEquilibrium", "againstsupervisedSimulation", "randomSimulationEquilibrium"]:
            compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN"], "learnings": ["structured", "supervised"],
                                                                           "perturbations": ["additive"], "time_variant_thetas": [0], "trip_individual_thetas": [0],
                                                                           "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                           "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                           "simulation": str(simulation),
                                                                           "num_discrete_time_steps": [20], "num_bundled_capacities": [3],
                                                                           "experiment_name": "entropy", "simulator": str(simulator),
                                                                           "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                           "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                           "evaluation_metric": "original", "num_perturbations": [50], "features": [""]},
                                                                          ],
                                 result_directory="./results_paper/", metric="mean_absolute_error", best=True) #results_20240511

    for simulation in ["squareWorlds_short"]:
        for features in [["square"]]:
            compare_performances(args_original=args_original, args_lists=[{"mode": "Test", "models": ["NN", "GNN"], "learnings": ["structured", "supervised"],
                                                                           "perturbations": ["additive"], "time_variant_thetas": [1], "trip_individual_thetas": [0],
                                                                           "capacity_individual_thetas": [0, 1], "capacity_multiplicator": ["inf"],
                                                                           "optimizers": ["multicommodityflow", "wardropequilibria", "wardropequilibriaRegularized"],
                                                                           "simulation": str(simulation),
                                                                           "num_discrete_time_steps": [40], "num_bundled_capacities": [3],
                                                                           "experiment_name": "numDiscreteTimeStepsWithTest", "simulator": "matsim",
                                                                           "percentage_original": 1, "percentage_new": 0.1, "seed_new": 1,
                                                                           "evaluation_procedure": ["original"], "sd_perturbation_evaluation": [1.0],
                                                                           "evaluation_metric": "timedist", "num_perturbations": [50], "features": features}],
                                 result_directory="./results_paper/", metric="mean_absolute_error", best=True)


