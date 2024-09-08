import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-1])
sys.path.append(main_directory)
import multiprocessing as mp
from functools import partial
from surrogate.pipelines.prediction.features import load_training_instances
from surrogate.pipelines.structured_learning import optimize as optimize_SL
from surrogate.pipelines.supervised_learning import optimize as optimize_Sup
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest
from surrogate.src.util import create_directories, update_args, get_result_file, save_json, get_result_file_name, get_read_in_iteration
from surrogate.src import config
from surrogate.pipelines.optimizing.multicommodityflow import optimize as multicommodityflow_optimize
from surrogate.pipelines.optimizing.wardropequilibrium import optimize as wardrop_optimize
from surrogate.pipelines.optimizing.nooptimizer import optimize as no_optimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from surrogate.pipelines.optimizing.helpers_multicommodity.get_capacity_expanded_components import load_capacities
from surrogate.pipelines.structured_learning import loss_for_perturbation
from surrogate.pipelines.structured_learning_wardrop import solve_pipeline

LEARNING = {"supervised": optimize_Sup, "structured": optimize_SL}
MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}


def save_result(args, result):
    result_file = get_result_file(args)
    save_json(result, result_file)


def get_co_layer(args):
    if args.learning == "supervised":
        return no_optimize
    else:
        return {"multicommodityflow": multicommodityflow_optimize,
                "wardropequilibria": wardrop_optimize,
                "wardropequilibriaRegularized": wardrop_optimize}[args.co_optimizer]


def get_link_count_y(args, ys, ys_hat, instances):
    y_new = []
    y_hat_new = []
    for instance, y, y_hat in zip(instances, ys, ys_hat):

        y_pandas = pd.DataFrame(y, index=instance["solution_representation"].solution_scheme.index, columns=["target"])
        if args.evaluation_metric == "timedist":
            y_pandas = y_pandas.groupby(["link_id", "time"]).sum()
        else:
            y_pandas = y_pandas.groupby("link_id").sum()
        y_new.append(y_pandas)

        y_hat_pandas = pd.DataFrame(y_hat, index=instance["solution_representation"].solution_scheme.index, columns=["target"])
        if args.evaluation_metric == "timedist":
            y_hat_pandas = y_hat_pandas.groupby(["link_id", "time"]).sum()
        else:
            y_hat_pandas = y_hat_pandas.groupby("link_id").sum()
        y_hat_new.append(y_hat_pandas)

    return y_new, y_hat_new


class Pipeline:
    def __init__(self, args, mlLayer, coLayer):
        self.loss1 = mean_absolute_error #mean_squared_error
        self.loss2 = mean_squared_error
        self.loss3 = r2_score
        self.mlLayer = mlLayer
        self.coLayer = coLayer
        self.args = args

    def predict(self, instance):
        thetas = self.mlLayer.predict([instance])
        y_hat = self.coLayer(thetas, instance)
        return thetas, y_hat

    def evaluate(self, instances):
        if self.args.evaluation_procedure == "original":
            theta_hat, y_hat, y, y_compressed, y_hat_compressed = self.evaluate_original(instances=instances)
        elif self.args.evaluation_procedure == "perturbed":
            theta_hat, y_hat, y, y_compressed, y_hat_compressed = self.evaluate_perturbed(instances=instances)
        else:
            raise Exception("Wrong evaluation procedure defined")
        if self.args.mode == "Test":
            return {self.loss1.__name__: [self.loss1(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    self.loss2.__name__: [self.loss2(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    self.loss3.__name__: [self.loss3(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    "time_dist": [self.time_dist(y_i.reset_index(), y_hat_i.reset_index(), instance) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    "time_dist_discretized": [self.time_dist_discretized(y_i.reset_index(), y_hat_i.reset_index(), instance) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    "theta_hat": theta_hat, "y_hat": y_hat, "y": y}
        else:
            return {self.loss1.__name__: [self.loss1(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    self.loss2.__name__: [self.loss2(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    self.loss3.__name__: [self.loss3(list(y_i.target), list(y_hat_i.target)) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    "time_dist": [self.time_dist(y_i.reset_index(), y_hat_i.reset_index(), instance) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)],
                    "time_dist_discretized": [self.time_dist_discretized(y_i.reset_index(), y_hat_i.reset_index(), instance) for y_i, y_hat_i, instance in zip(y_compressed, y_hat_compressed, instances)]}

    def evaluate_perturbed(self, instances):
        theta_hat = []
        y_hat = []
        for training_instance in instances:
            theta = self.mlLayer.predict([training_instance])
            with mp.Pool(args.num_cpus - 1) as pool:
                saver = pool.map(partial(loss_for_perturbation, (args, training_instance, theta, self.coLayer, None)), list(range(args.num_perturbations)))
                y_hat_perturbed_mean = np.mean([sav[0] for sav in saver], axis=0)
            theta_hat.append(theta)
            y_hat.append(y_hat_perturbed_mean)
        _, y = self.mlLayer.batch_dataset(instances)
        y = np.split(y, np.cumsum([len(y_hat_i) for y_hat_i in y_hat]))[:-1]
        y_compressed, y_hat_compressed = get_link_count_y(self.args, y, y_hat, instances)
        return theta_hat, y_hat, y, y_compressed, y_hat_compressed

    def evaluate_original(self, instances):
        mp.set_start_method('spawn')
        with mp.Pool(args.num_cpus - 1) as pool:
            saver = pool.map(partial(solve_pipeline, (args, self.coLayer, args.read_in_iteration)), instances)
        theta_hat = [x[2] for x in saver]
        y_hat = [x[1] for x in saver]
        _, y = self.mlLayer.batch_dataset(instances)
        y = np.split(y, np.cumsum([len(y_hat_i) for y_hat_i in y_hat]))[:-1]
        y_compressed, y_hat_compressed = get_link_count_y(self.args, y, y_hat, instances)
        return theta_hat, y_hat, y, y_compressed, y_hat_compressed

    def get_time_drive_distribution(self, instance, solution_links, solution_times, solution_amount, divisor=1):
        edge_flow_times = np.array(instance["link_length"]) / np.array(instance["link_freespeed"])
        time_drive_distribution = np.zeros((len(instance["links_id"]), int((instance["maximum_end"] - instance["minimum_start"])/divisor + divisor)))
        for enter_link, enter_time, enter_amount in zip(solution_links, solution_times, solution_amount):
            if enter_link < 0:
                continue
            time_drive_distribution[int(enter_link), int((enter_time - instance["minimum_start"])/divisor): int((enter_time - instance["minimum_start"] + edge_flow_times[int(enter_link)])/divisor)+1] += enter_amount
        return time_drive_distribution

    def time_dist_discretized(self, y, y_hat, instance):
        if self.args.evaluation_metric == "timedist":
            time_drive_distribution_discretized_y = self.get_time_drive_distribution(instance, solution_links=y["link_id"], solution_times=y["time"], solution_amount=y["target"])
            time_drive_distribution_discretized_y_hat = self.get_time_drive_distribution(instance, solution_links=y_hat["link_id"], solution_times=y_hat["time"], solution_amount=y_hat["target"])
            return mean_absolute_error(time_drive_distribution_discretized_y, time_drive_distribution_discretized_y_hat)
        else:
            return 0

    def time_dist(self, y, y_hat, instance):
        if self.args.evaluation_metric == "timedist":
            time_drive_distribution_original = self.get_time_drive_distribution(instance, solution_links=instance["solution_link"], solution_times=instance["solution_time"], solution_amount=np.ones(len(instance["solution_link"])))
            time_drive_distribution_discretized_y_hat = self.get_time_drive_distribution(instance, solution_links=y_hat["link_id"], solution_times=y_hat["time"], solution_amount=y_hat["target"])
            if self.args.mode == "Test": #or self.args.mode == "Validate":
                save_json({"original": time_drive_distribution_original, "y_hat": time_drive_distribution_discretized_y_hat},
                          f"{self.args.dir_visualization}EVALUATION_DATA_{get_result_file_name(args)}_{instance['training_instance_name']}")
                print_prediction_over_time(args, time_drive_distribution_original, time_drive_distribution_discretized_y_hat)
            return mean_absolute_error(time_drive_distribution_original, time_drive_distribution_discretized_y_hat)
        else:
            return 0


def print_prediction_over_time(args, time_drive_distribution_original, time_drive_distribution_discretized_y_hat, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(time_drive_distribution_original, ax=axes[0])
    axes[0].set_title('Original')
    sns.heatmap(time_drive_distribution_discretized_y_hat, ax=axes[1])
    axes[1].set_title('y_hat')
    sns.heatmap(time_drive_distribution_discretized_y_hat - time_drive_distribution_original, ax=axes[2])
    axes[2].set_title('Difference')
    plt.xlabel("Time")
    plt.ylabel("Links")
    plt.tight_layout()
    plt.savefig(f"{prefix}{args.dir_visualization}EVALUATION{get_result_file_name(args)}.png")
    plt.show()


if __name__ == "__main__":
    args = config.parser.parse_args()
    if args.mode not in ["Test", "Validate"]:
        raise Exception("Invalid mode. Mode in 'Test / Validate' ")
    args = update_args(args)
    if args.mode == "Test":
        args.read_in_iteration = get_read_in_iteration(args)
    print(args)
    create_directories([args.dir_results])

    # set seeds
    random.seed(args.seed_algorithm)
    np.random.seed(args.seed_algorithm)
    tf.random.set_seed(args.seed_algorithm)

    print("Create model...")
    model = MODEL[args.model](args)

    print("Load training instances...")
    training_instances = load_training_instances(args)

    print("Calculate solution for training instances...")
    training_instances = model.get_solution(args, training_instances)

    print("Calculate features for training instances...")
    training_instances = model.get_features(args, training_instances)

    print("Load model architecture and weights...")
    model.load_model(X=training_instances[0]["X"])

    print("Standardize features of instances...")
    training_instances = model.standardize(training_instances)

    print("Load capacities")
    training_instances = load_capacities(args, training_instances)

    print("Get CO-layer...")
    coLayer = get_co_layer(args)

    print("Create pipeline...")
    pipeline = Pipeline(args=args, mlLayer=model, coLayer=coLayer)

    print("Evaluate model...")
    result = pipeline.evaluate(training_instances)

    print("Save result...")
    save_result(args, result)
