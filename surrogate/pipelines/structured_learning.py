import time
import math
from surrogate.src import util
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from surrogate.visualizing.learning_results import generate_plot_learning_evolution
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest
from surrogate.src.util import Graph

MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}


def save_learning_evaluation(args, learning_evolution):
    util.save_json(learning_evolution, util.get_learning_file(args))


def optimize(args, model, training_instances, co_optimizer, suffix=""):

    overall_loss = []

    time_start_training = time.time()
    for epoch in range(args.num_training_epochs):
        print(f"EPOCH: {epoch}")
        mse_values = []
        mae_values = []
        loss_values = []
        for training_instance in training_instances:
            y, y_hat, predicted_objective, correct_objective, scaled_y_hat_perturbed_mean, scaled_y_hat_perturbed_squared_mean, perturbation_mean, theta = solve_perturbed_pipeline(args, model, training_instance, co_optimizer, epoch)
            mse_value, mae_value, loss_value = get_loss(y, y_hat, predicted_objective, correct_objective)
            if not any(math.isnan(predicted_objective_value) for predicted_objective_value in predicted_objective):
                model.sl_optimize(y, scaled_y_hat_perturbed_mean, scaled_y_hat_perturbed_squared_mean, [input.values for input in training_instance["X"]], perturbation_mean, theta)
                mse_values.append(mse_value)
                mae_values.append(mae_value)
                loss_values.append(loss_value)
        overall_loss.append({"loss": np.mean(loss_values), "mean_squared_error": np.mean(mse_values), "mean_absolute_error": np.mean(mae_values)})
        print("Epoch: {} ----> Loss: {} | MSE: {} | MAE: {}".format(epoch, np.mean(loss_values), np.mean(mse_values), np.mean(mae_value)))
        model.save_model(epoch)
        save_learning_evaluation(args, overall_loss)
        generate_plot_learning_evolution(args, args, overall_loss, suffix=suffix)
        if time.time() - time_start_training > args.max_training_time.total_seconds():
            print("STOP LEARNING DUE TO TIME LIMIT")
            break


def loss_for_perturbation(kwargs, perturbation_num):

    np.random.seed(perturbation_num)

    def apply_additive_perturbation(vector):
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        perturbation = np.random.normal(loc=0, scale=args.sd_perturbation, size=vector.shape)
        vector_perturbed = vector + perturbation
        return vector_perturbed, perturbation

    def apply_multiplicative_perturbation(vector):
        perturbation = np.exp(args.sd_perturbation * np.random.normal(loc=0, scale=1, size=vector.shape) - ((args.sd_perturbation**2) * (1/2)))
        vector_perturbed = vector * perturbation
        return vector_perturbed, perturbation

    args, training_instance, theta_hat, co_optimizer, latency_function_package = kwargs

    latency_function = 0

    if args.perturbation == 'additive':
        theta_hat_perturbed, perturbation = apply_additive_perturbation(theta_hat)
        y_hat_perturbed = co_optimizer(args=args, thetas=theta_hat_perturbed, instance=training_instance, latency_function=latency_function,
                                       verbose=args.verbose)
        if y_hat_perturbed is np.nan:
            return np.array([np.nan] * len(theta_hat)), np.array([np.nan] * len(theta_hat)), np.array([np.nan] * len(perturbation))
        if args.co_optimizer == "wardropequilibria":
            predicted_objective = theta_hat_perturbed[0] * y_hat_perturbed + (1/2) * theta_hat_perturbed[1] * y_hat_perturbed**2
        else:
            predicted_objective = theta_hat_perturbed * y_hat_perturbed
        return y_hat_perturbed, predicted_objective, perturbation
    elif args.perturbation == "multiplicative":
        theta_hat_perturbed, perturbation = apply_multiplicative_perturbation(theta_hat)
        y_hat_perturbed = co_optimizer(args=args, thetas=theta_hat_perturbed, instance=training_instance, latency_function=latency_function, verbose=args.verbose)
        return perturbation * y_hat_perturbed, theta_hat_perturbed * y_hat_perturbed, perturbation
    else:
        raise Exception("Wrong perturbation type.")


def solve_perturbed_pipeline(args, model, training_instance, co_optimizer, epoch):
    latency_function = None
    latency_function_package = None

    theta = model.predict([training_instance])
    y = training_instance["y"].to_numpy()
    y_hat = co_optimizer(args, theta, training_instance, latency_function=latency_function, verbose=args.verbose)
    if args.co_optimizer == "wardropequilibria":
        correct_objective = theta[0] * y + (1/2) * theta[1] * y**2
        predicted_objective = theta[0] * y_hat + (1 / 2) * theta[1] * y_hat ** 2
        print(f"Theta MAX: {max(theta[0])}, Theta MIN: {min(theta[0])}")
    else:
        correct_objective = theta * y
        predicted_objective = theta * y_hat
        print(f"Theta MAX: {max(theta)}, Theta MIN: {min(theta)}")

    if args.verbose and epoch % (args.num_training_epochs - 1) == 0:
        graph = Graph()
        graph.set_nodes_from_instance(training_instance)
        graph.set_links_from_instance(training_instance, weights=y)
        graph.draw()
        plt.title("y")
        plt.show()
        graph.set_links_from_instance(training_instance, weights=y_hat)
        graph.draw()
        plt.title("y_hat")
        plt.show()
        graph.set_links_from_instance(training_instance, weights=theta)
        graph.draw()
        plt.title("Theta")
        plt.show()

    if args.feature != "withoutSpawn":
        with mp.get_context("spawn").Pool(args.num_cpus-1) as pool:  # Alternatively use without 'spawn': mp.Pool() as pool:
            saver = pool.map(partial(loss_for_perturbation, (args, training_instance, theta, co_optimizer, latency_function_package)), list(range(args.num_perturbations)))
    else:
        with mp.Pool(args.num_cpus-1) as pool:  # Alternatively use without 'spawn': mp.Pool() as pool:
            saver = pool.map(partial(loss_for_perturbation, (args, training_instance, theta, co_optimizer, latency_function_package)), list(range(args.num_perturbations)))
    if len(saver) == 0:
        scaled_y_hat_perturbed_mean = y_hat
        scaled_y_hat_perturbed_squared_mean = (1 / 2) * y_hat ** 2
        predicted_objective = predicted_objective
        perturbation_mean = np.zeros(theta.shape)
    else:
        scaled_y_hat_perturbed_mean = np.nanmean([sav[0] for sav in saver], axis=0)
        scaled_y_hat_perturbed_squared_mean = np.nanmean([(1/2) * sav[0]**2 for sav in saver], axis=0)
        predicted_objective = np.nanmean([sav[1] for sav in saver], axis=0)
        perturbation_mean = np.nanmean([sav[2] for sav in saver], axis=0)
    return y, y_hat, predicted_objective, correct_objective, scaled_y_hat_perturbed_mean, scaled_y_hat_perturbed_squared_mean, perturbation_mean, theta


def get_loss(y, y_hat, predicted_objective, correct_objective):
    mse_value = np.square(np.subtract(y, y_hat)).mean()
    mae_value = np.abs(np.subtract(y, y_hat)).mean()
    loss_value = np.sum(predicted_objective - correct_objective)
    return mse_value, mae_value, loss_value

