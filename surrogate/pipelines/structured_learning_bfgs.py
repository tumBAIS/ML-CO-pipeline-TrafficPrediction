import time
import copy
from surrogate.src import util
from multiprocessing import Pool
import multiprocessing as mp
import os
from functools import partial
import numpy as np
import scipy
import matplotlib.pyplot as plt
from surrogate.visualizing.learning_results import generate_plot_learning_evolution
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest
from surrogate.src.util import Graph

MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}
overall_loss = []
epoch = 0


def save_learning_evaluation(args, learning_evolution):
    util.save_json(learning_evolution, util.get_learning_file(args))


def loss_for_perturbation(kwargs, perturbation_num):

    np.random.seed(perturbation_num)

    def apply_multiplicative_perturbation(vector):
        perturbation = np.exp(args.sd_perturbation * np.random.normal(loc=0, scale=1, size=vector.shape) - ((args.sd_perturbation**2) * (1/2)))
        vector_perturbed = vector * perturbation
        return vector_perturbed, perturbation

    def apply_additive_perturbation(vector):
        perturbation = np.random.normal(loc=0, scale=args.sd_perturbation, size=vector.shape)
        vector_perturbed = vector + perturbation
        return vector_perturbed, perturbation

    args, training_instance, theta_hat, co_optimizer, latency_function_package = kwargs

    latency_function = MODEL[args.model](args)
    latency_function.load_weights(latency_function_package)

    if args.perturbation == 'additive':
        perturbed_w_values, perturbation = apply_additive_perturbation(latency_function.get_w_values())
        latency_function.set_w_values(perturbed_w_values)
        theta_hat_perturbed = latency_function.predict([training_instance])
        y_hat_perturbed = co_optimizer(args=args, thetas=theta_hat_perturbed, instance=training_instance, latency_function=latency_function, verbose=args.verbose)
        gradient_perturbed = np.sum(training_instance["X"][0].values.T * y_hat_perturbed, axis=1)
        return theta_hat_perturbed, y_hat_perturbed, gradient_perturbed
    else:
        raise Exception("Wrong perturbation type.")


def solve_perturbed_pipeline(args, model, training_instance, co_optimizer, epoch):

    theta = model.predict([training_instance])
    y = training_instance["y"].to_numpy()
    y_hat = co_optimizer(args, theta, training_instance, latency_function=model, verbose=args.verbose)
    gradient_correct = np.sum(training_instance["X"][0].values.T * y, axis=1)

    with mp.Pool(args.num_cpus-1) as pool:
        saver = pool.map(partial(loss_for_perturbation, (args, training_instance, theta, co_optimizer, model.w)), list(range(args.num_perturbations)))
    theta_hat_perturbed_mean = np.mean([sav[0] for sav in saver], axis=0)
    y_hat_perturbed_mean = np.mean([sav[1] for sav in saver], axis=0)
    gradient_perturbed_mean = np.mean([sav[2] for sav in saver], axis=0)
    correct_objective = theta * y
    predicted_objective = theta * y_hat
    y_merged = y_hat_perturbed_mean - y
    loss_value = np.sum(predicted_objective - correct_objective)
    gradient = gradient_perturbed_mean - gradient_correct
    return y, y_hat, loss_value, gradient


def get_loss(y, y_hat):
    mse_value = np.square(np.subtract(y, y_hat)).mean()
    mae_value = np.abs(np.subtract(y, y_hat)).mean()
    return mse_value, mae_value


def objective(x, args, model, training_instances, co_optimizer, suffix):
    global epoch
    epoch += 1
    model.set_w_values(x)

    mse_values = []
    mae_values = []
    loss_values = []
    gradients = []
    for training_instance in training_instances:
        y, y_hat, loss_value, gradient = solve_perturbed_pipeline(args, model, training_instance, co_optimizer, epoch)
        mse_value, mae_value = get_loss(y, y_hat)
        mse_values.append(mse_value)
        mae_values.append(mae_value)
        loss_values.append(loss_value)
        gradients.append(gradient)
    overall_loss.append({"loss": np.mean(loss_values), "mean_squared_error": np.mean(mse_values), "mean_absolute_error": np.mean(mae_values)})
    print("Epoch: {} ----> Loss: {} | MSE: {} | MAE: {}".format(epoch, np.mean(loss_values), np.mean(mse_values), np.mean(mae_value)))
    model.save_model(epoch)
    save_learning_evaluation(args, overall_loss)
    generate_plot_learning_evolution(args, overall_loss, suffix=suffix)

    return float(np.sum(loss_values)), np.sum(gradients, axis=0)


def optimize(args, model, training_instances, co_optimizer, suffix=""):
    start_value = model.get_w_values()
    x_point = scipy.optimize.minimize(objective, start_value, args=(args, model, training_instances, co_optimizer, suffix), method="L-BFGS-B", jac=True, options={"maxiter": args.num_training_epochs, "disp": True})
    print(f"Final value: {x_point}")
