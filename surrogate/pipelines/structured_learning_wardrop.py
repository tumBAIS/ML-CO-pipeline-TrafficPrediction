import time
from surrogate.src import util
import multiprocessing as mp
from functools import partial
import numpy as np
from surrogate.visualizing.learning_results import generate_plot_learning_evolution
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest

MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}


def save_learning_evaluation(args, learning_evolution):
    util.save_json(learning_evolution, util.get_learning_file(args))


def optimize(args, model, training_instances, co_optimizer, suffix=""):

    if (args.co_optimizer in ["successiveaverages", "wardropequilibria", "wardropequilibriaRegularized", "frankwolfe", "multicommodityflow"]) and (args.model in ["NN"]):
        mp.set_start_method('spawn')

    overall_loss = []

    time_start_training = time.time()
    model.save_model(step=-1)
    for epoch in range(args.num_training_epochs):
        print(f"EPOCH: {epoch}")
        mse_values = []
        mae_values = []
        loss_values = []
        with mp.get_context("spawn").Pool(args.num_cpus - 1) as pool:
            saver = pool.map(partial(solve_pipeline, (args, co_optimizer, epoch-1)), training_instances)
        for (y, y_hat, theta, learned_objective, correct_objective), training_instance in zip(saver, training_instances):
            mse_value, mae_value, loss_value = get_loss(y, y_hat, learned_objective, correct_objective)
            model.sl_optimize(y, y_hat, (1 / 2) * y_hat ** 2, [input.values for input in training_instance["X"]], None, None)
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


def solve_pipeline(kwargs, training_instance):
    args, co_optimizer, epoch = kwargs

    model = MODEL[args.model](args)
    model.load_model(training_instance["X"], epoch)

    theta = model.predict([training_instance])
    y = training_instance["y"].to_numpy()
    y_hat = co_optimizer(args, theta, training_instance, latency_function=model, verbose=args.verbose)
    if args.co_optimizer == "wardropequilibria" and args.learning == "structured":
        correct_objective = np.sum(theta[0] * y + (1 / 2) * theta[1] * (y ** 2))
        learned_objective = np.sum(theta[0] * y_hat + (1 / 2) * theta[1] * (y_hat ** 2))
        print(f"Theta MAX: {max(theta[0])}, Theta MIN: {min(theta[0])}")
    elif args.co_optimizer == "wardropequilibriaRegularized" and args.learning == "structured":
        correct_objective = np.sum(theta * y - (1 / 2) * y ** 2)
        learned_objective = np.sum(theta * y_hat - (1 / 2) * y_hat**2)
        print(f"Theta MAX: {max(theta)}, Theta MIN: {min(theta)}")
    else:
        correct_objective = theta * y
        learned_objective = theta * y_hat
        print(f"Theta MAX: {max(theta)}, Theta MIN: {min(theta)}")
    return y, y_hat, theta, learned_objective, correct_objective


def get_loss(y, y_hat, predicted_objective, correct_objective):
    mse_value = np.square(np.subtract(y, y_hat)).mean()
    mae_value = np.abs(np.subtract(y, y_hat)).mean()
    loss_value = np.mean(predicted_objective - correct_objective)
    return mse_value, mae_value, loss_value