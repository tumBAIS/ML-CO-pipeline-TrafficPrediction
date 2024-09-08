import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import sys
import tensorflow as tf
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-1])
sys.path.append(main_directory)
from surrogate.pipelines.prediction.features import load_training_instances
from surrogate.pipelines.structured_learning import optimize as optimize_SL
from surrogate.pipelines.supervised_learning import optimize as optimize_Sup
from surrogate.pipelines.structured_learning_bfgs import optimize as optimize_SL_bfgs
from surrogate.pipelines.structured_learning_wardrop import optimize as optimize_SL_easy
from surrogate.pipelines.prediction.models.GNN import GNN
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.pipelines.prediction.models.Base import Base
from surrogate.pipelines.prediction.models.Linear import Linear
from surrogate.pipelines.prediction.models.RandomForest import RandomForest
from surrogate.src import config
from surrogate.src.util import create_directories, update_args
from surrogate.pipelines.optimizing.helpers_multicommodity.get_capacity_expanded_components import load_capacities
from surrogate.pipelines.optimizing.multicommodityflow import optimize as multicommodityflow_optimize
from surrogate.pipelines.optimizing.wardropequilibrium import optimize as wardrop_optimize


if __name__ == "__main__":
    args = config.parser.parse_args()
    args.mode = "Train"
    args = update_args(args)
    print(args)
    create_directories([args.dir_models, args.dir_learning, args.dir_standardization, args.dir_visualization])

    LEARNING = {"supervised": optimize_Sup, "structured": optimize_SL if args.num_perturbations > 0 else optimize_SL_easy, "structured_bfgs": optimize_SL_bfgs}
    MODEL = {"GNN": GNN, "NN": NN, "Base": Base, "Linear": Linear, "RandomForest": RandomForest}
    CO_OPTIMIZER = {"multicommodityflow": multicommodityflow_optimize,
                    "wardropequilibria": wardrop_optimize,
                    "wardropequilibriaRegularized": wardrop_optimize}

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

    print("Standardize features of training instances...")
    training_instances = model.standardize(training_instances)

    print("Load capacities")
    training_instances = load_capacities(args, training_instances)

    print("Create model architecture...")
    model.create_model(training_instances[0]["X"])

    print("Create CO-layer")
    co_optimizer = CO_OPTIMIZER[args.co_optimizer]

    print("Learn the model...")
    LEARNING[args.learning](args, model, training_instances, co_optimizer)
