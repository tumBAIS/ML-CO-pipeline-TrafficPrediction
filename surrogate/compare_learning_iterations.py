import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
main_directory = "/".join(SCRIPT_DIR.split("/")[:-1])
sys.path.append(main_directory)
import numpy as np
import shutil
import matplotlib.pyplot as plt
from surrogate.src import config
from surrogate.src.util import (load_json, update_args, get_result_name, create_directories,
                                get_iteration_file, get_model_file_name, save_json)


def save(value):
    with open(f"{get_iteration_file(args)}", 'w') as f:
        f.write(f"{value}")


def get_best_iteration(args):
    iteration_saver = {}
    for filename in os.listdir(args.dir_results):
        if get_result_name(args) in filename:
            iteration = int(filename.split("_it-")[1])
            data = load_json(args.dir_results + filename)
            if args.evaluation_metric == "timedist":
                iteration_saver[iteration] = np.mean(data["time_dist"])
            else:
                iteration_saver[iteration] = np.mean(data["mean_absolute_error"])
    iteration_saver_sorted = sorted(iteration_saver.items())
    x, y = zip(*iteration_saver_sorted)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.title("Validation")
    plt.xlabel("Training epochs")
    plt.ylabel("Error")
    plt.savefig(f"{args.dir_visualization}VALIDATION{get_model_file_name(args)}.png")
    save_json({"training_epochs": x, "error": y}, f"{args.dir_visualization}VALIDATIONDATA{get_model_file_name(args)}")
    return min(iteration_saver, key=iteration_saver.get)


def delete_useless_models(keep_iteration):
    for filename in os.listdir(args.dir_models):
        if get_model_file_name(args) in filename:
            if f"_it:{keep_iteration}." in filename:
                continue
            dir_to_delete = os.path.join(args.dir_models, filename)
            if os.path.isfile(dir_to_delete):
                os.remove(dir_to_delete)
                print(f"The file {dir_to_delete} has been deleted.")
            elif os.path.isdir(dir_to_delete):
                shutil.rmtree(dir_to_delete)
                print(f"The directory {dir_to_delete} has been deleted.")
            else:
                print(f"The directory {dir_to_delete} does not exist.")


def delete_validation_results():
    for filename in os.listdir(args.dir_results):
        if get_result_name(args) in filename:
            if "Validate" in filename:
                file_to_delete = os.path.join(args.dir_results, filename)
                os.remove(file_to_delete)
                print(f"The file {file_to_delete} has been deleted.")


if __name__ == "__main__":
    args = config.parser.parse_args()
    args.mode = "Validate"
    args = update_args(args)
    create_directories([args.dir_best_iteration_saver])

    # Derive the best training epoch by comparing all validation results
    best_iteration = get_best_iteration(args)

    # Saving the best training epoch
    save(value=best_iteration)

    # If there are memory problems we can delete the model savings from non-best training epochs
    if (args.evaluation_procedure == "original") and (best_iteration != -1):
        delete_useless_models(best_iteration)

    """delete_validation_results()"""
