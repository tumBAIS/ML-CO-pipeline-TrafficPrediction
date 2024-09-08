import time

import keras
import matplotlib.pyplot as plt
from surrogate.pipelines.structured_learning import save_learning_evaluation
from surrogate.src.util import Graph
from surrogate.visualizing.learning_results import generate_plot_learning_evolution


class LearningCallback(keras.callbacks.Callback):
    def __init__(self, args, prediction_model, training_instances, suffix):
        super().__init__()
        self.args = args
        self.prediction_model = prediction_model
        self.epoch_logs = []
        self.training_instances = training_instances
        self.suffix = suffix
        self.time_start_training = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_logs.append(logs)
        self.prediction_model.save_model(epoch)
        save_learning_evaluation(self.args, self.epoch_logs)
        if epoch % 10 == 0:
            if self.args.verbose:
                y = self.prediction_model.predict([self.training_instances[0]])
                print(f"Theta MAX: {max(y)}, Theta MIN: {min(y)}")
                graph = Graph()
                graph.set_nodes_from_instance(self.training_instances[0])
                graph.set_links_from_instance(self.training_instances[0], weights=y)
                graph.draw()
                plt.show()
        if time.time() - self.time_start_training > self.args.max_training_time.total_seconds():
            print("STOP LEARNING DUE TO TIME LIMIT")
            self.model.stop_training = True

    def on_train_end(self, logs=None, plotting=True):
        print("--- Save epoch logs ---")
        self.epoch_logs.append(logs)
        self.prediction_model.save_model(-1)
        save_learning_evaluation(self.args, self.epoch_logs)
        if plotting:
            generate_plot_learning_evolution(self.args, self.args, self.epoch_logs, suffix=self.suffix)


def optimize(args, model, training_instances, co_optimizer, suffix=""):
    model.sup_optimize(training_instances, LearningCallback(args, model, training_instances, suffix=suffix))
