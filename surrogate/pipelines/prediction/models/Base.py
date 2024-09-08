import pandas as pd
import numpy as np
import tensorflow as tf
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.src.util import save_json, get_model_file, load_json


class Base(NN):
    def __init__(self, args):
        super().__init__(args)
        self.prediction_value = None

    def create_model(self, X):
        ...

    def save_model(self, step):
        if self.args.learning == "supervised":
            save_json({"prediction": self.prediction_value}, get_model_file(self.args))

    def load_model(self, X, epoch=None):
        if self.args.learning == "supervised":
            prediction_file = load_json(get_model_file(self.args))
            self.prediction_value = prediction_file["prediction"]

    def predict(self, instances):
        if self.args.learning == "supervised":
            _, y = self.batch_dataset(instances)
            return np.array([self.prediction_value] * len(y))
        elif self.args.learning in ["structured", "structured_outer"]:
            link_distances = []
            for training_instance in instances:
                link_distance = pd.DataFrame({"link_id": training_instance["links_id"], "link_length": -1 * np.array(training_instance["link_length"])})
                link_distance = training_instance["solution_representation"].solution_scheme.merge(link_distance, on="link_id", how="left")
                link_distance["link_length"].fillna(0, inplace=True)
                link_distances.append(link_distance["link_length"].values)
            if self.args.co_optimizer in ["wardropequilibria"]:
                return np.array([(link_dist, -1) for link_dist in np.concatenate(link_distances)]).T
            else:
                return np.concatenate(link_distances)

    def batch_dataset(self, instances):
        return None, pd.concat([i["y"] for i in instances]).values

    def sl_optimize(self, y, y_hat_mean, instance):
        ...

    def sup_optimize(self, instances, callback):
        _, y = self.batch_dataset(instances)
        self.prediction_value = np.mean(y)
        logs = {"loss": self.loss(self.predict(instances), y).numpy(),
                "mean_squared_error": tf.keras.metrics.mean_squared_error(self.predict(instances), y).numpy(),
                "mean_absolute_error": tf.keras.metrics.mean_absolute_error(self.predict(instances), y).numpy()}
        callback.on_train_end(logs)

