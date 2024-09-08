import numpy as np
from sklearn.linear_model import LinearRegression
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.src.util import get_model_file, get_standardization_file, save_json, load_json
from surrogate.pipelines.prediction.models.helpers_latencyfunctions.affine_function import AffineFunction


class Linear(NN):
    def __init__(self, args):
        super().__init__(args)
        self.w = None
        self.learning_rate = args.learning_rate
        self.latency_function = AffineFunction()

    def create_model(self, X):
        self.w = {feature: start_value for feature, start_value in zip(X[0].columns, np.random.uniform(low=-10, high=0, size=(self.get_input_dimension(X),)))}

    def save_model(self, step):
        save_json(self.w, get_model_file(self.args) + f"_it:{step}")
        save_json(self.divisor, get_standardization_file(self.args))

    def load_model(self, epoch=None):
        if epoch is None:
            self.w = load_json(get_model_file(self.args) + f"_it:{self.args.read_in_iteration}")
        else:
            self.w = load_json(get_model_file(self.args) + f"_it:{epoch}")
        self.divisor = load_json(get_standardization_file(self.args))

    def get_w_values(self):
        return np.array(list(self.w.values()))

    def set_w_values(self, values):
        self.w = {key: value for key, value in zip(self.w.keys(), values)}

    def prepare_prediction(self):
        self.w_intercept = np.array([value for key, value in self.w.items() if "traffic_count" not in key])
        self.w_intercept_columns = [True if "traffic_count" not in key else False for key in self.w.keys()]
        self.w_x = np.array([value for key, value in self.w.items() if "traffic_count" in key])
        self.w_x_columns = [True if "traffic_count" in key else False for key in self.w.keys()]

    def predict(self, instances):
        X, y = self.batch_dataset(instances)

        if self.args.co_optimizer in ["wardropequilibria"]:
            # we assume a latency function a_e + b_e * x_e, with x_e being the aggregated flow on edge e
            self.prepare_prediction()
            latency_intercept = (X[:, self.w_intercept_columns] * self.w_intercept).sum(axis=1)
            latency_x = (X[:, self.w_x_columns] * self.w_x).sum(axis=1)
            return (latency_intercept, latency_x)
        else:
            y_hat = np.squeeze(np.matmul(X, self.get_w_values()))
            #y_hat[y_hat > 0] = np.random.uniform(low=-0.000001, high=0, size=sum(y_hat > 0))
            return y_hat

    def batch_dataset(self, instances):
        X, y = self.batch_dataset_from_instances(instances)
        return X, y

    def sl_optimize(self, y, scaled_y_hat_mean, X, Z_mean, thetas_original):
        if self.args.co_optimizer in ["wardropequilibria"]:
            self.prepare_prediction()
            gradient_offline_intercept = (X[0][:, self.w_intercept_columns] * y.reshape((len(y), 1))).sum(axis=0)
            gradient_offline_x = (X[0][:, self.w_x_columns] * np.square(y).reshape((len(y), 1))).sum(axis=0)
            gradient_online_mean_intercept = (X[0][:, self.w_intercept_columns] * scaled_y_hat_mean.reshape((len(y), 1))).sum(axis=0)
            gradient_online_x = (X[0][:, self.w_x_columns] * np.square(scaled_y_hat_mean).reshape((len(y), 1))).sum(axis=0)

            gradient_intercept = gradient_online_mean_intercept - gradient_offline_intercept
            gradient_x = gradient_online_x - gradient_offline_x

            self.set_w_values(self.get_w_values() - self.args.learning_rate * np.concatenate([gradient_intercept, gradient_x]))
        else:
            gradient_offline = (X[0] * y.reshape((len(y), 1))).sum(axis=0)
            gradient_online_mean = (X[0] * scaled_y_hat_mean.reshape((len(y), 1))).sum(axis=0)
            gradient = gradient_online_mean - gradient_offline
            #self.set_w_values(self.get_w_values() - 0.0001 * gradient.reshape((len(gradient), 1)))
            #self.set_w_values(self.get_w_values() - 0.0001 * gradient)
            self.set_w_values(self.get_w_values() - self.args.learning_rate * gradient)

    def sup_optimize(self, instances, callback):
        X, y = self.batch_dataset_from_instances(instances)
        reg = LinearRegression().fit(X, y)
        self.set_w_values(reg.coef_.reshape((len(reg.coef_), 1)))

    def get_weights(self):
        return self.w

    def load_weights(self, weights):
        self.w = weights

    def get_gradient(self, instance):
        return np.repeat(self.w["traffic_count"], len(instance["X"][0]))
