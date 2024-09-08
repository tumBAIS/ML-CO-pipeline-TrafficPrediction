import pickle
from sklearn.ensemble import RandomForestRegressor
from surrogate.pipelines.prediction.models.NN import NN
from surrogate.src.util import get_model_file, get_standardization_file, load_json, save_json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class RandomForest(NN):
    def __init__(self, args):
        super().__init__(args)

    def sup_optimize(self, instances, callback):
        X, y = self.batch_dataset(instances)
        self.model.fit(X, y)
        y_hat = self.model.predict(X)

        logs = {mean_absolute_error.__name__: mean_absolute_error(y, y_hat),
                mean_squared_error.__name__: mean_squared_error(y, y_hat)}
        callback.on_train_end(logs, plotting=False)

    def create_model(self, X):
        self.model = RandomForestRegressor()

    def save_model(self, step):
        pickle.dump(self.model, open(get_model_file(self.args), 'wb'))
        save_json(self.divisor, get_standardization_file(self.args))

    def load_model(self, X, epoch=None):
        self.model = pickle.load(open(get_model_file(self.args), 'rb'))
        self.divisor = load_json(get_standardization_file(self.args))

    def batch_dataset(self, instances):
        X, y = self.batch_dataset_from_instances(instances)
        return X, y

    def predict(self, instances):
        dataset, y = self.batch_dataset(instances)
        return np.squeeze(self.model.predict(dataset))
