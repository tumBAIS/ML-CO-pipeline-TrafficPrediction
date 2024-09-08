import tensorflow as tf
from surrogate.pipelines.prediction.models.NN import NN


class Linear(NN):
    def __init__(self, args):
        super().__init__(args)

    def create_model(self, X):
        input_dimension = self.get_input_dimension(X)
        node_features = tf.keras.layers.Input(input_dimension, dtype="float32", name="node_features")
        linear = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, use_bias=False)
        if self.args.learning in ["structured", "structured_outer"]:
            y_neg = linear(node_features)
            y = tf.multiply(y_neg, -1)
        elif self.args.learning == "supervised":
            y = linear(node_features)
        else:
            raise Exception("Unknown learning argument.")

        self.model = tf.keras.Model(
            inputs=[node_features],
            outputs=[y],
        )

        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])
