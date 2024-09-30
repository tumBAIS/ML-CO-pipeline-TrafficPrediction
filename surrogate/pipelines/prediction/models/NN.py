import tensorflow as tf
import pandas as pd
import numpy as np
from surrogate.src.util import get_model_file, save_json, get_standardization_file, load_json
from surrogate.pipelines.prediction import features
from surrogate.pipelines.solution_representation import Solution
from collections import Counter
from surrogate.pipelines.optimizing.multicommodityflow_uncapacitated import optimize as optimize_mcfp


def create_ffn(hidden_units):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(tf.keras.layers.Dense(units, activation="relu"))
    return tf.keras.Sequential(fnn_layers)


class NN:
    def __init__(self, args):
        self.args = args
        self.hidden_units = [100, 500, 100, 10, 5]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model = None
        self.divisor = None
        tf.random.set_seed(args.seed)

    def get_solution(self, args, training_instances):
        for training_instance in training_instances:
            solution_representation = Solution(args)
            training_instance["solution_representation"] = solution_representation
            training_instance["y"] = solution_representation.load_solution_scheme(training_instance)
        return training_instances

    def get_features(self, args, training_instances):
        for training_instance in training_instances:
            self.get_features_instance(args, training_instance)
        return training_instances

    def get_features_for_links(self, args, training_instance):
        training_instance["links_x"], training_instance["links_y"] = features.get_link_coordinates(data=training_instance)

        count_to_from, count_from_to, count_from_from, count_to_to = features.get_num_streets_ingoingoutgoing(data=training_instance)

        feature_num_homes1 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="home", divisor=1)
        feature_num_homes2 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="home", divisor=2)
        feature_num_homes5 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="home", divisor=5)
        feature_num_homes10 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="home", divisor=10)
        feature_num_homes15 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="home", divisor=15)

        feature_num_works1 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="work", divisor=1)
        feature_num_works2 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="work", divisor=2)
        feature_num_works5 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="work", divisor=5)
        feature_num_works10 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="work", divisor=10)
        feature_num_works15 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="work", divisor=15)

        feature_num_nodes1 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="nodes", divisor=1)
        feature_num_nodes2 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="nodes", divisor=2)
        feature_num_nodes5 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="nodes", divisor=5)
        feature_num_nodes10 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="nodes", divisor=10)
        feature_num_nodes15 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="nodes", divisor=15)

        feature_num_links1 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=1)
        feature_num_links2 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=2)
        feature_num_links5 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=5)
        feature_num_links10 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=10)
        feature_num_links15 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=15)

        feature_num_links_capacity_1000_1 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=1, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_2 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=2, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_5 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=5, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_10 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=10, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_15 = features.get_num_items_around_location(self.args, data=training_instance, location="links", item="links", divisor=15, attribute="link_capacity", threshold=1000)

        feature_link_length = training_instance["link_length"]
        feature_link_speed = training_instance["link_freespeed"]
        feature_link_capacity = training_instance["link_capacity"]
        feature_link_permlanes = training_instance["link_permlanes"]
        feature_link_time = [length / speed for length, speed in zip(training_instance["link_length"], training_instance["link_freespeed"])]

        X = pd.DataFrame({"feature_num_homes1": feature_num_homes1,
                          "feature_num_homes2": feature_num_homes2,
                          "feature_num_homes5": feature_num_homes5,
                          "feature_num_homes10": feature_num_homes10,
                          "feature_num_homes15": feature_num_homes15,
                          "feature_num_works1": feature_num_works1,
                          "feature_num_works2": feature_num_works2,
                          "feature_num_works5": feature_num_works5,
                          "feature_num_works10": feature_num_works10,
                          "feature_num_works15": feature_num_works15,
                          "feature_num_nodes1": feature_num_nodes1,
                          "feature_num_nodes2": feature_num_nodes2,
                          "feature_num_nodes5": feature_num_nodes5,
                          "feature_num_nodes10": feature_num_nodes10,
                          "feature_num_nodes15": feature_num_nodes15,
                          "feature_num_links1": feature_num_links1,
                          "feature_num_links2": feature_num_links2,
                          "feature_num_links5": feature_num_links5,
                          "feature_num_links10": feature_num_links10,
                          "feature_num_links15": feature_num_links15,
                          "feature_num_links_capacity_1000_1": feature_num_links_capacity_1000_1,
                          "feature_num_links_capacity_1000_2": feature_num_links_capacity_1000_2,
                          "feature_num_links_capacity_1000_5": feature_num_links_capacity_1000_5,
                          "feature_num_links_capacity_1000_10": feature_num_links_capacity_1000_10,
                          "feature_num_links_capacity_1000_15": feature_num_links_capacity_1000_15,
                          "feature_link_length": feature_link_length,
                          "feature_link_speed": feature_link_speed,
                          "feature_link_capacity": feature_link_capacity,
                          "feature_link_permlanes": feature_link_permlanes,
                          "feature_link_time": feature_link_time,
                          "count_to_from": count_to_from,
                          "count_from_to": count_from_to,
                          "count_from_from": count_from_from,
                          "count_to_to": count_to_to
                          })

        X["link_id"] = training_instance["links_id"]
        X = X.set_index("link_id")
        return X

    def get_co_features(self, args, training_instance):
        theta = pd.DataFrame({"link_id": training_instance["links_id"], "link_length": -1 * np.array(training_instance["link_length"])})
        theta = training_instance["solution_representation"].solution_scheme.merge(theta, on="link_id", how="left")
        theta["link_length"].fillna(0, inplace=True)
        theta = theta["link_length"].values
        solution = optimize_mcfp(args, thetas=theta, instance=training_instance)
        co_features = training_instance["solution_representation"].load_y(solution)
        co_features = co_features.rename("co_feature")
        return co_features

    def get_features_instance(self, args, training_instance, add_features=None):
        solution_representation = training_instance["solution_representation"]

        # Add features for links
        X_links = self.get_features_for_links(args, training_instance)

        if add_features is not None:
            X_links.reset_index(drop=True, inplace=True)
            add_features.reset_index(drop=True, inplace=True)
            X_links = pd.concat([X_links, add_features], axis=1)
        X = solution_representation.solution_scheme.join(X_links, how="left", on="link_id")[X_links.columns]

        if args.time_variant_theta:
            if args.feature == "original":
                X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time - training_instance["minimum_start"],
                                       "time_feature_inverse": np.flip(solution_representation.solution_scheme_time - training_instance["minimum_start"]),
                                       "time": solution_representation.solution_scheme_time,
                                       "time_index": np.arange(len(solution_representation.solution_scheme_time)),
                                       "time_index_inverse": np.flip(np.arange(len(solution_representation.solution_scheme_time))),
                                       })
                X_time = X_time.set_index(["time"])
                X = X.join(X_time, how="left", on="time")[list(X_time.columns) + list(X.columns)]
                X_time_origins = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 0]), orient="index", columns=["time_origins"])
                X_time_origins["link_id"] = -X_time_origins.index - 1
                X_time_origins = X_time_origins.set_index(["link_id"])
                X_time_destinations = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 1]), orient="index", columns=["time_destinations"])
                X_time_destinations["link_id"] = -X_time_destinations.index - 1
                X_time_destinations = X_time_destinations.set_index(["link_id"])
                X = X.merge(X_time_origins, how="left", left_index=True, right_index=True)
                X = X.merge(X_time_destinations, how="left", left_index=True, right_index=True)
                X["time_origins"].fillna(0, inplace=True)
                X["time_destinations"].fillna(0, inplace=True)
                X["artificial_link"] = 0
                X.loc[X.index.get_level_values("link_id") < 0, "artificial_link"] = 1
            elif args.feature == "logic" or args.feature == "withoutedgeslogic":
                X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time - training_instance["minimum_start"],
                                       "time_feature_inverse": np.flip(solution_representation.solution_scheme_time - training_instance["minimum_start"]),
                                       "time": solution_representation.solution_scheme_time,
                                       "distance_from_half": solution_representation.solution_scheme_time - training_instance["minimum_start"] - (
                                                   (training_instance["maximum_end"] - training_instance["minimum_start"]) / 2)
                                       })
                X_time = X_time.set_index(["time"])
                X = X.join(X_time, how="left", on="time")[list(X_time.columns) + list(X.columns)]
                X_time_origins = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 0]), orient="index", columns=["time_origins"])
                X_time_origins["link_id"] = -X_time_origins.index - 1
                X_time_origins = X_time_origins.set_index(["link_id"])
                X = X.merge(X_time_origins, how="left", left_index=True, right_index=True)
                X["time_origins"].fillna(0, inplace=True)
                X["artificial_link"] = 0
                X.loc[X.index.get_level_values("link_id") < 0, "artificial_link"] = 1
            elif args.feature == "square" or args.feature == "withoutedgessquare":
                X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time - training_instance["minimum_start"],
                                       "time_feature_inverse": np.flip(solution_representation.solution_scheme_time - training_instance["minimum_start"]),
                                       "time": solution_representation.solution_scheme_time,
                                       "distance_from_half": solution_representation.solution_scheme_time - training_instance["minimum_start"] - (
                                                   (training_instance["maximum_end"] - training_instance["minimum_start"]) / 2),
                                       "distance_from_half_square": (solution_representation.solution_scheme_time - training_instance["minimum_start"] - (
                                                   (training_instance["maximum_end"] - training_instance["minimum_start"]) / 2)) ** 2,
                                       "distance_from_beginning_square": (solution_representation.solution_scheme_time - training_instance["minimum_start"]) ** 2,
                                       "distance_from_half_three": (solution_representation.solution_scheme_time - training_instance["minimum_start"] - (
                                                   (training_instance["maximum_end"] - training_instance["minimum_start"]) / 2)) ** 3,
                                       "distance_from_beginning_three": (solution_representation.solution_scheme_time - training_instance["minimum_start"]) ** 3
                                       })
                X_time = X_time.set_index(["time"])
                X = X.join(X_time, how="left", on="time")[list(X_time.columns) + list(X.columns)]
                X_time_origins = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 0]), orient="index", columns=["time_origins"])
                X_time_origins["link_id"] = -X_time_origins.index - 1
                X_time_origins = X_time_origins.set_index(["link_id"])
                X = X.merge(X_time_origins, how="left", left_index=True, right_index=True)
                X["time_origins"].fillna(0, inplace=True)
                X["artificial_link"] = 0
                X.loc[X.index.get_level_values("link_id") < 0, "artificial_link"] = 1
            elif args.learning == "supervised" or ((args.learning == "structured") and (args.co_optimizer == "multicommodityflow")):
                X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time - training_instance["minimum_start"],
                                       "time_feature_inverse": np.flip(solution_representation.solution_scheme_time - training_instance["minimum_start"]),
                                       "time": solution_representation.solution_scheme_time,
                                       "time_index": np.arange(len(solution_representation.solution_scheme_time)),
                                       "time_index_inverse": np.flip(np.arange(len(solution_representation.solution_scheme_time))),
                                       })
                X_time = X_time.set_index(["time"])
                X = X.join(X_time, how="left", on="time")[list(X_time.columns) + list(X.columns)]
                X_time_origins = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 0]), orient="index", columns=["time_origins"])
                X_time_origins["link_id"] = -X_time_origins.index - 1
                X_time_origins = X_time_origins.set_index(["link_id"])
                X = X.merge(X_time_origins, how="left", left_index=True, right_index=True)
                X["time_origins"].fillna(0, inplace=True)
                X["artificial_link"] = 0
                X.loc[X.index.get_level_values("link_id") < 0, "artificial_link"] = 1
            else:
                X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time - training_instance["minimum_start"],
                                       "time_feature_inverse": np.flip(solution_representation.solution_scheme_time - training_instance["minimum_start"]),
                                       "time": solution_representation.solution_scheme_time,
                                       "distance_from_half": solution_representation.solution_scheme_time - training_instance["minimum_start"] - ((training_instance["maximum_end"] - training_instance["minimum_start"]) / 2),
                                       "distance_from_half_square": (solution_representation.solution_scheme_time - training_instance["minimum_start"] - ((training_instance["maximum_end"] - training_instance["minimum_start"]) / 2))**2,
                                       "distance_from_beginning_square": (solution_representation.solution_scheme_time - training_instance["minimum_start"])**2,
                                       "distance_from_half_three": (solution_representation.solution_scheme_time - training_instance["minimum_start"] - ((training_instance["maximum_end"] - training_instance["minimum_start"]) / 2)) ** 3,
                                       "distance_from_beginning_three": (solution_representation.solution_scheme_time - training_instance["minimum_start"]) ** 3
                                       })
                X_time = X_time.set_index(["time"])
                X = X.join(X_time, how="left", on="time")[list(X_time.columns) + list(X.columns)]
                X_time_origins = pd.DataFrame.from_dict(Counter(np.array(training_instance["o_d_pairs"])[:, 0]), orient="index", columns=["time_origins"])
                X_time_origins["link_id"] = -X_time_origins.index - 1
                X_time_origins = X_time_origins.set_index(["link_id"])
                X = X.merge(X_time_origins, how="left", left_index=True, right_index=True)
                X["time_origins"].fillna(0, inplace=True)
                X["artificial_link"] = 0
                X.loc[X.index.get_level_values("link_id") < 0, "artificial_link"] = 1

        if args.trip_individual_theta:
            X_commodity = pd.DataFrame({"commodity_feature": solution_representation.solution_scheme_commodities, "commodity": solution_representation.solution_scheme_commodities})
            X_commodity = X_commodity.set_index(["commodity"])
            X = X.join(X_commodity, how="left", on="commodity")[list(X_commodity.columns) + list(X.columns)]

        if args.capacity_individual_theta:
            X_capacity = pd.DataFrame({"capacity_feature": np.arange(solution_representation.maximum_capacity), "index_capacity_group": np.arange(solution_representation.maximum_capacity)})
            X_capacity = X_capacity.set_index(["index_capacity_group"])
            X = X.join(X_capacity, how="left", on="index_capacity_group")[list(X_capacity.columns) + list(X.columns)]

        if False:
            feature_target = np.array(training_instance["y"])
            X["feature_target"] = feature_target
            for _ in range(20):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CAUTION : TARGET FEATURE ENABLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        X = X.fillna(0)
        training_instance["X"] = [X]

    def get_standardization_divisor(self, training_instances):
        if self.divisor is None:
            divisor = pd.concat([training_instance["X"][0] for training_instance in training_instances]).mean()
            self.divisor = {key: (value if np.abs(value) > 1 else 1) for key, value in divisor.items()}
        return self.divisor

    def standardize(self, training_instances):
        if self.args.standardize:
            divisor = self.get_standardization_divisor(training_instances)
            for training_instance in training_instances:
                training_instance["X"][0] = training_instance["X"][0] / divisor
        return training_instances

    def get_input_dimension(self, X):
        return X[0].shape[1]

    def create_model(self, X):
        input_dimension = self.get_input_dimension(X)

        node_features = tf.keras.layers.Input(input_dimension, dtype="float32", name="node_features")
        feedforward_layer = create_ffn(self.hidden_units)

        ffn_output = feedforward_layer(node_features)
        if self.args.learning in ["structured", "structured_outer", "structured_wardrop"]:
            pre_output_layer_structured = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, use_bias=False)  #
            pre_output = pre_output_layer_structured(ffn_output)
            y = tf.multiply(pre_output, -1)
        elif self.args.learning == "supervised":
            output_layer_supervised = tf.keras.layers.Dense(1, use_bias=False)
            y = output_layer_supervised(ffn_output)

        if self.args.learning in ["structured"] and self.args.co_optimizer in ["wardropequilibria"]:
            feedforward_layer_wardrop = create_ffn(self.hidden_units)
            ffn_output_wardrop = feedforward_layer_wardrop(node_features)
            pre_output_layer_structured_wardrop = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, use_bias=False)  #
            pre_output_wardrop = pre_output_layer_structured_wardrop(ffn_output_wardrop)
            y_wardrop = tf.multiply(pre_output_wardrop, -1)
            self.model = tf.keras.Model(
                inputs=[node_features],
                outputs=[y, y_wardrop],
            )
        else:
            self.model = tf.keras.Model(
                inputs=[node_features],
                outputs=[y],
            )

        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])

    def save_model(self, step):
        file_output_weights = get_model_file(self.args) + f"_it:{step}.keras"
        self.model.save_weights(file_output_weights)
        save_json(self.divisor, get_standardization_file(self.args))

    def load_model(self, X, epoch=None):
        if epoch is None:
            self.create_model(X)
            self.model.load_weights(get_model_file(self.args) + f"_it:{self.args.read_in_iteration}.keras")
        else:
            self.create_model(X)
            self.model.load_weights(get_model_file(self.args) + f"_it:{epoch}.keras")
        self.divisor = load_json(get_standardization_file(self.args))

    def predict(self, instances):
        dataset, y = self.batch_dataset(instances)
        return np.squeeze(self.model.predict(dataset, verbose=0))

    def batch_dataset_from_instances(self, instances):
        X = pd.concat([i["X"][0] for i in instances]).values.astype('float32')
        y = pd.concat([i["y"] for i in instances]).values.astype('float32')
        return X, y

    def batch_dataset(self, instances):
        X, y = self.batch_dataset_from_instances(instances)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.batch(32), y

    def sl_optimize(self, y, scaled_y_hat_mean, scaled_y_hat_squared_mean, X, Z_mean, thetas_original):
        if self.args.learning in ["structured"] and self.args.co_optimizer in ["wardropequilibria"]:
            with tf.GradientTape() as tape_node:
                thetas = tf.squeeze(self.model(X, training=True))
                assert thetas.shape[1] == len(y) and thetas.shape[1] == len(scaled_y_hat_mean)
                # intercept gradient
                correct_objective_profits_intercept = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(y.astype("float32"))))
                predicted_objective_profits_intercept = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(scaled_y_hat_mean.astype("float32"))))
                loss_intercept = tf.math.subtract(predicted_objective_profits_intercept, correct_objective_profits_intercept)

                # slope gradient
                y_scaled = (1 / 2) * y ** 2
                correct_objective_profits_slope = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(y_scaled.astype("float32"))))
                predicted_objective_profits_slope = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(scaled_y_hat_squared_mean.astype("float32"))))
                loss_slope = tf.math.subtract(predicted_objective_profits_slope, correct_objective_profits_slope)

                loss = tf.math.add(loss_intercept, loss_slope)

            self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))
        elif self.args.learning in ["structured"] and self.args.co_optimizer in ["wardropequilibriaRegularized"]:
            with tf.GradientTape() as tape_node:
                thetas = tf.squeeze(self.model(X, training=True))
                assert len(thetas) == len(y) and len(thetas) == len(scaled_y_hat_mean)
                correct_objective_profits = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(y.astype("float32"))))
                predicted_objective_profits = tf.math.reduce_sum(tf.multiply(1 / 2, tf.math.square(thetas)))
                loss = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
            self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))
        else:
            with tf.GradientTape() as tape_node:
                thetas = tf.squeeze(self.model(X, training=True))
                assert len(thetas) == len(y) and len(thetas) == len(scaled_y_hat_mean)
                correct_objective_profits = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(y.astype("float32"))))
                predicted_objective_profits = tf.math.reduce_sum(tf.multiply(thetas, tf.squeeze(scaled_y_hat_mean.astype("float32"))))
                loss = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
            self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))

    def sup_optimize(self, instances, callback):
        dataset, y = self.batch_dataset(instances)
        self.model.fit(dataset, epochs=self.args.num_training_epochs, callbacks=[callback])

    def evaluate(self, instances):
        y_pred = self.predict(instances)
        _, y = self.batch_dataset(instances)
        y_y_pred = np.split([y, y_pred], np.cumsum([len(instance["link_counts"]) for instance in instances]), axis=1)[:-1]
        return {"result": [self.loss(y_i, y_pred_i).numpy() for y_i, y_pred_i in y_y_pred]}

    def get_weights(self):
        return self.model.get_weights()

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def get_gradient(self, instance):
        input_data = tf.Variable(initial_value=tf.constant(instance["X"][0].values), trainable=True)
        with tf.GradientTape() as tape_node:
            thetas = tf.squeeze(self.model(input_data, training=True))
        gradients = tape_node.gradient(thetas, input_data)
        gradients = gradients.numpy()[:, -1]
        return gradients
