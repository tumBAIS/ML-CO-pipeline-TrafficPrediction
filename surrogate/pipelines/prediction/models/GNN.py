import tensorflow as tf
import numpy as np
import pandas as pd
from surrogate.pipelines.prediction.models.NN import NN, create_ffn
from surrogate.pipelines.prediction import features


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units):
        super(GraphConvLayer, self).__init__()
        self.ffn_prepare = create_ffn(hidden_units)
        self.update_fn = create_ffn(hidden_units)

    def prepare(self, node_repesentations, edge_features):
        input_features = tf.concat([edge_features, node_repesentations], axis=1)
        messages = self.ffn_prepare(input_features)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        num_nodes = tf.shape(node_repesentations)[0]
        aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(self, inputs, **kwargs):
        node_repesentations, edges, edge_features = inputs
        neighbour_indices, node_indices = edges[0], edges[1]
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)
        neighbour_messages = self.prepare(neighbour_repesentations, edge_features)
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)
        return self.update(node_repesentations, aggregated_messages)


class GraphConvLayerFinal(tf.keras.layers.Layer):
    def __init__(self, hidden_units):
        super().__init__()
        self.ffn_prepare = create_ffn(hidden_units)
        self.final_ff = create_ffn(hidden_units)

    def update_edges(self, aggregated_messages):
        node_embeddings = self.final_ff(aggregated_messages)
        return node_embeddings

    def prepare(self, node_repesentations, edge_features):
        input_features = tf.concat([edge_features, node_repesentations], axis=1)
        messages = self.ffn_prepare(input_features)
        return messages

    def call(self, inputs, **kwargs):
        node_repesentations, edges, edge_features = inputs
        neighbour_indices, node_indices = edges[0], edges[1]
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)
        neighbour_messages = self.prepare(neighbour_repesentations, edge_features)
        return self.update_edges(neighbour_messages)


class GNN(NN):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_units_embeddings = [20, 20]

    def get_input_dimension(self, X):
        edge_features_dim, node_features_dim = X[0].shape[1], X[1].shape[1]
        return edge_features_dim, node_features_dim

    def create_model(self, X):
        input_dimension_edges, input_dimension_nodes = self.get_input_dimension(X)
        # Create a process layer.
        preprocess = create_ffn(self.hidden_units_embeddings)
        # Create the first GraphConv layer.
        conv1 = GraphConvLayer(self.hidden_units_embeddings)
        # Create the second GraphConv layer.
        conv2 = GraphConvLayer(self.hidden_units_embeddings)
        # Create the third GraphConv layer.
        conv3 = GraphConvLayer(self.hidden_units_embeddings)
        postprocess = GraphConvLayerFinal(self.hidden_units)
        # Create an output layer.
        pre_output_layer_structured = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, use_bias=False)
        output_layer_supervised = tf.keras.layers.Dense(1, use_bias=False)

        edge_features = tf.keras.layers.Input((input_dimension_edges), dtype="float32", name="edge_features")
        node_features = tf.keras.layers.Input((input_dimension_nodes), dtype="float32", name="node_features")
        edges = tf.keras.layers.Input((2), dtype="int32", name="edges")

        edges_trans = tf.transpose(edges)
        # Preprocess the node_features to produce node representations.
        x = preprocess(node_features)
        # Apply the first graph conv layer.
        x1 = conv1((x, edges_trans, edge_features))
        # Apply the second graph conv layer.
        x2 = conv2((x1, edges_trans, edge_features))
        # Apply the third graph conv layer.
        x3 = conv3((x2, edges_trans, edge_features))
        # Postprocess node embedding.
        edge_embeddings = postprocess((x3, edges_trans, edge_features))
        # Process output
        if self.args.learning in ["structured", "structured_wardrop"]:
            pre_output = pre_output_layer_structured(edge_embeddings)
            y = tf.multiply(pre_output, -1)
        elif self.args.learning == "supervised":
            y = output_layer_supervised(edge_embeddings)

        if self.args.learning in ["structured"] and self.args.co_optimizer in ["wardropequilibria"]:
            # Create a process layer.
            preprocess_wardrop = create_ffn(self.hidden_units_embeddings)
            # Create the first GraphConv layer.
            conv1_wardrop = GraphConvLayer(self.hidden_units_embeddings)
            # Create the second GraphConv layer.
            conv2_wardrop = GraphConvLayer(self.hidden_units_embeddings)
            # Create the third GraphConv layer.
            conv3_wardrop = GraphConvLayer(self.hidden_units_embeddings)
            postprocess_wardrop = GraphConvLayerFinal(self.hidden_units)
            # Create an output layer.
            pre_output_layer_structured_wardrop = tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus, use_bias=False)

            # Preprocess the node_features to produce node representations.
            x_wardrop = preprocess_wardrop(node_features)
            # Apply the first graph conv layer.
            x1_wardrop = conv1_wardrop((x_wardrop, edges_trans, edge_features))
            # Apply the second graph conv layer.
            x2_wardrop = conv2_wardrop((x1_wardrop, edges_trans, edge_features))
            # Apply the third graph conv layer.
            x3_wardrop = conv3_wardrop((x2_wardrop, edges_trans, edge_features))
            # Postprocess node embedding.
            edge_embeddings_wardrop = postprocess_wardrop((x3_wardrop, edges_trans, edge_features))
            pre_output_wardrop = pre_output_layer_structured_wardrop(edge_embeddings_wardrop)
            y_wardrop = tf.multiply(pre_output_wardrop, -1)
            self.model = tf.keras.Model(
                inputs=[edge_features, node_features, edges],
                outputs=[y, y_wardrop],
            )
        else:
            self.model = tf.keras.Model(
                inputs=[edge_features, node_features, edges],
                outputs=[y],
            )

        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])

    def get_features_for_nodes(self, args, training_instance):

        feature_num_homes1 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="home", divisor=1)
        feature_num_homes2 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="home", divisor=2)
        feature_num_homes5 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="home", divisor=5)
        feature_num_homes10 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="home", divisor=10)
        feature_num_homes15 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="home", divisor=15)

        feature_num_works1 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="work", divisor=1)
        feature_num_works2 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="work", divisor=2)
        feature_num_works5 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="work", divisor=5)
        feature_num_works10 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="work", divisor=10)
        feature_num_works15 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="work", divisor=15)

        feature_num_nodes1 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="nodes", divisor=1)
        feature_num_nodes2 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="nodes", divisor=2)
        feature_num_nodes5 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="nodes", divisor=5)
        feature_num_nodes10 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="nodes", divisor=10)
        feature_num_nodes15 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="nodes", divisor=15)

        feature_num_links1 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=1)
        feature_num_links2 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=2)
        feature_num_links5 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=5)
        feature_num_links10 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=10)
        feature_num_links15 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=15)

        feature_num_links_capacity_1000_1 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=1, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_2 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=2, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_5 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=5, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_10 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=10, attribute="link_capacity", threshold=1000)
        feature_num_links_capacity_1000_15 = features.get_num_items_around_location(self.args, data=training_instance, location="nodes", item="links", divisor=15, attribute="link_capacity", threshold=1000)

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
                          })
        X["node_id"] = training_instance["nodes_id"]
        X = X.set_index("node_id")
        return X

    def get_features_for_links(self, args, training_instance):

        training_instance["links_x"], training_instance["links_y"] = features.get_link_coordinates(data=training_instance)

        count_to_from, count_from_to, count_from_from, count_to_to = features.get_num_streets_ingoingoutgoing(data=training_instance)

        feature_link_length = training_instance["link_length"]
        feature_link_speed = training_instance["link_freespeed"]
        feature_link_capacity = training_instance["link_capacity"]
        feature_link_permlanes = training_instance["link_permlanes"]
        feature_link_time = [length / speed for length, speed in zip(training_instance["link_length"], training_instance["link_freespeed"])]

        X = pd.DataFrame({"feature_link_length": feature_link_length,
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

    def get_features_instance(self, args, training_instance, add_features=None):
        solution_representation = training_instance["solution_representation"]

        # Add features for links
        super().get_features_instance(args, training_instance, add_features=None)

        # Add features for nodes
        X_nodes = self.get_features_for_nodes(args, training_instance)
        X_nodes = solution_representation.solution_scheme_nodes.join(X_nodes, how="left", on="node_id")[X_nodes.columns]

        if args.time_variant_theta:
            X_time = pd.DataFrame({"time_feature": solution_representation.solution_scheme_time, "time": solution_representation.solution_scheme_time})
            X_time = X_time.set_index(["time"])
            X_nodes = X_nodes.join(X_time, how="left", on="time")[list(X_time.columns) + list(X_nodes.columns)]

        edges = pd.DataFrame({"from": solution_representation.solution_scheme["new_link_from"], "to": solution_representation.solution_scheme["new_link_to"]})
        #edges = pd.DataFrame({"from": training_instance["link_from"], "to": training_instance["link_to"]})
        training_instance["X"] = [training_instance["X"][0], X_nodes, edges]

    def get_standardization_divisor(self, training_instances):
        if self.divisor is None:
            divisor_links = pd.concat([training_instance["X"][0] for training_instance in training_instances]).mean()
            divisor_nodes = pd.concat([training_instance["X"][1] for training_instance in training_instances]).mean()
            self.divisor = [{key: (value if np.abs(value) > 1 else 1) for key, value in divisor_links.items()},
                            {key: (value if np.abs(value) > 1 else 1) for key, value in divisor_nodes.items()}]
        return self.divisor

    def standardize(self, training_instances):
        if self.args.standardize:
            divisor_links, divisor_nodes = self.get_standardization_divisor(training_instances)
            for training_instance in training_instances:
                training_instance["X"][0] = training_instance["X"][0] / divisor_links
                training_instance["X"][1] = training_instance["X"][1] / divisor_nodes
        return training_instances

    def batch_dataset_from_instances(self, instances):
        edge_features = [i["X"][0].values for i in instances]
        node_features = [i["X"][1].values for i in instances]
        edges = [i["X"][2].values for i in instances]
        target = [i["y"].values for i in instances]

        edge_features_tensor = tf.ragged.constant(edge_features, dtype=tf.float32)
        node_features_tensor = tf.ragged.constant(node_features, dtype=tf.float32)
        edge_tensor = tf.ragged.constant(edges, dtype=tf.int64)
        target_tensor = tf.ragged.constant(target, dtype=tf.float32)
        return (edge_features_tensor, node_features_tensor, edge_tensor), (target_tensor)

    def batch_dataset(self, instances):

        def prepare_batch(x_batch, y_batch):
            edge_features, node_features, edges = x_batch
            edges = edges.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            y_batch = y_batch.merge_dims(outer_axis=0, inner_axis=1)
            return (edge_features, node_features, edges), y_batch

        X, y = self.batch_dataset_from_instances(instances)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.batch(1).map(prepare_batch, -1).prefetch(-1), pd.concat([i["y"] for i in instances]).values
