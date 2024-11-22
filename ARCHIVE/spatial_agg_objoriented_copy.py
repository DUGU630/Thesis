import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB


class Network:
    def __init__(self, nodes_df, time_series_dict, lines_df, time_horizon=None, with_correration=False):
        """
        Initializes a network with nodes, time series, and line data.
        - nodes_df: DataFrame with columns ['node_num', 'Lat', 'Lon', ...] for node properties.
        - time_series_dict: Dictionary with {feature_name: DataFrame} where each DataFrame has
                            rows as time steps and columns as nodes.
        - lines_df: DataFrame describing the connectivity of nodes.
        - time_horizon: Optional time horizon (int). Defaults to shortest time series length.
        """
        self.nodes_df = nodes_df
        self.lines_df = lines_df
        self.time_series_dict = time_series_dict
        self.with_correration = with_correration
        if time_horizon is None:
            self.time_horizon = min(ts.shape[0]
                                    for ts in time_series_dict.values())
        else:
            self.time_horizon = time_horizon

        # Validate and trim time series to the time horizon
        for key in self.time_series_dict:
            self.time_series_dict[key] = self.time_series_dict[key].iloc[:self.time_horizon, :]

        self.features = self.compute_node_features()

        # Print a clean message about the features dictionary
        print(
            "The 'features' dictionary has been created and can be accessed as '.features'")
        print(f"It is a dictionary with keys for each node in {
              range(len(nodes_df))}.")
        print("Each value is a dictionary with the features of that node.")
        print("\nExample structure:")
        print(f"network.features[0].keys() = {self.features[0].keys()}")
        print("\nDetails:")
        print("  - Position: A tuple (latitude, longitude) of that node.")
        print(f"  - Time series: A dictionary with keys for each time series type in {
              self.time_series_dict.keys()}")
        print("    and values as the time series itself.")
        if self.with_correration:
            print(
                "  - Correlation: A dictionary with keys as pairs of tuples (type of time series, node)")
            print(
                "    and values as the time series (ts_type1_node1 - mean1) * (ts_type2_node2 - mean2).")
            print("\nExample correlation keys:")
            example_keys = list(self.features[0]['correlation'].keys())[:3]
            for key in example_keys:
                print(f"  {key}")
            print("  ...")

    def compute_node_features(self):
        """
        Computes features for each node including position and correlations between time series types.
        Returns a dictionary of node features.
        """
        features = {}
        for node1 in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node1]['Lat'], self.nodes_df.iloc[node1]['Lon']),
                'time_series': {key: ts.iloc[:, node1].values for key, ts in self.time_series_dict.items()}
            }

            # Compute correlation matrices between different time series types
            correlation_time_series = {}
            processed_pairs = set()

            if self.with_correration:
                # Precompute means for all time series
                means = {key: ts.mean(axis=0)
                         for key, ts in self.time_series_dict.items()}
                for key1, ts1 in self.time_series_dict.items():
                    for key2, ts2 in self.time_series_dict.items():
                        if key1 != key2:
                            for node2 in range(len(self.nodes_df)):
                                if node1 != node2:
                                    # Create a sorted tuple of the keys and nodes to avoid repetition
                                    pair = tuple(
                                        sorted([(key1, node1), (key2, node2)]))
                                    if pair not in processed_pairs:
                                        # Get precomputed means
                                        mean_ts1_node1 = means[key1].iloc[node1]
                                        mean_ts2_node2 = means[key2].iloc[node2]

                                        # Compute centered time series
                                        centered_ts1 = ts1.iloc[:,
                                                                node1] - mean_ts1_node1
                                        centered_ts2 = ts2.iloc[:,
                                                                node2] - mean_ts2_node2

                                        # Compute correlation using vectorized operations
                                        # numerator = np.sum(centered_ts1 * centered_ts2)
                                        # denominator = np.sqrt(np.sum(centered_ts1**2) * np.sum(centered_ts2**2))
                                        # correlation = numerator / denominator
                                        correlation = centered_ts1 * centered_ts2

                                        # Store the correlation
                                        correlation_time_series[pair] = correlation

                                        # Add the pair to the set of processed pairs
                                        processed_pairs.add(pair)
                node_features['correlation'] = correlation_time_series

            features[node1] = node_features
        return features

    def display_node_features(self, node_index):
        """
        Displays the features of a specified node in a readable format.
        """
        node_features = self.features[node_index]
        print(f"Node {node_index} Features:")
        print(f"  - Position: {node_features['position']}")
        print(f"  - Time Series:")
        for key, series in node_features['time_series'].items():
            print(f"    {key}: {series}")
        if self.with_correration:
            print(f"  - Correlations:")
            for pair, corr in node_features['correlation'].items():
                print(f"    {pair}: {corr}")

    def duration_curve_features(self):
        """
        Updates self.feature chnaging the time series into duration curve features for each node.
        """
        for node, node_features in self.features.items():
            for key, ts in node_features['time_series'].items():
                np.flip(np.sort(ts))

        if self.with_correration:
            print("Duration curve for correlation not yet developped.")


class AggregationOptimizer:
    def __init__(self, nodes_df, time_series_dict, n_repr, n_total):
        """
        Initialize the optimizer with nodes, time series, and aggregation parameters.
        - nodes_df: DataFrame with ['node_num', 'Lat', 'Lon', ...] for node properties.
        - time_series_dict: Dictionary {feature_name: DataFrame} of time series data.
        - n_repr: Number of representative nodes to select.
        - n_total: Total weight to distribute across nodes.
        """
        self.nodes_df = nodes_df
        self.time_series_dict = time_series_dict
        self.n_repr = n_repr
        self.n_total = n_total
        self.num_nodes = len(nodes_df)
        self.features = self.compute_features()

    def compute_features(self):
        """
        Compute features for each node: positions and duration curves for each time series.
        """
        features = {}
        for node in range(self.num_nodes):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'duration_curves': {key: np.flip(np.sort(ts.iloc[:, node].values))
                                    for key, ts in self.time_series_dict.items()}
            }
            features[node] = node_features
        return features

    def compute_bins(self, num_bins=10):
        """
        Compute duration curve bins for all nodes and time series.
        Returns A_cbi, a dictionary of bin fractions for each feature and node.
        """
        A_cbi = {}
        for feature, ts in self.time_series_dict.items():
            A_cbi[feature] = {}
            for node in range(self.num_nodes):
                duration_curve = np.flip(np.sort(ts.iloc[:, node].values))
                bin_edges = np.linspace(
                    0, len(duration_curve), num_bins + 1, dtype=int)
                A_cbi[feature][node] = [
                    np.mean(duration_curve[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)
                ]
        return A_cbi

    def optimize(self):
        """
        Formulate and solve the optimization model.
        """
        # Compute bin fractions
        num_bins = 10
        A_cbi = self.compute_bins(num_bins=num_bins)

        # Create the optimization model
        model = Model("Node Aggregation")

        # Decision variables
        u = model.addVars(self.num_nodes, vtype=GRB.BINARY,
                          name="u")  # Node selection
        w = model.addVars(self.num_nodes, vtype=GRB.CONTINUOUS,
                          lb=0, name="w")  # Weights
        error = model.addVars(len(self.time_series_dict),
                              num_bins, vtype=GRB.CONTINUOUS, lb=0, name="error")

        # Objective: Minimize total error
        model.setObjective(error.sum(), GRB.MINIMIZE)

        # Constraints
        # 1. Number of representative nodes
        model.addConstr(u.sum() == self.n_repr,
                        name="Number_of_Representatives")

        # 2. Weight assignment and node selection coupling
        for i in range(self.num_nodes):
            model.addConstr(w[i] <= u[i] * self.n_total,
                            name=f"Weight_Coupling_Node_{i}")

        # 3. Total weight distribution
        model.addConstr(w.sum() == self.n_total, name="Total_Weight")

        # 4. Error constraints for each feature and bin
        for c_idx, feature in enumerate(self.time_series_dict.keys()):
            for b in range(num_bins):
                original_bin_fraction = sum(
                    A_cbi[feature][node][b] for node in range(self.num_nodes))
                aggregated_bin_fraction = sum(
                    w[node] / self.n_total * A_cbi[feature][node][b] for node in range(self.num_nodes)
                )
                model.addConstr(
                    error[c_idx, b] >= original_bin_fraction - aggregated_bin_fraction, name=f"Error_Pos_{feature}_{b}"
                )
                model.addConstr(
                    error[c_idx, b] >= aggregated_bin_fraction - original_bin_fraction, name=f"Error_Neg_{feature}_{b}"
                )

        # Solve the model
        model.optimize()

        # Extract results
        if model.status == GRB.OPTIMAL:
            selected_nodes = [i for i in range(self.num_nodes) if u[i].X > 0.5]
            weights = {i: w[i].X for i in range(self.num_nodes)}
            print("Optimal Solution Found!")
            print(f"Selected Nodes: {selected_nodes}")
            print(f"Weights: {weights}")
            return selected_nodes, weights
        else:
            print("No Optimal Solution Found.")
            return None, None


# class AggregationOptimizer:
#     def __init__(self, network, target_node_count):
#         """
#         Initializes the optimizer for node aggregation.
#         - network: The Network object containing node data and features.
#         - target_node_count: The target number of aggregated nodes, N'.
#         """
#         self.network = network
#         self.N = len(network.nodes_df)
#         self.N_prime = target_node_count
#         self.model = Model("node_aggregation")
#         self.model.Params.NonConvex = 2

#         # Variables for optimization
#         self.alpha = self.model.addVars(
#             self.N, self.N_prime, vtype=GRB.BINARY, name="alpha")
#         self.agg_features = {}

#     def optimize_aggregation(self):
#         """
#         Sets up and optimizes the aggregation model to minimize the error in time series and spatial distance.
#         """
#         time_series_features = self.network.time_series_dict.keys()
#         T = self.network.time_horizon
#         coords_agg = self.model.addVars(
#             self.N_prime, 2, vtype=GRB.CONTINUOUS, name="coords_agg")

#         # Create variables for aggregated time series
#         for feature_name in time_series_features:
#             self.agg_features[feature_name] = self.model.addVars(
#                 self.N_prime, T, vtype=GRB.CONTINUOUS, name=f"{feature_name}_agg", lb=0
#             )

#         # Objective components
#         # NRMSE
#         NRMSE_av = 0
#         for n in range(self.N):
#             for feature_name, time_series_df in self.network.time_series_dict.items():
#                 max_val = time_series_df.iloc[:, n].max()
#                 min_val = time_series_df.iloc[:, n].min()
#                 norm_factor = max_val - min_val if max_val != min_val else 1
#                 squared_error = (1 / T) * sum(
#                     (time_series_df.iloc[t, n] -
#                      self.agg_features[feature_name][n_prime, t])**2
#                     for t in range(T) for n_prime in range(self.N_prime)
#                 )
#                 NRMSE_av += self.alpha[n, n_prime] * \
#                     squared_error / norm_factor

#         # Spatial distance error
#         spatial_distance_error = 0
#         for n in range(self.N):
#             for n_prime in range(self.N_prime):
#                 lat_lon_n = np.array(
#                     [self.network.nodes_df.iloc[n]['Lat'], self.network.nodes_df.iloc[n]['Lon']])
#                 spatial_distance_error += self.alpha[n, n_prime] * sum(
#                     (lat_lon_n - coords_agg[n_prime, dim]) ** 2 for dim in range(2)
#                 )

#         # Define the objective and constraints
#         self.model.setObjective(
#             NRMSE_av + spatial_distance_error, GRB.MINIMIZE)
#         self.model.addConstrs(
#             (self.alpha.sum(n, '*') == 1 for n in range(self.N)), "assignment")

#         # Optimize the model
#         self.model.optimize()
#         self._extract_results()

#     def _extract_results(self):
#         """
#         Extracts and stores the results after optimization, including aggregated features and coordinates.
#         """
#         self.aggregated_nodes = {
#             'positions': [(self.model.getVarByName(f'coords_agg[{n_prime},0]').X,
#                            self.model.getVarByName(f'coords_agg[{n_prime},1]').X)
#                           for n_prime in range(self.N_prime)],
#             'features': {feature: [self.agg_features[feature][n_prime, t].X
#                                    for t in range(self.network.time_horizon)]
#                          for feature in self.agg_features for n_prime in range(self.N_prime)}
#         }

#     def display_aggregated_network(self):
#         """
#         Prints the aggregated network features in a readable format.
#         """
#         print("Aggregated Network Features:")
#         for n_prime, pos in enumerate(self.aggregated_nodes['positions']):
#             print(f"Aggregated Node {n_prime}:")
#             print(f"  Position: {pos}")
#             for feature_name, feature_values in self.aggregated_nodes['features'].items():
#                 print(f"  {feature_name}: {feature_values[n_prime]}")

# # Example usage
# nodes_df = pd.DataFrame({
#     'node_num': [0, 1, 2],
#     'Lat': [42.642711, 42.479477, 42.331960],
#     'Lon': [-70.865107, -71.396507, -71.020173],
# })
# lines_df = pd.DataFrame(columns=['from_node', 'to_node'])  # Placeholder for lines
# time_series_dict = {
#     'load': pd.DataFrame(np.random.rand(24, 3)),
#     'wind_capacity': pd.DataFrame(np.random.rand(24, 3))
# }

# network = Network(nodes_df, time_series_dict, lines_df)
# network.display_node_features(0)

# optimizer = AggregationOptimizer(network, target_node_count=2)
# optimizer.optimize_aggregation()
# optimizer.display_aggregated_network()
