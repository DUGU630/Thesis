import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB


class Network:
    def __init__(self, nodes_df, time_series_dict, lines_df, time_horizon=None):
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
        self.time_horizon = time_horizon or min(
            ts.shape[0] for ts in time_series_dict.values())

        # Validate and trim time series to the time horizon
        for key in self.time_series_dict:
            self.time_series_dict[key] = self.time_series_dict[key].iloc[:self.time_horizon, :]

        self.features = self._compute_node_features()

    def _compute_node_features(self):
        """
        Computes features for each node including position and correlations between time series types.
        Returns a dictionary of node features.
        """
        features = {}
        for node in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'time_series': {key: ts.iloc[:, node].values for key, ts in self.time_series_dict.items()}
            }

            # Compute correlation matrices between different time series types
            correlation_matrices = {}
            for key1, ts1 in self.time_series_dict.items():
                for key2, ts2 in self.time_series_dict.items():
                    if key1 != key2:
                        correlation = np.corrcoef(
                            ts1.iloc[:, node], ts2.iloc[:, node])[0, 1]
                        correlation_matrices[(key1, key2)] = correlation
            node_features['correlation_matrices'] = correlation_matrices

            features[node] = node_features
        return features

    def display_node_features(self, node_index):
        """
        Displays the features of a specified node in a readable format.
        """
        node_features = self.features[node_index]
        print(f"Node {node_index} Features:")
        print(f"  Position: {node_features['position']}")
        print(f"  Time Series:")
        for key, series in node_features['time_series'].items():
            print(f"    {key}: {series}")
        print(f"  Correlations:")
        for (key1, key2), corr in node_features['correlation_matrices'].items():
            print(f"    {key1}-{key2}: {corr}")


class AggregationOptimizer:
    def __init__(self, network, target_node_count):
        """
        Initializes the optimizer for node aggregation.
        - network: The Network object containing node data and features.
        - target_node_count: The target number of aggregated nodes, N'.
        """
        self.network = network
        self.N = len(network.nodes_df)
        self.N_prime = target_node_count
        self.model = Model("node_aggregation")
        self.model.Params.NonConvex = 2

        # Variables for optimization
        self.alpha = self.model.addVars(
            self.N, self.N_prime, vtype=GRB.BINARY, name="alpha")
        self.agg_features = {}

    def optimize_aggregation(self):
        """
        Sets up and optimizes the aggregation model to minimize the error in time series and spatial distance.
        """
        time_series_features = self.network.time_series_dict.keys()
        T = self.network.time_horizon
        coords_agg = self.model.addVars(
            self.N_prime, 2, vtype=GRB.CONTINUOUS, name="coords_agg")

        # Create variables for aggregated time series
        for feature_name in time_series_features:
            self.agg_features[feature_name] = self.model.addVars(
                self.N_prime, T, vtype=GRB.CONTINUOUS, name=f"{feature_name}_agg", lb=0
            )

        # Objective components
        # NRMSE
        NRMSE_av = 0
        for n in range(self.N):
            for feature_name, time_series_df in self.network.time_series_dict.items():
                max_val = time_series_df.iloc[:, n].max()
                min_val = time_series_df.iloc[:, n].min()
                norm_factor = max_val - min_val if max_val != min_val else 1
                squared_error = (1 / T) * sum(
                    (time_series_df.iloc[t, n] -
                     self.agg_features[feature_name][n_prime, t])**2
                    for t in range(T) for n_prime in range(self.N_prime)
                )
                NRMSE_av += self.alpha[n, n_prime] * \
                    squared_error / norm_factor

        # Spatial distance error
        spatial_distance_error = 0
        for n in range(self.N):
            for n_prime in range(self.N_prime):
                lat_lon_n = np.array(
                    [self.network.nodes_df.iloc[n]['Lat'], self.network.nodes_df.iloc[n]['Lon']])
                spatial_distance_error += self.alpha[n, n_prime] * sum(
                    (lat_lon_n - coords_agg[n_prime, dim]) ** 2 for dim in range(2)
                )

        # Define the objective and constraints
        self.model.setObjective(
            NRMSE_av + spatial_distance_error, GRB.MINIMIZE)
        self.model.addConstrs(
            (self.alpha.sum(n, '*') == 1 for n in range(self.N)), "assignment")

        # Optimize the model
        self.model.optimize()
        self._extract_results()

    def _extract_results(self):
        """
        Extracts and stores the results after optimization, including aggregated features and coordinates.
        """
        self.aggregated_nodes = {
            'positions': [(self.model.getVarByName(f'coords_agg[{n_prime},0]').X,
                           self.model.getVarByName(f'coords_agg[{n_prime},1]').X)
                          for n_prime in range(self.N_prime)],
            'features': {feature: [self.agg_features[feature][n_prime, t].X
                                   for t in range(self.network.time_horizon)]
                         for feature in self.agg_features for n_prime in range(self.N_prime)}
        }

    def display_aggregated_network(self):
        """
        Prints the aggregated network features in a readable format.
        """
        print("Aggregated Network Features:")
        for n_prime, pos in enumerate(self.aggregated_nodes['positions']):
            print(f"Aggregated Node {n_prime}:")
            print(f"  Position: {pos}")
            for feature_name, feature_values in self.aggregated_nodes['features'].items():
                print(f"  {feature_name}: {feature_values[n_prime]}")

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
