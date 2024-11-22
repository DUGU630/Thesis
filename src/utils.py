import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial import distance


def import_data():
    lines_df = pd.read_csv('../DATA/Dev/Transmission_Lines.csv')
    nodes_df = pd.read_csv('../DATA/Dev/Power_Nodes.csv')
    wind_df = pd.read_csv(
        '../DATA/Dev/Availability_Factors/AvailabilityFactors_Wind_Onshore_2020.csv')
    solar_df = pd.read_csv(
        '../DATA/Dev/Availability_Factors/AvailabilityFactors_Solar_2020.csv')

    return lines_df, nodes_df, wind_df, solar_df


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    """
    R = 6371.0  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


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
        print(f'  - Duration Curves: A dictionary with keys for each time series type in {
              self.time_series_dict.keys()}')
        print("    and values as the duration curve of the time series.")
        print(
            "  - Correlation: A dictionary with keys as tuples of types of time series")
        print("    and values as correlation factors between those time series.")

    def compute_node_features(self):
        """
        Computes features for each node including position and correlations between time series types.
        Returns a dictionary of node features.
        """
        features = {}
        for node in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'time_series': {key: ts.iloc[:, node].values for key, ts in self.time_series_dict.items()},
                'duration_curves': {key: np.flip(np.sort(ts.iloc[:, node].values.copy())) for key, ts in self.time_series_dict.items()}
            }

            correlation = {}
            processed_pairs = set()
            for key1, ts1 in self.time_series_dict.items():
                for key2, ts2 in self.time_series_dict.items():
                    if key1 != key2:
                        pair = tuple(sorted([key1, key2]))
                        if pair not in processed_pairs:
                            correlation[pair] = np.corrcoef(
                                ts1.iloc[:, node], ts2.iloc[:, node])[0, 1]
                            processed_pairs.add(pair)
            node_features['correlation'] = correlation

            features[node] = node_features
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
        print(f"  - Duration Curves:")
        for key, series in node_features['duration_curves'].items():
            print(f"    {key}: {series}")
        print(f"  - Correlations:")
        for pair, corr in node_features['correlation'].items():
            print(f"    {pair}: {corr}")


class Visualization:
    def __init__(self, network, u_result, n_repr, save_fig=False, save_dir=None):
        """
        Initialize the visualization with the network, optimizer, and aggregation results.
        """
        self.network = network
        self.u_result = u_result
        self.nodes_df = self.network.nodes_df
        # self.n_repr = self.u_result.shape[1]
        self.n_repr = n_repr
        self.save_fig = save_fig
        if self.save_fig:
            if save_dir is None:
                self.save_dir = '../results/'
            else:
                self.save_dir = os.path.join('../results/', save_dir)

            if os.path.exists(self.save_dir) is False:
                os.makedirs(self.save_dir)

    def save_figure(self, fig, fig_name):
        """
        Save the figure to the specified directory.
        """
        filepath = os.path.join(self.save_dir, fig_name)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved as {fig_name} at {self.save_dir}")

    # def plot_map(self):
    #     """
    #     Plot the map of nodes with representative nodes highlighted.
    #     """

    #     # Plot the nodes
    #     plt.figure(figsize=(7, 7))
    #     for i in range(len(self.nodes_df)):
    #         plt.plot(self.nodes_df.iloc[i]['Lon'],
    #                  self.nodes_df.iloc[i]['Lat'], 'bo')
    #         plt.text(self.nodes_df.iloc[i]['Lon'],
    #                  self.nodes_df.iloc[i]['Lat'], f"{i}")

    #     # Plot the representative nodes and lines between original nodes and their representatives
    #     for j in range(self.n_repr):
    #         for i in range(len(self.nodes_df)):
    #             if self.u_result[i, j] == 1:
    #                 # Plot the representative node
    #                 plt.plot(self.nodes_df.iloc[i]['Lon'],
    #                          self.nodes_df.iloc[i]['Lat'], 'ro')
    #                 # Find the representative node coordinates
    #                 rep_node_coords = (
    #                     self.nodes_df.iloc[j]['Lon'], self.nodes_df.iloc[j]['Lat'])
    #                 # Plot a line between the original node and its representative node
    #                 plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]],
    #                          [self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

    #     plt.xlabel("Longitude")
    #     plt.ylabel("Latitude")
    #     plt.title("Node Aggregation Map")
    #     plt.show()

    #     if self.save_fig:
    #         self.save_figure(f"node_aggregation_map_{
    #             len(self.nodes_df)}_to_{self.n_repr}.png")

    def plot_map(self, method='optimization', kmeans_cluster_centers=None, kmedoids_medoids=None, labels=None, AggregationClustering=None, cluster_mapping=None):
        """
        Plot the map of nodes with representative nodes highlighted.

        Parameters:
        method (str): The method used for clustering ('optimization', 'kmeans', 'kmedoids').
        """

        # Plot the nodes
        fig = plt.figure(figsize=(7, 7))
        for i in range(len(self.nodes_df)):
            plt.plot(self.nodes_df.iloc[i]['Lon'],
                     self.nodes_df.iloc[i]['Lat'], 'bo')
            plt.text(self.nodes_df.iloc[i]['Lon'],
                     self.nodes_df.iloc[i]['Lat'], f"{i}")

        if method == 'optimization':
            # Plot the representative nodes and lines between original nodes and their representatives
            for j in range(self.u_result.shape[1]):
                for i in range(len(self.nodes_df)):
                    if self.u_result[i, j] == 1:
                        # Plot the original node
                        plt.plot(
                            self.nodes_df.iloc[i]['Lon'], self.nodes_df.iloc[i]['Lat'], 'ro')
                        # Find the representative node coordinates
                        rep_node_coords = (
                            self.nodes_df.iloc[j]['Lon'], self.nodes_df.iloc[j]['Lat'])
                        # Plot the representative node in blue
                        plt.plot(rep_node_coords[0], rep_node_coords[1], 'bo')
                        # Plot a line between the original node and its representative node
                        plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]], [
                                 self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

        elif method == 'kmeans':
            # Plot the centroids as new nodes
            for j, centroid in enumerate(kmeans_cluster_centers):
                # Assuming centroid is in (Lat, Lon) format
                plt.plot(centroid[1], centroid[0], 'ro')
                for i in range(len(self.nodes_df)):
                    if labels[i] == j:
                        # Plot a line between the original node and its centroid
                        plt.plot([self.nodes_df.iloc[i]['Lon'], centroid[1]],
                                 [self.nodes_df.iloc[i]['Lat'], centroid[0]], 'k-')

        # elif method == 'kmeans':
        #     # Find the closest actual node to each centroid in the feature space
        #     feature_matrix = AggregationClustering.compute_feature_matrix()
        #     closest_nodes = []
        #     for centroid in kmeans_cluster_centers:
        #         distances = distance.cdist(
        #             [centroid], feature_matrix, 'euclidean')
        #         closest_node = np.argmin(distances)
        #         closest_nodes.append(closest_node)

        #     # Plot the representative nodes and lines between original nodes and their representatives
        #     for j in range(self.n_repr):
        #         rep_node_coords = (
        #             self.nodes_df.iloc[closest_nodes[j]]['Lon'], self.nodes_df.iloc[closest_nodes[j]]['Lat'])
        #         plt.plot(rep_node_coords[0], rep_node_coords[1], 'ro')
        #         for i in range(len(self.nodes_df)):
        #             if labels[i] == j:
        #                 # Plot a line between the original node and its representative node
        #                 plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]],
        #                          [self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

        elif method == 'kmedoids':
            # Plot the representative nodes and lines between original nodes and their representatives
            for j in range(self.n_repr):
                rep_node_coords = (
                    self.nodes_df.iloc[kmedoids_medoids[j]]['Lon'], self.nodes_df.iloc[kmedoids_medoids[j]]['Lat'])
                plt.plot(rep_node_coords[0], rep_node_coords[1], 'ro')
                for node in cluster_mapping[j]:
                    # Plot a line between the original node and its representative node
                    plt.plot([self.nodes_df.iloc[node]['Lon'], rep_node_coords[0]], [
                             self.nodes_df.iloc[node]['Lat'], rep_node_coords[1]], 'k-')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Node Aggregation Map")
        plt.tight_layout()
        plt.show()

        if self.save_fig:
            if method == 'optimization':
                self.save_figure(fig, f"opti_agg_map_{
                    len(self.nodes_df)}_to_{self.u_result.shape[1]}.png")
            elif method == 'kmeans':
                self.save_figure(fig, f"kmeans_agg_map_{
                    len(self.nodes_df)}_to_{self.n_repr}.png")
            elif method == 'kmedoids':
                self.save_figure(fig, f"kmedoids_agg_map_{
                    len(self.nodes_df)}_to_{self.n_repr}.png")
