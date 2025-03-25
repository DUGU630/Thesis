import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import xarray as xr
from scipy.spatial import distance

class Config:
    """
    Configuration management for the script.
    """
    def __init__(self, year=2013, time_scale="monthly", total_demand=1, k_interpolation=3):
        self.year = year
        self.time_scale = time_scale
        self.total_demand = total_demand
        self.k_interpolation = k_interpolation

class Utils:
    """
    Utility functions for spatial calculations and data processing.
    """
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points on Earth's surface."""
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def custom_interpolate(points, values, new_points, k):
        """Interpolate values at new points based on nearest neighbors."""
        interpolated_values = []
        for new_point in new_points:
            distances = [Utils.haversine(new_point[0], new_point[1], p[0], p[1]) for p in points]
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = np.array(distances)[nearest_indices]
            nearest_values = np.array(values)[:, nearest_indices]
            weights = 1 / nearest_distances
            weights /= weights.sum()
            interpolated_values.append(np.dot(nearest_values, weights))
        return np.column_stack(interpolated_values)

class DataProcessor:
    """
    Class to handle data import, cleaning, and interpolation.
    """
    @staticmethod
    def import_and_interpolate_data(config, drop_duplicates=True):
        """Import and process data for nodes and capacity factors."""
        nodes_df = pd.read_csv("../DATA/dev/NewEngland-HVbuses.csv")
        demand_df = pd.read_csv(f"../DATA/dev/demand_hist/county_demand_local_hourly_{config.year}.csv")
        if drop_duplicates:
            nodes_df = nodes_df.drop_duplicates(subset=['Lat', 'Lon'])

        wind_nc = xr.open_dataset(f'../DATA/dev/CapacityFactors_ISONE/Wind/cf_Wind_0.22m_{config.year}.nc')['cf']
        solar_nc = xr.open_dataset(f'../DATA/dev/CapacityFactors_ISONE/Solar/cf_Solar_0.22m_{config.year}.nc')['cf']

        new_points = np.column_stack((nodes_df['Lat'], nodes_df['Lon']))

        wind_data = wind_nc.stack(z=("lat", "lon")).dropna('z', how='all')
        solar_data = solar_nc.stack(z=("lat", "lon")).dropna('z', how='all')

        wind_interpolated = Utils.custom_interpolate(
            np.column_stack((wind_data.lat.values, wind_data.lon.values)),
            wind_data.values, new_points, config.k_interpolation
        )
        solar_interpolated = Utils.custom_interpolate(
            np.column_stack((solar_data.lat.values, solar_data.lon.values)),
            solar_data.values, new_points, config.k_interpolation
        )

        return nodes_df, demand_df, pd.DataFrame(wind_interpolated), pd.DataFrame(solar_interpolated)

class Network:
    """
    Class to represent the network of nodes and time series data.
    """
    def __init__(self, nodes_df, demand_df, time_series_dict, config, lines_df=None, time_horizon=None):
        self.config = config
        self.nodes_df = nodes_df
        self.demand_df = demand_df
        self.time_series_dict = time_series_dict
        self.lines_df = lines_df
        self.time_horizon = time_horizon or min(ts.shape[0] for ts in time_series_dict.values())
        self.features = self.compute_node_features()

    def compute_node_features(self):
        """Compute features for each node, including position and time series statistics."""
        features = {}
        if self.config.total_demand:
            demand = self.demand_df.sum(axis=1).values
        else:
            new_points = np.column_stack((self.nodes_df['Lat'], self.nodes_df['Lon']))
            demand_lat_lon = pd.read_csv('../DATA/Dev/new_england_counties2019.csv')
            demand = Utils.custom_interpolate(
                np.column_stack((demand_lat_lon['Lat'], demand_lat_lon['Lon'])),
                self.demand_df.values, new_points, self.config.k_interpolation
            )

        for node in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'time_series': {key: ts.iloc[:, node].values for key, ts in self.time_series_dict.items()},
                'duration_curves': {key: np.sort(ts.iloc[:, node].values)[::-1] for key, ts in self.time_series_dict.items()},
                'ramp_duration_curves': {key: np.sort(np.diff(ts.iloc[:, node].values))[::-1] for key, ts in self.time_series_dict.items()}
            }
            features[node] = node_features

        return features

# Example usage
if __name__ == "__main__":
    config = Config(year=2013, time_scale="monthly", total_demand=1, k_interpolation=3)
    nodes_df, demand_df, wind_df, solar_df = DataProcessor.import_and_interpolate_data(config)
    time_series_dict = {"wind": wind_df, "solar": solar_df}
    network = Network(nodes_df, demand_df, time_series_dict, config)
    print(network.features)



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
