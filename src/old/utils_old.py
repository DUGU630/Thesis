import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial import distance
import xarray as xr


def import_data():
    lines_df = pd.read_csv('../DATA/Dev/Transmission_Lines.csv')
    nodes_df = pd.read_csv('../DATA/Dev/Power_Nodes.csv')
    wind_df = pd.read_csv(
        '../DATA/Dev/Availability_Factors/AvailabilityFactors_Wind_Onshore_2020.csv')
    solar_df = pd.read_csv(
        '../DATA/Dev/Availability_Factors/AvailabilityFactors_Solar_2020.csv')

    return lines_df, nodes_df, wind_df, solar_df


def import_data_county():
    nodes_df = pd.read_csv('../DATA/Dev/new_england_counties2019.csv')
    wind_df = pd.read_csv(
        '../DATA/Dev/county-level-CFs/hist/wind/cf_local_county_2014.csv')
    solar_df = pd.read_csv(
        '../DATA/Dev/county-level-CFs/hist/solar/cf_local_county_2014.csv')

    return nodes_df, wind_df, solar_df


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


def import_and_interpolate_data(drop_duplicates = 1, k=1, year = 2013):
    """
    Imports and processes data for nodes, wind capacity factors, and solar capacity factors.
    This function reads node data from a CSV file and wind and solar capacity factor data from NetCDF files.
    It optionally removes duplicate nodes based on latitude and longitude and interpolates the capacity factor
    data to the node locations using a linear k-interpolation method.
    Parameters:
    drop_duplicates (int): Flag to indicate whether to drop duplicate nodes. 
                           0 means not dropping duplicates, 1 means dropping duplicates. Default is 1.
    k (int): The number of nearest neighbors to use for interpolation. Must be a positive integer. Default is 1.
    Returns:
    tuple: A tuple containing three pandas DataFrames:
        - nodes_df: DataFrame containing the node data.
        - wind_df: DataFrame containing the interpolated wind capacity factors.
        - solar_df: DataFrame containing the interpolated solar capacity factors.
    """
    # Validate drop_duplicates
    if drop_duplicates not in [0, 1]:
        raise ValueError("drop_duplicates must be 0 (not dropping duplicates) or 1 (dropping duplicates)")
    # Validate k
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    
    nodes_df = pd.read_csv("../DATA/dev/NewEngland-HVbuses.csv")
    demand_df = pd.read_csv(f"../DATA/dev/demand_hist/county_demand_local_hourly_{year}.csv")
    
    if drop_duplicates == 1:
        initial_count = len(nodes_df)
        nodes_df = nodes_df.drop_duplicates(subset=['Lat', 'Lon'])
        final_count = len(nodes_df)
        print(f"Number of duplicates deleted in nodes_df: {initial_count - final_count}")

    wind_nc = xr.open_dataset(f'../DATA/dev/CapacityFactors_ISONE/Wind/cf_Wind_0.22m_{year}.nc')['cf']
    solar_nc = xr.open_dataset(f'../DATA/dev/CapacityFactors_ISONE/Solar/cf_Solar_0.22m_{year}.nc')['cf']

    new_points = np.column_stack((nodes_df['Lat'], nodes_df['Lon']))

    wind_data = wind_nc.stack(z=("lat", "lon")).dropna('z', how='all')
    wind_CF = wind_data.values
    wind_lat = wind_data.lat.values
    wind_lon = wind_data.lon.values
    wind_points = np.column_stack((wind_lat, wind_lon))

    solar_data = solar_nc.stack(z=("lat", "lon")).dropna('z', how='all')
    solar_CF = solar_data.values
    solar_lat = solar_data.lat.values
    solar_lon = solar_data.lon.values
    solar_points = np.column_stack((solar_lat, solar_lon))

    # Interpolate wind CF data
    wind_interpolated = custom_interpolate(wind_points, wind_CF, new_points, k)
    wind_df = pd.DataFrame(wind_interpolated)

    # Interpolate solar CF data
    solar_interpolated = custom_interpolate(solar_points, solar_CF, new_points, k)
    solar_df = pd.DataFrame(solar_interpolated)

    return nodes_df, demand_df, wind_df, solar_df

def custom_interpolate(points, values, new_points, k=3):
    """
    Interpolates values at new points based on given points and their values using various methods.
    Parameters:
    points (list of tuples): List of (latitude, longitude) tuples representing the known points.
    values (list): List of values corresponding to the known points.
    new_points (list of tuples): List of (latitude, longitude) tuples where interpolation is desired.
    k (int, optional): Number of nearest neighbors to consider for 'k_nearest' method. Default is 3.
    Returns:
    np.ndarray: Array of interpolated values at the new points.
    """
    interpolated_values = []

    for new_point in new_points:
        distances = [haversine(new_point[0], new_point[1], point[0], point[1]) for point in points]
        sorted_indices = np.argsort(distances)       
        nearest_indices = sorted_indices[:k]
        nearest_distances = np.array(distances)[nearest_indices]
        nearest_values = np.array(values)[:, nearest_indices]
        weights = 1 / nearest_distances
        weights /= weights.sum()
        interpolated_value = np.dot(nearest_values, weights)
        interpolated_values.append(interpolated_value)
    
    return np.column_stack(interpolated_values)

class Network:
    def __init__(self, nodes_df, demand_df, time_series_dict, lines_df=None, time_horizon=None, total_demand = 1, time_scale = "monthly", year = 2013):
        """
        Initializes a network with nodes, time series, and line data.
        - nodes_df: DataFrame with columns ['node_num', 'Lat', 'Lon', ...] for node properties.
        - time_series_dict: Dictionary with {feature_name: DataFrame} where each DataFrame has
                            rows as time steps and columns as nodes.
        - lines_df: DataFrame describing the connectivity of nodes.
        - time_horizon: Optional time horizon (int). Defaults to shortest time series length.
        """
        # Validate drop_duplicates
        if total_demand not in [0, 1]:
            raise ValueError("Total_demand must be 0 (using total_demand) or 1 (using k-interpolation)")
        if time_scale not in ["weekly", "monthly", "yearly"]:
            raise ValueError("time_scale must be 'weekly', 'monthly', or 'yearly'")
        self.year = year
        self.time_scale = time_scale
        self.total_demand = total_demand
        self.demand_df = demand_df
        self.nodes_df = nodes_df
        if lines_df is not None:
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
        print(f"  - Ramp Duration Curves (RDCs): A dictionary with keys for each time series type {
              self.time_series_dict.keys()}")
        print("    and values as the RDC (found by differentiating and subsequently sorting) of the time series.")
        print("  - Correlation: A dictionary with keys as tuples of types of time series")
        print("    and values as correlation factors between those time series.")

    def compute_node_features(self):
        """
        Computes features for each node including position and correlations between time series types.
        Returns a dictionary of node features.
        """
        features = {}

        #Suply demand mismatch
        if self.total_demand == 1:
            demand = self.demand_df.sum(axis=1).values
        if self.total_demand == 0:
            new_points = np.column_stack((self.nodes_df['Lat'], self.nodes_df['Lon']))
            values = self.demand_df.values
            demand_lat_lon = pd.read_csv('../DATA/Dev/new_england_counties2019.csv')
            points = np.column_stack((demand_lat_lon['Lat'], demand_lat_lon['Lon']))
            demand = custom_interpolate(points, values, new_points, k=3)

        for node in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'time_series': {key: ts.iloc[:, node].values for key, ts in self.time_series_dict.items()},
                'duration_curves': {key: np.flip(np.sort(ts.iloc[:, node].values.copy())) for key, ts in self.time_series_dict.items()},
                'ramp_duration_curves': {key: np.flip(np.sort(np.diff(ts.iloc[:, node].values.copy()))) for key, ts in self.time_series_dict.items()}
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

            if self.total_demand == 1:
                node_features['supply_demand_mismatch'] = self.supply_demand_mismatch(node, demand, self.time_scale)
            elif self.total_demand == 0:
                node_features['supply_demand_mismatch'] = self.supply_demand_mismatch(node, demand[:, node], self.time_scale)

            features[node] = node_features
        return features
    
    def supply_demand_mismatch(self, node, demand, time_scale):
        """
        Computes the supply-demand mismatch for each node in the network.
        Returns a dictionary of supply-demand mismatches.
        """
        date_range = pd.date_range(start=f'{self.year}-01-01', periods=len(demand), freq='h')
        demand_series = pd.Series(demand, index=date_range)
        
        correlation_dict = {}
        
        for key, ts in self.time_series_dict.items():
            series = ts.iloc[:, node]
            series.index = date_range
            
            if time_scale == 'yearly':
                # Calculate yearly correlation
                correlation = series.corr(demand_series)
                correlation_dict[key] = correlation
            
            elif time_scale == 'monthly':
                # Calculate monthly correlation
                monthly_corr = series.groupby(series.index.month).apply(lambda x: x.corr(demand_series[x.index]))
                correlation_dict[key] = monthly_corr.values
            
            elif time_scale == 'weekly':
                # Calculate weekly correlation
                weekly_corr = series.groupby(series.index.isocalendar().week).apply(lambda x: x.corr(demand_series[x.index]))
                correlation_dict[key] = weekly_corr.values
            
            else:
                raise ValueError("Unsupported time scale. Choose from 'yearly', 'monthly', or 'weekly'.")
        
        return correlation_dict
    
    # def supply_demand_mismatch(self, time_scale):
    #     """
    #     Computes the supply-demand mismatch for each node in the network.
    #     Returns a dictionary of supply-demand mismatches.
    #     """
    #     date_range = pd.date_range(start=f'{self.year}-01-01', periods=len(demand), freq='H')
    #     demand_series = pd.Series(demand, index=date_range)
        
    #     # Extract the time series for the specified node
    #     for key, ts in self.time_series_dict.items():
    #         series = ts.iloc[:, node]
    #         series.index = date_range
        
    #         if time_scale == 'yearly':
    #             # Calculate yearly correlation
    #             correlation = series.corr(demand_series)
    #             return correlation
            
    #         elif time_scale == 'monthly':
    #             # Calculate monthly correlation
    #             monthly_corr = []
    #             for month in range(1, 13):
    #                 series_month = series[series.index.month == month]
    #                 corr = series_month.corr(demand_series[demand_series.index.month == month])
    #                 monthly_corr.append(corr)
    #             return np.array(monthly_corr)
            
            
    #     supply_demand_mismatch = {}
    #     if self.total_demand == 1:
    #         demand = self.demand_df.sum(axis=1).values
    #     for node in range(len(self.nodes_df)):
    #         hourly_mismatch = {key: demand - ts.iloc[:, node].values.copy() for key, ts in self.time_series_dict.items()}
    #         if time_scale == "weekly":
    #             mismatch = {key: hourly_mismatch[key].reshape(-1, 24).sum(axis=1) for key in hourly_mismatch.keys()]}
            
    #         node_demand = demand if self.total_demand == 1 else self.demand_df.iloc[:, node + 1].values
    #         node_supply = np.zeros(node_demand.shape)
    #         for key, ts in self.time_series_dict.items():
    #             node_supply += ts.iloc[:, node].values
    #         supply_demand_mismatch[node] = node_supply - node_demand

    #     return supply_demand_mismatch


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
        print(f"  - Ramp Duration Curves:")
        for key, rdc in node_features['ramp_duration_curves'].items():
            print(f"    {key}: {rdc}")
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
