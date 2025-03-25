import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# from scipy.spatial import distance
import xarray as xr
from dataclasses import dataclass, field, asdict

# @dataclass(frozen=True)
@dataclass(frozen=False)
class Config:
    year: int = 2013
    demand: str = "k-interpolation"
    k_neighbors_CF: int = 3
    k_weight_demand: int = 3
    time_scale: str = "monthly"
    drop_duplicates: bool = True
    n_repr: int = 5
    k_representative_days: int = 5
    weights: dict = field(default_factory=lambda: {
        'position': 1.0,
        'time_series': 1.0,
        'duration_curves': 1.0,
        'rdc': 1.0,
        'intra_correlation': 1.0,
        'inter_correlation': 1.0,
        'supply_demand_mismatch': 1.0
    })
    file_paths: dict = field(default_factory=lambda: {
        "nodes": "../DATA/dev/NewEngland-HVbuses.csv",
        "demand": f"../DATA/dev/demand_hist/county_demand_local_hourly_{2013}.csv",
        "wind_cf": f'../DATA/dev/CapacityFactors_ISONE/Wind/cf_Wind_0.22m_{2013}.nc',
        "solar_cf": f'../DATA/dev/CapacityFactors_ISONE/Solar/cf_Solar_0.22m_{2013}.nc',
        "demand_lat_lon": '../DATA/Dev/new_england_counties2019.csv'
    })
    

    def __post_init__(self):
        if self.demand not in ["k-interpolation", "total_demand"]:
            raise ValueError("demand must be 'total_demand' or 'k-interpolation'")
        if self.time_scale not in ["weekly", "monthly", "yearly"]:
            raise ValueError("time_scale must be 'weekly', 'monthly', or 'yearly'")
        if self.k_neighbors_CF <= 0:
            raise ValueError("k_neighbors_CF must be a positive integer")
        if self.k_weight_demand <= 0:
            raise ValueError("k_weight_demand must be a positive integer")
        if not all(weight >= 0 for weight in self.weights.values()):
            raise ValueError("All weights must be non negative.")
        if self.n_repr <= 0:
            raise ValueError("n_repr must be a positive integer.")

        # Update file paths with the correct year
        object.__setattr__(self, 'file_paths', {
            "nodes": "../DATA/dev/NewEngland-HVbuses.csv",
            "demand": f"../DATA/dev/demand_hist/county_demand_local_hourly_{self.year}.csv",
            "wind_cf": f'../DATA/dev/CapacityFactors_ISONE/Wind/cf_Wind_0.22m_{self.year}.nc',
            "solar_cf": f'../DATA/dev/CapacityFactors_ISONE/Solar/cf_Solar_0.22m_{self.year}.nc',
            "demand_lat_lon": '../DATA/Dev/new_england_counties2019.csv'
        })

    def explain_config(self):
        explanations = {
            "year": "The year considered for the capacity factors and the demand time series.",
            "demand": "The method used to calculate the demand mismatch. Options are 'total_demand' (compare supply at each node to the total demand) or 'k-interpolation' (compare supply to a weighted sum of the demand, considering k number of points).",
            "k_weight_demand": "The number of nearest demand points considered for the weighted sum of the demand when using 'k-interpolation'. Must be a positive integer and less than or equal to the total number of demand points.",
            "time_scale": "The time scale used for the supply-demand mismatch calculation. Options are 'yearly' (a single correlation number for the whole year), 'monthly' (correlation calculated for each month), or 'weekly' (correlation calculated for each week).",
            "k_neighbors_CF": "The number of nearest nodes considered for the linear interpolation of capacity factor nodes to match the network nodes. If k=1, only the nearest node is considered. If k equals the number of CF nodes, all nodes are considered, with weights linearly proportional to the distance.",
            "drop_duplicates": "A flag to indicate whether to drop duplicate nodes that have the same latitude and longitude in the nodes DataFrame.",
            "n_repr": "The number of representative nodes in the aggregated network.",
            "k_representative_days": "The number of representative days used for the time series aggregation. Must be a positive integer and less than or equal to the total number of days in the year.",
            "weights": "A dictionary containing weights for each feature in the distance matrix. Features include 'position', 'time_series', 'duration_curves', 'rdc', 'intra_correlation', 'inter_correlation', and 'supply_demand_mismatch'.",
            "file_paths": "A dictionary containing file paths for various data files. Keys include 'nodes' (CSV), 'demand' (CSV), 'wind_cf' (NetCDF), 'solar_cf' (NetCDF), and 'demand_lat_lon' (CSV).",
        }

        for key, explanation in explanations.items():
            print(f"{key}: {explanation}")


def haversine(point1, point2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def find_matching_folder(config, time_series, num_nodes1=None, num_nodes2=None):
    base_path = '../results/distance_metrics'
    
    # Convert config to a comparable format
    config_dict = asdict(config)
    config_dict = {k: str(v) for k, v in config_dict.items() if k not in ['file_paths', 'weights', 'k_representative_days', 'n_repr']}
    if config_dict.get('demand') == 'total_demand':
        del config_dict['k_weight_demand']
    time_series = sorted(time_series)

    config_str = ', \n'.join(f'{k}: {v}' for k, v in config_dict.items())
    print(f"Searching for a folder with the following configuraton:\n{config_str},\ntime series: {time_series}\n...")
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        metadata_path = os.path.join(folder_path, 'metadata.csv')

        if not os.path.isfile(metadata_path):
            print(f"Metadata file not found in '{folder_name}'")
            continue

        metadata = pd.read_csv(metadata_path)
        metadata_dict = metadata.iloc[0].to_dict()
        
        # Check node numbers if provided
        if num_nodes1 is not None and metadata_dict.get('num_nodes1') != num_nodes1:
            continue
        if num_nodes2 is not None and metadata_dict.get('num_nodes2') != num_nodes2:
            continue
        
        # Check config match
        metadata_config = {k: str(metadata_dict[k]) for k in config_dict.keys() if k in metadata_dict}
        if metadata_config != config_dict:
            continue

        # Check time_series match
        metadata_time_series = eval(metadata_dict.get('timeseries', '[]'))
        if sorted(metadata_time_series) != time_series:
            continue
        
        print(f"Found matching folder: '{folder_name}'")
        return folder_name
    
    print("No matching folder found")
    return None

class DataProcessor:
    def __init__(self, config: Config):
        self.nodes_df, self.demand_df, self.wind_df, self.solar_df, self.wind_CF, self.solar_CF = DataProcessor.import_and_interpolate_data(config)
    
    @staticmethod
    def import_and_interpolate_data(config):
        """
        Imports and processes data for nodes, wind capacity factors, and solar capacity factors.
        """
        nodes_df, demand_df, wind_data, solar_data = DataProcessor._import_data(config)

        new_points = np.column_stack((nodes_df['Lat'], nodes_df['Lon']))
    
        wind_CF = wind_data.values
        wind_lat = wind_data.lat.values
        wind_lon = wind_data.lon.values
        wind_points = np.column_stack((wind_lat, wind_lon))

        solar_CF = solar_data.values
        solar_lat = solar_data.lat.values
        solar_lon = solar_data.lon.values
        solar_points = np.column_stack((solar_lat, solar_lon))

        # Interpolate wind CF data
        wind_interpolated = DataProcessor.custom_interpolate(wind_points, wind_CF, new_points, config.k_neighbors_CF)
        wind_df = pd.DataFrame(wind_interpolated)

        # Interpolate solar CF data
        solar_interpolated = DataProcessor.custom_interpolate(solar_points, solar_CF, new_points, config.k_neighbors_CF)
        solar_df = pd.DataFrame(solar_interpolated)

        return nodes_df, demand_df, wind_df, solar_df, wind_CF, solar_CF
    
    @staticmethod
    def _import_data(config):
        nodes_df = pd.read_csv(config.file_paths["nodes"])
        demand_df = pd.read_csv(config.file_paths["demand"]).iloc[:, 1:]

        if config.drop_duplicates:
            initial_count = len(nodes_df)
            nodes_df = nodes_df.drop_duplicates(subset=['Lat', 'Lon'])
            final_count = len(nodes_df)
            print(f"Number of duplicates deleted in nodes_df: {initial_count - final_count}")

        wind_nc = xr.open_dataset(config.file_paths["wind_cf"])['cf']
        solar_nc = xr.open_dataset(config.file_paths["solar_cf"])['cf']

        wind_data = wind_nc.stack(z=("lat", "lon")).dropna('z', how='all')
        solar_data = solar_nc.stack(z=("lat", "lon")).dropna('z', how='all')
    
        return nodes_df, demand_df, wind_data, solar_data     

    @staticmethod
    def custom_interpolate(points, values, new_points, k=3):
        """
        Interpolates values at new points based on given points and their values using k linear interpolation.
        """
        interpolated_values = []
        distances = cdist(new_points, points, metric=haversine)

        for i, new_point in enumerate(new_points):
            nearest_indices = np.argsort(distances[i])[:k]
            nearest_distances = distances[i][nearest_indices]
            nearest_values = values[:, nearest_indices]
            weights = 1 / nearest_distances
            weights /= weights.sum()
            interpolated_value = np.dot(nearest_values, weights)
            interpolated_values.append(interpolated_value)

        return np.column_stack(interpolated_values)

class Network:
    def __init__(self, nodes_df, demand_df, time_series_dict, config: Config, time_horizon=None):
        """
        Initializes a network with nodes, time series, and line data.
        """
        self.config = config
        self.demand_df = demand_df
        self.nodes_df = nodes_df
        self.time_series_dict = time_series_dict
        self.time_horizon = time_horizon if time_horizon is not None else min(ts.shape[0] for ts in time_series_dict.values())
        # self.monthly = {}
        # self.monthly_corr = {}
        # self.series = {}
        for key in self.time_series_dict:
            self.time_series_dict[key] = self.time_series_dict[key].iloc[:self.time_horizon, :]

        self.features = self.compute_node_features()
        self.print_features_info()

    def print_features_info(self):
        print("The 'features' dictionary has been created and can be accessed as '.features'")
        print(f"It is a dictionary with keys for each node in {range(len(self.nodes_df))}.")
        print("Each value is a dictionary with the features of that node.")
        print("\nExample structure:")
        print(f"network.features[0].keys() = {self.features[0].keys()}")
        print("\nDetails:")
        print("  - Position: A tuple (latitude, longitude) of that node.")
        print(f"  - Time series: A dictionary with keys for each time series type in {self.time_series_dict.keys()}")
        print("    and values as the time series itself.")
        print(f'  - Duration Curves: A dictionary with keys for each time series type in {self.time_series_dict.keys()}')
        print("    and values as the duration curve of the time series.")
        print(f"  - Ramp Duration Curves (RDCs): A dictionary with keys for each time series type {self.time_series_dict.keys()}")
        print("    and values as the RDC (found by differentiating and subsequently sorting) of the time series.")
        print("  - Correlation: A dictionary with keys as tuples of types of time series")
        print("    and values as correlation factors between those time series.")
        print("  - Supply-demand mismatch: A dictionary with keys as types of time series")
        print("    and values as the supply-demand mismatch correlation factor for that node, either for year, monthly or weekly.")

    def compute_node_features(self):
        """
        Computes features for each node including position and correlations between time series types.
        Returns a dictionary of node features.
        """
        features = {}

        if self.config.demand == "total_demand":
            demand = self.demand_df.sum(axis=1).values
            # self.demand = demand
        elif self.config.demand == "k-interpolation":
            new_points = np.column_stack((self.nodes_df['Lat'], self.nodes_df['Lon']))
            values = self.demand_df.values
            demand_lat_lon = pd.read_csv(self.config.file_paths["demand_lat_lon"])
            points = np.column_stack((demand_lat_lon['Lat'], demand_lat_lon['Lon']))
            demand = DataProcessor.custom_interpolate(points, values, new_points, k=self.config.k_weight_demand)
            # self.demand = demand

        for node in range(len(self.nodes_df)):
            node_features = {
                'position': (self.nodes_df.iloc[node]['Lat'], self.nodes_df.iloc[node]['Lon']),
                'time_series': {key: ts.iloc[:, node].values for key, ts in self.time_series_dict.items()},
                'duration_curves': {key: np.flip(np.sort(ts.iloc[:, node].values.copy())) for key, ts in self.time_series_dict.items()},
                'ramp_duration_curves': {key: np.flip(np.sort(np.diff(ts.iloc[:, node].values.copy()))) for key, ts in self.time_series_dict.items()}
            }

            if len(self.time_series_dict) > 1:
                correlation = {}
                processed_pairs = set()
                for key1, ts1 in self.time_series_dict.items():
                    for key2, ts2 in self.time_series_dict.items():
                        if key1 != key2:
                            pair = tuple(sorted([key1, key2]))
                            if pair not in processed_pairs:
                                correlation[pair] = np.corrcoef(ts1.iloc[:, node], ts2.iloc[:, node])[0, 1]
                                processed_pairs.add(pair)
                node_features['correlation'] = correlation

            if self.config.demand == "total_demand":
                node_features['supply_demand_mismatch'] = self.supply_demand_mismatch(node, demand, self.config.time_scale)
            elif self.config.demand == "k-interpolation":
                node_features['supply_demand_mismatch'] = self.supply_demand_mismatch(node, demand[:, node], self.config.time_scale)

            features[node] = node_features
        return features

    def supply_demand_mismatch(self, node, demand, time_scale):
        """
        Computes the supply-demand mismatch for each node in the network.
        Returns a dictionary of supply-demand mismatches.
        """
        date_range = pd.date_range(start=f'{self.config.year}-01-01', periods=len(demand), freq='h')
        demand_series = pd.Series(demand, index=date_range)
        # self.demand_series = demand_series

        # self.series[node] = {}
        # self.monthly[node] = {}
        # self.monthly_corr[node] = {}
        correlation_dict = {}
        for key, ts in self.time_series_dict.items():
            series = ts.iloc[:, node]
            series.index = date_range
            # self.series[node][key] = series

            if time_scale == 'yearly':
                correlation = series.corr(demand_series)
                correlation_dict[key] = correlation

            elif time_scale == 'monthly':
                # monthly = series.groupby(series.index.month)
                # monthly_corr = []
                # for month, group in monthly:
                #     monthly_corr.append(group.corr(demand_series[group.index]))
                # self.monthly[node][key] = monthly
                # self.monthly_corr[node][key] = monthly_corr
                # correlation_dict[key] = np.array(monthly_corr)
                monthly = series.groupby(series.index.month).apply(lambda x: x.corr(demand_series[x.index]))
                correlation_dict[key] = monthly.values

            elif time_scale == 'weekly':
                weekly_corr = series.groupby(series.index.isocalendar().week).apply(lambda x: x.corr(demand_series[x.index]))
                correlation_dict[key] = weekly_corr.values

        return correlation_dict

    # def display_node_features(self, node_index):
    #     """
    #     Displays the features of a specified node in a readable format.
    #     """
    #     node_features = self.features[node_index]
    #     print(f"Node {node_index} Features:")
    #     print(f"  - Position: {node_features['position']}")
    #     print(f"  - Time Series:")
    #     for key, series in node_features['time_series'].items():
    #         print(f"    {key}: {series}")
    #     print(f"  - Duration Curves:")
    #     for key, series in node_features['duration_curves'].items():
    #         print(f"    {key}: {series}")
    #     print(f"  - Ramp Duration Curves:")
    #     for key, rdc in node_features['ramp_duration_curves'].items():
    #         print(f"    {key}: {rdc}")
    #     print(f"  - Correlations:")
    #     for pair, corr in node_features['correlation'].items():
    #         print(f"    {pair}: {corr}")



     

# class Visualization:
#     def __init__(self, network, u_result, n_repr, save_fig=False, save_dir=None):
#         """
#         Initialize the visualization with the network, optimizer, and aggregation results.
#         """
#         self.network = network
#         self.u_result = u_result
#         self.nodes_df = self.network.nodes_df
#         # self.n_repr = self.u_result.shape[1]
#         self.n_repr = n_repr
#         self.save_fig = save_fig
#         if self.save_fig:
#             if save_dir is None:
#                 self.save_dir = '../results/'
#             else:
#                 self.save_dir = os.path.join('../results/', save_dir)

#             if os.path.exists(self.save_dir) is False:
#                 os.makedirs(self.save_dir)

#     def save_figure(self, fig, fig_name):
#         """
#         Save the figure to the specified directory.
#         """
#         filepath = os.path.join(self.save_dir, fig_name)
#         fig.savefig(filepath, bbox_inches='tight')
#         print(f"Figure saved as {fig_name} at {self.save_dir}")

#     # def plot_map(self):
#     #     """
#     #     Plot the map of nodes with representative nodes highlighted.
#     #     """

#     #     # Plot the nodes
#     #     plt.figure(figsize=(7, 7))
#     #     for i in range(len(self.nodes_df)):
#     #         plt.plot(self.nodes_df.iloc[i]['Lon'],
#     #                  self.nodes_df.iloc[i]['Lat'], 'bo')
#     #         plt.text(self.nodes_df.iloc[i]['Lon'],
#     #                  self.nodes_df.iloc[i]['Lat'], f"{i}")

#     #     # Plot the representative nodes and lines between original nodes and their representatives
#     #     for j in range(self.n_repr):
#     #         for i in range(len(self.nodes_df)):
#     #             if self.u_result[i, j] == 1:
#     #                 # Plot the representative node
#     #                 plt.plot(self.nodes_df.iloc[i]['Lon'],
#     #                          self.nodes_df.iloc[i]['Lat'], 'ro')
#     #                 # Find the representative node coordinates
#     #                 rep_node_coords = (
#     #                     self.nodes_df.iloc[j]['Lon'], self.nodes_df.iloc[j]['Lat'])
#     #                 # Plot a line between the original node and its representative node
#     #                 plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]],
#     #                          [self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

#     #     plt.xlabel("Longitude")
#     #     plt.ylabel("Latitude")
#     #     plt.title("Node Aggregation Map")
#     #     plt.show()

#     #     if self.save_fig:
#     #         self.save_figure(f"node_aggregation_map_{
#     #             len(self.nodes_df)}_to_{self.n_repr}.png")

#     def plot_map(self, method='optimization', kmeans_cluster_centers=None, kmedoids_medoids=None, labels=None, AggregationClustering=None, cluster_mapping=None):
#         """
#         Plot the map of nodes with representative nodes highlighted.

#         Parameters:
#         method (str): The method used for clustering ('optimization', 'kmeans', 'kmedoids').
#         """

#         # Plot the nodes
#         fig = plt.figure(figsize=(7, 7))
#         for i in range(len(self.nodes_df)):
#             plt.plot(self.nodes_df.iloc[i]['Lon'],
#                      self.nodes_df.iloc[i]['Lat'], 'bo')
#             plt.text(self.nodes_df.iloc[i]['Lon'],
#                      self.nodes_df.iloc[i]['Lat'], f"{i}")

#         if method == 'optimization':
#             # Plot the representative nodes and lines between original nodes and their representatives
#             for j in range(self.u_result.shape[1]):
#                 for i in range(len(self.nodes_df)):
#                     if self.u_result[i, j] == 1:
#                         # Plot the original node
#                         plt.plot(
#                             self.nodes_df.iloc[i]['Lon'], self.nodes_df.iloc[i]['Lat'], 'ro')
#                         # Find the representative node coordinates
#                         rep_node_coords = (
#                             self.nodes_df.iloc[j]['Lon'], self.nodes_df.iloc[j]['Lat'])
#                         # Plot the representative node in blue
#                         plt.plot(rep_node_coords[0], rep_node_coords[1], 'bo')
#                         # Plot a line between the original node and its representative node
#                         plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]], [
#                                  self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

#         elif method == 'kmeans':
#             # Plot the centroids as new nodes
#             for j, centroid in enumerate(kmeans_cluster_centers):
#                 # Assuming centroid is in (Lat, Lon) format
#                 plt.plot(centroid[1], centroid[0], 'ro')
#                 for i in range(len(self.nodes_df)):
#                     if labels[i] == j:
#                         # Plot a line between the original node and its centroid
#                         plt.plot([self.nodes_df.iloc[i]['Lon'], centroid[1]],
#                                  [self.nodes_df.iloc[i]['Lat'], centroid[0]], 'k-')

#         # elif method == 'kmeans':
#         #     # Find the closest actual node to each centroid in the feature space
#         #     feature_matrix = AggregationClustering.compute_feature_matrix()
#         #     closest_nodes = []
#         #     for centroid in kmeans_cluster_centers:
#         #         distances = distance.cdist(
#         #             [centroid], feature_matrix, 'euclidean')
#         #         closest_node = np.argmin(distances)
#         #         closest_nodes.append(closest_node)

#         #     # Plot the representative nodes and lines between original nodes and their representatives
#         #     for j in range(self.n_repr):
#         #         rep_node_coords = (
#         #             self.nodes_df.iloc[closest_nodes[j]]['Lon'], self.nodes_df.iloc[closest_nodes[j]]['Lat'])
#         #         plt.plot(rep_node_coords[0], rep_node_coords[1], 'ro')
#         #         for i in range(len(self.nodes_df)):
#         #             if labels[i] == j:
#         #                 # Plot a line between the original node and its representative node
#         #                 plt.plot([self.nodes_df.iloc[i]['Lon'], rep_node_coords[0]],
#         #                          [self.nodes_df.iloc[i]['Lat'], rep_node_coords[1]], 'k-')

#         elif method == 'kmedoids':
#             # Plot the representative nodes and lines between original nodes and their representatives
#             for j in range(self.n_repr):
#                 rep_node_coords = (
#                     self.nodes_df.iloc[kmedoids_medoids[j]]['Lon'], self.nodes_df.iloc[kmedoids_medoids[j]]['Lat'])
#                 plt.plot(rep_node_coords[0], rep_node_coords[1], 'ro')
#                 for node in cluster_mapping[j]:
#                     # Plot a line between the original node and its representative node
#                     plt.plot([self.nodes_df.iloc[node]['Lon'], rep_node_coords[0]], [
#                              self.nodes_df.iloc[node]['Lat'], rep_node_coords[1]], 'k-')

#         plt.xlabel("Longitude")
#         plt.ylabel("Latitude")
#         plt.title("Node Aggregation Map")
#         plt.tight_layout()
#         plt.show()

#         if self.save_fig:
#             if method == 'optimization':
#                 self.save_figure(fig, f"opti_agg_map_{
#                     len(self.nodes_df)}_to_{self.u_result.shape[1]}.png")
#             elif method == 'kmeans':
#                 self.save_figure(fig, f"kmeans_agg_map_{
#                     len(self.nodes_df)}_to_{self.n_repr}.png")
#             elif method == 'kmedoids':
#                 self.save_figure(fig, f"kmedoids_agg_map_{
#                     len(self.nodes_df)}_to_{self.n_repr}.png")