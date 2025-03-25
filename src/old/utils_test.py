import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import xarray as xr
from dataclasses import dataclass, field

# Utility Functions
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@dataclass
class Config:
    year: int = 2013
    drop_duplicates: int = 1
    k_neighbors: int = 1
    time_scale: str = "monthly"
    total_demand: int = 1
    data_dir: str = "../DATA/dev"

    def __post_init__(self):
        if self.drop_duplicates not in [0, 1]:
            raise ValueError("drop_duplicates must be 0 or 1")
        if not isinstance(self.k_neighbors, int) or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer")
        if self.time_scale not in ["weekly", "monthly", "yearly"]:
            raise ValueError("time_scale must be 'weekly', 'monthly', or 'yearly'")

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def import_and_interpolate_data(self):
        """
        Import and interpolate data for nodes, wind capacity factors, and solar capacity factors.
        """
        # Load node and demand data
        nodes_path = os.path.join(self.config.data_dir, "NewEngland-HVbuses.csv")
        demand_path = os.path.join(
            self.config.data_dir, f"demand_hist/county_demand_local_hourly_{self.config.year}.csv")
        
        nodes_df = pd.read_csv(nodes_path)
        demand_df = pd.read_csv(demand_path)

        if self.config.drop_duplicates:
            nodes_df = nodes_df.drop_duplicates(subset=['Lat', 'Lon'])

        wind_cf_path = os.path.join(
            self.config.data_dir, f"CapacityFactors_ISONE/Wind/cf_Wind_0.22m_{self.config.year}.nc")
        solar_cf_path = os.path.join(
            self.config.data_dir, f"CapacityFactors_ISONE/Solar/cf_Solar_0.22m_{self.config.year}.nc")

        wind_nc = xr.open_dataset(wind_cf_path)['cf']
        solar_nc = xr.open_dataset(solar_cf_path)['cf']

        new_points = np.column_stack((nodes_df['Lat'], nodes_df['Lon']))

        wind_df = self._interpolate_capacity_factors(wind_nc, new_points)
        solar_df = self._interpolate_capacity_factors(solar_nc, new_points)

        return nodes_df, demand_df, wind_df, solar_df

    def _interpolate_capacity_factors(self, nc_data, new_points):
        """
        Interpolates capacity factor data to new points.
        """
        data = nc_data.stack(z=("lat", "lon")).dropna('z', how='all')
        values = data.values
        points = np.column_stack((data.lat.values, data.lon.values))
        interpolated_values = self._custom_interpolate(points, values, new_points, self.config.k_neighbors)
        return pd.DataFrame(interpolated_values)

    def _custom_interpolate(self, points, values, new_points, k):
        """
        Custom interpolation using weighted nearest neighbors.
        """
        interpolated_values = []

        for new_point in new_points:
            distances = [haversine(new_point[0], new_point[1], point[0], point[1]) for point in points]
            sorted_indices = np.argsort(distances)[:k]
            nearest_distances = np.array(distances)[sorted_indices]
            nearest_values = np.array(values)[:, sorted_indices]
            weights = 1 / nearest_distances
            weights /= weights.sum()
            interpolated_value = np.dot(nearest_values, weights)
            interpolated_values.append(interpolated_value)

        return np.column_stack(interpolated_values)

class Network:
    def __init__(self, config: Config, nodes_df, demand_df, wind_df, solar_df):
        """
        Initialize the network with node and time series data.
        """
        self.config = config
        self.nodes_df = nodes_df
        self.demand_df = demand_df
        self.wind_df = wind_df
        self.solar_df = solar_df
        self.features = self._compute_node_features()

    def _compute_node_features(self):
        """
        Compute features for each node, including demand and correlations.
        """
        features = {}
        for idx, node in self.nodes_df.iterrows():
            lat, lon = node['Lat'], node['Lon']
            features[idx] = {
                "position": (lat, lon),
                "demand": self.demand_df.iloc[:, idx].sum()
            }
        return features

# Example Usage
if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config)
    nodes, demand, wind, solar = processor.import_and_interpolate_data()
    network = Network(config, nodes, demand, wind, solar)
    print(network.features)
