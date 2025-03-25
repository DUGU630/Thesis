import numpy as np
import pandas as pd
import os
from dataclasses import asdict
import gurobipy as gp
from gurobipy import GRB
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import cdist
import utils_mistral as utils
from datetime import datetime


import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist
from dataclasses import asdict

class SpatialAggregation:
    def __init__(self, node_features, config):
        """
        Initialize the optimizer with nodes, time series, and aggregation parameters.
        """
        self.nodes_features = node_features
        self.config = config
        self.num_original_nodes = len(node_features)
        self.optimized_assignment_dict = None
        self.cluster_assignment_dict = None
        self.distance_metrics = self.compute_distance_metrics(node_features, node_features)
        
    def compute_distance_metrics(self, node_features_set1, node_features_set2, normalize=True):
        """
        Compute or load cached distance metrics between two sets of nodes.
        """
        # Extract features for both sets
        positions_set1 = np.array([node['position'] for node in node_features_set1.values()])
        positions_set2 = np.array([node['position'] for node in node_features_set2.values()])
        
        time_series_set1 = self._extract_time_series(node_features_set1)
        time_series_set2 = self._extract_time_series(node_features_set2)
        
        # Compute each component with caching
        position_distance = self._get_cached_component(
            'position', positions_set1, positions_set2,
            self._compute_position_distances, {'metric': 'haversine'}, normalize
        )
        
        time_series_distance = self._get_cached_component(
            'time_series', time_series_set1, time_series_set2,
            self._compute_euclidean_distances, {'metric': 'euclidean'}, normalize
        )
        
        # Add other components similarly (duration_curves, ramp_duration_curves, etc.)
        # Example for supply_demand_mismatch:
        supply_demand_set1 = self._extract_supply_demand(node_features_set1)
        supply_demand_set2 = self._extract_supply_demand(node_features_set2)
        supply_demand_distance = self._get_cached_component(
            'supply_demand', supply_demand_set1, supply_demand_set2,
            self._compute_euclidean_distances, {'metric': 'euclidean'}, normalize
        )
        
        # Compute inter_correlation (example with caching)
        inter_correlation_distance = self._get_cached_component(
            'inter_correlation', time_series_set1, time_series_set2,
            self._compute_inter_correlation, {'method': 'pearson'}, normalize
        )
        
        return {
            'position_distance': position_distance,
            'time_series_distance': time_series_distance,
            'supply_demand_mismatch_distance': supply_demand_distance,
            'inter_correlation_distance': inter_correlation_distance,
            # Include other components similarly
        }

    def _get_cached_component(self, component_name, data_set1, data_set2, compute_func, params, normalize):
        """
        Retrieve a component from cache or compute and cache it.
        """
        data_hash = self._get_data_hash(data_set1, data_set2)
        cache_dir = os.path.join(self.config.cache_root, component_name, data_hash)
        os.makedirs(cache_dir, exist_ok=True)
        
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        raw_path = os.path.join(cache_dir, 'raw.npy')
        
        # Check if valid cache exists
        if os.path.exists(metadata_path) and os.path.exists(raw_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata['params'] == params:
                raw_matrix = np.load(raw_path)
                min_val = metadata['min']
                max_val = metadata['max']
            else:
                raw_matrix, min_val, max_val = self._compute_and_save(
                    compute_func, data_set1, data_set2, params, raw_path, metadata_path
                )
        else:
            raw_matrix, min_val, max_val = self._compute_and_save(
                compute_func, data_set1, data_set2, params, raw_path, metadata_path
            )
        
        return self._normalize(raw_matrix, min_val, max_val) if normalize else raw_matrix

    def _compute_and_save(self, compute_func, data_set1, data_set2, params, raw_path, metadata_path):
        """Compute the component and save to cache."""
        raw_matrix = compute_func(data_set1, data_set2)
        min_val = np.min(raw_matrix)
        max_val = np.max(raw_matrix)
        
        np.save(raw_path, raw_matrix)
        metadata = {
            'params': params,
            'min': min_val,
            'max': max_val,
            'data_hash_set1': self._get_data_hash(data_set1),
            'data_hash_set2': self._get_data_hash(data_set2),
            'timestamp': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        return raw_matrix, min_val, max_val

    @staticmethod
    def _get_data_hash(*data_arrays):
        """Generate a hash from numpy arrays."""
        hash_obj = hashlib.sha256()
        for data in data_arrays:
            hash_obj.update(data.tobytes())
        return hash_obj.hexdigest()

    @staticmethod
    def _normalize(matrix, min_val, max_val):
        """Normalize matrix using precomputed min/max."""
        return (matrix - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

    def _compute_position_distances(self, positions1, positions2):
        """Compute Haversine distance."""
        return cdist(positions1, positions2, metric=utils.haversine)

    def _compute_euclidean_distances(self, data1, data2):
        """Compute Euclidean distance."""
        return cdist(data1, data2, metric='euclidean')

    def _compute_inter_correlation(self, ts1, ts2):
        """Compute pairwise correlation distance."""
        n1, n2 = ts1.shape[0], ts2.shape[0]
        dist_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                corr = np.corrcoef(ts1[i], ts2[j])[0, 1]
                dist_matrix[i, j] = -corr
        return dist_matrix

    def _extract_time_series(self, node_features):
        """Helper to extract time series data."""
        return np.array([list(node['time_series'].values()) for node in node_features.values()])

    def _extract_supply_demand(self, node_features):
        """Helper to extract supply-demand mismatch data."""
        return np.array([list(node['supply_demand_mismatch'].values()) for node in node_features.values()])

    def compute_distance_matrix(self, weights):
        """Combine cached components with current weights."""
        total = np.zeros_like(self.distance_metrics['position_distance'])
        total += weights['position'] * self.distance_metrics['position_distance']
        total += weights['time_series'] * self.distance_metrics['time_series_distance']
        total -= weights['inter_correlation'] * self.distance_metrics['inter_correlation_distance']
        # Add other components similarly
        return total

    def optimize(self):
        """Run optimization using current distance matrix."""
        total_distance = self.compute_distance_matrix(self.config.weights)
        # Formulate and solve optimization model here