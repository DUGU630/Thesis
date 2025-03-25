import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
# from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.utils import read_sample
from scipy.spatial.distance import cdist
import utils_mistral as utils


class Aggregation:
    def __init__(self, node_features, config: utils.Config):
        """
        Initialize the optimizer with nodes, time series, and aggregation parameters.
        """
        self.nodes_features = node_features
        self.config = config
        self.num_nodes = len(node_features)
        self.distance_matrices = self.compute_distance_matrices()

    def compute_distance_matrices(self, node_features1, node_features2):
        """
        Compute the distance matrix between node features.
        """
        weights = self.config.weights
        num_nodes = self.num_nodes
        dist_matrix = np.zeros((num_nodes, num_nodes))

        # Precompute positions and time series distances
        positions = np.array([node['position'] for node in self.nodes_features.values()])
        time_series = np.array([list(node['time_series'].values()) for node in self.nodes_features.values()]).reshape(num_nodes, -1)
        duration_curves = np.array([list(node['duration_curves'].values()) for node in self.nodes_features.values()]).reshape(num_nodes, -1)
        rdc = np.array([list(node['ramp_duration_curves'].values()) for node in self.nodes_features.values()]).reshape(num_nodes, -1)
        intra_correlation = np.array([list(node['correlation'].values()) for node in self.nodes_features.values()]).reshape(num_nodes, -1)
        supply_demand_mismatch = np.array([list(node['supply_demand_mismatch'].values()) for node in self.nodes_features.values()]).reshape(num_nodes, -1)

        position_distances = self._compute_position_distances(positions)
        time_series_distances = self._compute_euclidean_distances(time_series)
        duration_curves_distances = self._compute_euclidean_distances(duration_curves)
        rdc_distances = self._compute_euclidean_distances(rdc)
        intra_correlation_distances = self._compute_euclidean_distances(intra_correlation)
        supply_demand_mismatch_distances = self._compute_euclidean_distances(supply_demand_mismatch)

        # Normalize distances
        position_distances = self._normalize_distances(position_distances)
        time_series_distances = self._normalize_distances(time_series_distances)
        duration_curves_distances = self._normalize_distances(duration_curves_distances)
        rdc_distances = self._normalize_distances(rdc_distances)
        intra_correlation_distances = self._normalize_distances(intra_correlation_distances)
        supply_demand_mismatch_distances = self._normalize_distances(supply_demand_mismatch_distances)

        # Compute inter_correlation distances
        inter_correlation_distances = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                ts1 = time_series[i]
                ts2 = time_series[j]
                inter_correlation_distance = -np.corrcoef(ts1, ts2)[0, 1]
                inter_correlation_distances[i, j] = inter_correlation_distance
                inter_correlation_distances[j, i] = inter_correlation_distance

        inter_correlation_distances = self._normalize_distances(inter_correlation_distances)

        # Compute total distance
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                total_distance = (
                    weights['position'] * position_distances[i, j] +
                    weights['time_series'] * time_series_distances[i, j] +
                    weights['duration_curves'] * duration_curves_distances[i, j] +
                    weights['rdc'] * rdc_distances[i, j] +
                    weights['intra_correlation'] * intra_correlation_distances[i, j] +
                    weights['inter_correlation'] * inter_correlation_distances[i, j] +
                    weights['supply_demand_mismatch'] * supply_demand_mismatch_distances[i, j]
                )
                dist_matrix[i, j] = total_distance
                dist_matrix[j, i] = total_distance

        return {
            'total_distance': dist_matrix,
            'position_distance': position_distances,
            'time_series_distance': time_series_distances,
            'duration_curves_distance': duration_curves_distances,
            'rdc_distance': rdc_distances,
            'intra_correlation_distance': intra_correlation_distances,
            'inter_correlation_distance': inter_correlation_distances,
            'supply_demand_mismatch_distance': supply_demand_mismatch_distances
        }
    
    @staticmethod
    def _compute_position_distances(positions1, positions2):
        """
        Compute the Haversine distance matrix for positions.
        """
        return cdist(positions1, positions1, metric=utils.haversine)
    
    @staticmethod
    def _compute_euclidean_distances(data1, data2):
        """
        Compute the Euclidean distance matrix for the given data.
        """
        return cdist(data1, data2, metric='euclidean')
        # return np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=2)

    # def _compute_absolute_distances(self, data):
    #     """
    #     Compute the absolute difference distance matrix for the given data.
    #     """
    #     return np.abs(data[:, np.newaxis] - data[np.newaxis, :])

    @staticmethod
    def _normalize_distances(distances, min_val=None, max_val=None):
        """
        Normalize the distance matrix.
        """
        if min_val is None:
            min_val = np.min(distances)
        if max_val is None:
            max_val = np.max(distances)
        return (distances - min_val) / (max_val - min_val)

    def optimize(self):
        """
        Formulate and solve the optimization model.
        """
        n_repr = self.config.n_repr
        distance_matrices = self.distance_matrices
        total_distance_matrix = distance_matrices['total_distance']
        num_nodes = self.num_nodes

        model = gp.Model("node_aggregation")
        u = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="u")
        z = model.addVars(num_nodes, vtype=GRB.BINARY, name="z")

        model.setObjective(
            gp.quicksum(u[i, j] * total_distance_matrix[i, j]
                        for i in range(num_nodes) for j in range(num_nodes)),
            GRB.MINIMIZE
        )

        model.addConstrs(
            (gp.quicksum(u[i, j] for j in range(num_nodes)) == 1 for i in range(num_nodes)),
            "assignment"
        )

        model.addConstrs(
            (u[i, j] <= z[j] for i in range(num_nodes) for j in range(num_nodes)),
            "representative_assignment"
        )

        model.addConstr(
            gp.quicksum(z[j] for j in range(num_nodes)) == n_repr,
            "num_representatives"
        )

        model.optimize()

        u_result = np.zeros((num_nodes, num_nodes))
        z_result = np.zeros(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if u[i, j].X > 0.5:
                    u_result[i, j] = 1
        for j in range(num_nodes):
            if z[j].X > 0.5:
                z_result[j] = 1

        print("\nOptimization Results:")
        print(f"  - Objective Value: {model.objVal}")

        representatives = [j for j in range(num_nodes) if z_result[j] == 1]
        print(f"  - Representatives: {representatives}")
        for j in representatives:
            mapped_nodes = [i for i in range(num_nodes) if u_result[i, j] == 1]
            print(f"    - Representative node {j}: {mapped_nodes}")

        return u_result, z_result

    def cluster_KMedoids(self):
        """
        Perform k-medoids clustering on the node features using the precomputed distance matrix.
        """
        distance_matrices = self.distance_matrices
        total_distance_matrix = distance_matrices['total_distance']

        initial_medoids = np.random.choice(range(self.num_nodes), self.config.n_repr, replace=False)
        kmedoids_instance = kmedoids(total_distance_matrix, initial_medoids.tolist(), data_type='distance_matrix')
        kmedoids_instance.process()

        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        cluster_mapping = {i: [] for i in range(len(medoids))}
        for idx, cluster in enumerate(clusters):
            cluster_mapping[idx] = cluster

        return cluster_mapping, medoids
    
    def compute_metrics(self, type="custom"):
        """
        Compute all metrics based on the original and aggregated features.
        :return: Dictionary with metric values.
        """

        clusters_KMedoids, cluster_centers_KMedoids = self.cluster_KMedoids()
        u_results, z_results = self.optimize()

        aggregated_features_KMedoids = {}
        for cluster_id, nodes in clusters_KMedoids.items():
            representative = cluster_centers_KMedoids[cluster_id]
            for node in nodes:
                aggregated_features_KMedoids[node] = self.node_features[representative]

        aggregated_features_optimized = {}
        for i, selected in enumerate(z_results):
            if selected:
                for j in range(len(u_results)):
                    if u_results[j, i] == 1:
                        aggregated_features_optimized[j] = self.node_features[i]
        metrics = self._compute_metrics(aggregated_features, type)
        return metrics

    def _compute_metrics(self, aggregated_features, type="custom"):
        """
        Compute all metrics based on the original and aggregated features.
        :return: Dictionary with metric values.
        """
        self.aggregated_features = aggregated_features
        self.original_features = self.nodes_features

        if type == "literature":
            metrics = {
                "REEav": self.compute_reeav(),
                "NRMSEav": self.compute_nrmseav(),
                "CEav": self.compute_ceav(),
                "NRMSERDCav": self.compute_nrmsedcav()
            }
        
        elif type == "custom":
            metrics = {
                "position": self.compute_position(),
                "time_series": self.compute_time_series(),
                "duration_curves": self.compute_duration_curves(),
                "ramp_duration_curves": self.compute_ramp_duration_curves(),
                "intra_correlation": self.intra_compute_correlation(),
                "inter_correlation": self.inter_compute_correlation(),
                "supply_demand_mismatch": self.compute_supply_demand_mismatch()
            }
        else:
            raise ValueError("Metric type must be 'literature' or 'custom'.")

        return metrics

    def compute_reeav(self):
        """
        Compute Relative Energy Error Average (REEav).
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["duration_curves"]:
                orig_dc = orig["duration_curves"][key]
                agg_dc = agg["duration_curves"][key]
                total_energy_orig = np.sum(orig_dc)
                total_energy_agg = np.sum(agg_dc)
                relative_error = abs(total_energy_orig -
                                     total_energy_agg) / total_energy_orig
                errors.append(relative_error)
        return np.mean(errors)

    def compute_nrmseav(self):
        """
        Compute Normalized Root Mean Square Error Average (NRMSEav).
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["duration_curves"]:
                orig_dc = orig["duration_curves"][key]
                agg_dc = agg["duration_curves"][key]
                mse = np.mean((orig_dc - agg_dc) ** 2)
                nrmse = np.sqrt(mse) / (np.max(orig_dc) - np.min(orig_dc))
                errors.append(nrmse)
        return np.mean(errors)

    def compute_ceav(self):
        """
        Compute Correlation Error Average (CEav).
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["correlation"]:
                orig_corr = orig["correlation"][key]
                agg_corr = agg["correlation"][key]
                error = abs(orig_corr - agg_corr)
                errors.append(error)
        return np.mean(errors)

    def compute_nrmsedcav(self):
        """
        Compute Normalized RMS Error of Ramp Duration Curve Average (NRMSERDCav).
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["ramp_duration_curves"]:
                orig_rdc = orig["ramp_duration_curves"][key]
                agg_rdc = agg["ramp_duration_curves"][key]
                mse = np.mean((orig_rdc - agg_rdc) ** 2)
                nrmse = np.sqrt(mse) / (np.max(orig_rdc) - np.min(orig_rdc))
                errors.append(nrmse)
        return np.mean(errors)

    def compute_position(self):
        """
        Compute Position Distance.
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            error = utils.haversine(orig["position"], agg["position"])
            errors.append(error)
        return np.mean(errors)
    
    def compute_time_series(self):
        """
        Compute Time Series Distance.
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["time_series"]:
                orig_ts = orig["time_series"][key]
                agg_ts = agg["time_series"][key]
                error = np.linalg.norm(orig_ts - agg_ts)
                errors.append(error)
        return np.mean(errors)
    
    def compute_duration_curves(self):
        """
        Compute Duration Curves Distance.
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["duration_curves"]:
                orig_dc = orig["duration_curves"][key]
                agg_dc = agg["duration_curves"][key]
                error = np.linalg.norm(orig_dc - agg_dc)
                errors.append(error)
        return np.mean(errors)
    
    def compute_ramp_duration_curves(self):
        """
        Compute Ramp Duration Curves Distance.
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["ramp_duration_curves"]:
                orig_rdc = orig["ramp_duration_curves"][key]
                agg_rdc = agg["ramp_duration_curves"][key]
                error = np.linalg.norm(orig_rdc - agg_rdc)
                errors.append(error)
        return np.mean(errors)
    
    def intra_compute_correlation(self):
        """
        Compute Intra-node Correlation Distance.
        """
        errors = []
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["correlation"]:
                orig_corr = orig["correlation"][key]
                agg_corr = agg["correlation"][key]
                error = abs(orig_corr - agg_corr)
                errors.append(error)
        return np.mean(errors)
    
    def inter_compute_correlation(self):
        """
        Compute Inter-node Correlation Distance.
        """
        corrs = []
        i = 0
        for node, orig in self.original_features.items():
            agg = self.aggregated_features[node]
            for key in orig["time_series"]:
                orig_ts = orig["time_series"][key]
                agg_ts = agg["time_series"][key]
                corr = np.corrcoef(orig_ts, agg_ts)[0, 1]
                corrs.append(corr)
                
                
        return np.mean(corrs)