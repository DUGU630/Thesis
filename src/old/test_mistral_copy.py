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

class SpatialAggregation:
    def __init__(self, node_features, config: utils.Config):
        """
        Initialize the optimizer with nodes, time series, and aggregation parameters.
        """
        self.nodes_features = node_features
        self.config = config
        self.num_original_nodes = len(node_features)
        self.distance_metrics = self.compute_distance_metrics(node_features, node_features)
        self.optimized_assignment_dict = None
        self.cluster_assignment_dict = None
        self.distance_metrics = None

    def set_distance_metrics(self, load_distance_metrics = None):
        """
        Set the distance metrics from a precomputed file.
        """
        if load_distance_metrics is not None:
            self.distance_metrics = self.load_distance_metrics(load_distance_metrics)
        else:
            self.distance_metrics = self.compute_distance_metrics(self.nodes_features, self.nodes_features)
            if not os.path.exists("../results/distance_metrics"):
                os.makedirs("../results/distance_metrics", exist_ok=True)
            now = datetime.now()
            str = now.strftime("%Y-%m-%d %Hh%M")
            dir_name = f"{str} - Distance Metrics"
            self.save_distance_metrics(self.distance_metrics, dir_name)

    def compute_distance_metrics(self, node_features_set1, node_features_set2, normalize=True):
        """
        Compute the distance matrix between node features.
        """
        num_nodes1 = len(node_features_set1)
        num_nodes2 = len(node_features_set2)

        positions_set1 = np.array([node['position'] for node in node_features_set1.values()])
        positions_set2 = np.array([node['position'] for node in node_features_set2.values()])
        time_series_set1 = np.array([list(node['time_series'].values()) for node in node_features_set1.values()]).reshape(num_nodes1, -1)
        time_series_set2 = np.array([list(node['time_series'].values()) for node in node_features_set2.values()]).reshape(num_nodes2, -1)
        duration_curves_set1 = np.array([list(node['duration_curves'].values()) for node in node_features_set1.values()]).reshape(num_nodes1, -1)
        duration_curves_set2 = np.array([list(node['duration_curves'].values()) for node in node_features_set2.values()]).reshape(num_nodes2, -1)
        ramp_duration_curves_set1 = np.array([list(node['ramp_duration_curves'].values()) for node in node_features_set1.values()]).reshape(num_nodes1, -1)
        ramp_duration_curves_set2 = np.array([list(node['ramp_duration_curves'].values()) for node in node_features_set2.values()]).reshape(num_nodes2, -1)
        intra_correlation_set1 = np.array([list(node['correlation'].values()) for node in node_features_set1.values()]).reshape(num_nodes1, -1)
        intra_correlation_set2 = np.array([list(node['correlation'].values()) for node in node_features_set2.values()]).reshape(num_nodes2, -1)
        supply_demand_mismatch_set1 = np.array([list(node['supply_demand_mismatch'].values()) for node in node_features_set1.values()]).reshape(num_nodes1, -1)
        supply_demand_mismatch_set2 = np.array([list(node['supply_demand_mismatch'].values()) for node in node_features_set2.values()]).reshape(num_nodes2, -1)

        position_distance_matrix = self._compute_position_distances(positions_set1, positions_set2)
        time_series_distance_matrix = self._compute_euclidean_distances(time_series_set1, time_series_set2)
        duration_curves_distance_matrix = self._compute_euclidean_distances(duration_curves_set1, duration_curves_set2)
        ramp_duration_curves_distance_matrix = self._compute_euclidean_distances(ramp_duration_curves_set1, ramp_duration_curves_set2)
        intra_correlation_distance_matrix = self._compute_euclidean_distances(intra_correlation_set1, intra_correlation_set2)
        supply_demand_mismatch_distance_matrix = self._compute_euclidean_distances(supply_demand_mismatch_set1, supply_demand_mismatch_set2)
        
        inter_correlation_distance_matrix = np.zeros((num_nodes1, num_nodes2))
        for i in range(num_nodes1):
            for j in range(num_nodes2):
                ts1 = time_series_set1[i]
                ts2 = time_series_set2[j]
                inter_correlation_distance = -np.corrcoef(ts1, ts2)[0, 1]
                inter_correlation_distance_matrix[i, j] = inter_correlation_distance

        if normalize:
            position_distance_matrix = self._normalize_distances(position_distance_matrix)
            time_series_distance_matrix = self._normalize_distances(time_series_distance_matrix)
            duration_curves_distance_matrix = self._normalize_distances(duration_curves_distance_matrix)
            ramp_duration_curves_distance_matrix = self._normalize_distances(ramp_duration_curves_distance_matrix)
            intra_correlation_distance_matrix = self._normalize_distances(intra_correlation_distance_matrix)
            supply_demand_mismatch_distance_matrix = self._normalize_distances(supply_demand_mismatch_distance_matrix)
            inter_correlation_distance_matrix = self._normalize_distances(inter_correlation_distance_matrix)

        return {
            'position_distance': position_distance_matrix,
            'time_series_distance': time_series_distance_matrix,
            'duration_curves_distance': duration_curves_distance_matrix,
            'rdc_distance': ramp_duration_curves_distance_matrix,
            'intra_correlation_distance': intra_correlation_distance_matrix,
            'inter_correlation_distance': inter_correlation_distance_matrix,
            'supply_demand_mismatch_distance': supply_demand_mismatch_distance_matrix
        }
    
    @staticmethod
    def compute_distance_matrice(distance_metrics, weights):
        num_nodes1, num_nodes2 = distance_metrics['position_distance'].shape
        total_distance_matrix = np.zeros((num_nodes1, num_nodes2))

        for i in range(num_nodes1):
            for j in range(num_nodes2):
                total_distance = (
                    weights['position'] * distance_metrics['position_distance'][i, j] +
                    weights['time_series'] * distance_metrics['time_series_distance'][i, j] +
                    weights['duration_curves'] * distance_metrics['duration_curves_distance'][i, j] +
                    weights['rdc'] * distance_metrics['ramp_duration_curves_distance'][i, j] -
                    weights['intra_correlation'] * distance_metrics['intra_correlation_distance'][i, j] -
                    weights['inter_correlation'] * distance_metrics['inter_correlation_distance'][i, j] -
                    weights['supply_demand_mismatch'] * distance_metrics['supply_demand_mismatch_distance'][i, j]
                )
                total_distance_matrix[i, j] = total_distance
        
        return total_distance_matrix
    
    def save_distance_metrics(self, distance_metrics, dir_name):
        dir_path = os.path.join("../results/distance_metrics", dir_name)
        os.makedirs(dir_path, exist_ok=True)

        pd.DataFrame(distance_metrics['position_distance']).to_csv(f"{dir_path}/position_distance.csv")
        pd.DataFrame(distance_metrics['time_series_distance']).to_csv(f"{dir_path}/time_series_distance.csv")
        pd.DataFrame(distance_metrics['duration_curves_distance']).to_csv(f"{dir_path}/duration_curves_distance.csv")
        pd.DataFrame(distance_metrics['rdc_distance']).to_csv(f"{dir_path}/rdc_distance.csv")
        pd.DataFrame(distance_metrics['intra_correlation_distance']).to_csv(f"{dir_path}/intra_correlation_distance.csv")
        pd.DataFrame(distance_metrics['inter_correlation_distance']).to_csv(f"{dir_path}/inter_correlation_distance.csv")
        pd.DataFrame(distance_metrics['supply_demand_mismatch_distance']).to_csv(f"{dir_path}/supply_demand_mismatch_distance.csv")
        metadata = {
            'num_nodes1': distance_metrics['position_distance'].shape[0],
            'num_nodes2': distance_metrics['position_distance'].shape[1]
        }
        config_dict = asdict(self.config)
        if 'file_paths' in config_dict:
            del config_dict['file_paths']
        for key in config_dict:
            metadata[key] = config_dict[key]

        pd.DataFrame(metadata).to_csv(f"{dir_path}/metadata.csv")
    
    @staticmethod
    def load_distance_metrics(dir_name):
        dir_path = os.path.join("../results/distance_metrics", dir_name)
        position_distance = pd.read_csv(f"{dir_path}/position_distance.csv").values
        time_series_distance = pd.read_csv(f"{dir_path}/time_series_distance.csv").values
        duration_curves_distance = pd.read_csv(f"{dir_path}/duration_curves_distance.csv").values
        rdc_distance = pd.read_csv(f"{dir_path}/rdc_distance.csv").values
        intra_correlation_distance = pd.read_csv(f"{dir_path}/intra_correlation_distance.csv").values
        inter_correlation_distance = pd.read_csv(f"{dir_path}/inter_correlation_distance.csv").values
        supply_demand_mismatch_distance = pd.read_csv(f"{dir_path}/supply_demand_mismatch_distance.csv").values
        # metadata = pd.read_csv(f"{dir_path}/metadata.csv").to_dict()
        return {
            'position_distance': position_distance,
            'time_series_distance': time_series_distance,
            'duration_curves_distance': duration_curves_distance,
            'rdc_distance': rdc_distance,
            'intra_correlation_distance': intra_correlation_distance,
            'inter_correlation_distance': inter_correlation_distance,
            'supply_demand_mismatch_distance': supply_demand_mismatch_distance
        }

    @staticmethod
    def _compute_position_distances(positions_set1, positions_set2):
        """
        Compute the Haversine distance matrix for positions.
        """
        return cdist(positions_set1, positions_set2, metric=utils.haversine)

    @staticmethod
    def _compute_euclidean_distances(data1, data2):
        """
        Compute the Euclidean distance matrix for the given data.
        """
        return cdist(data1, data2, metric='euclidean')

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
        distance_metrics = self.distance_metrics
        weights = self.config.weights
        total_distance_matrix = self.compute_distance_matrice(distance_metrics, weights)
        num_nodes = self.num_original_nodes

        model = gp.Model("node_aggregation")
        assignment_matrix_vars = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="assignment_matrix")
        representative_vector_vars = model.addVars(num_nodes, vtype=GRB.BINARY, name="representative_vector")

        model.setObjective(
            gp.quicksum(assignment_matrix_vars[i, j] * total_distance_matrix[i, j]
                        for i in range(num_nodes) for j in range(num_nodes)),
            GRB.MINIMIZE
        )

        model.addConstrs(
            (gp.quicksum(assignment_matrix_vars[i, j] for j in range(num_nodes)) == 1 for i in range(num_nodes)),
            "assignment"
        )

        model.addConstrs(
            (assignment_matrix_vars[i, j] <= representative_vector_vars[j] for i in range(num_nodes) for j in range(num_nodes)),
            "representative_assignment"
        )

        model.addConstr(
            gp.quicksum(representative_vector_vars[j] for j in range(num_nodes)) == n_repr,
            "num_representatives"
        )

        model.optimize()

        assignment_matrix_result = np.zeros((num_nodes, num_nodes))
        representative_vector_result = np.zeros(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if assignment_matrix_vars[i, j].X > 0.5:
                    assignment_matrix_result[i, j] = 1
        for j in range(num_nodes):
            if representative_vector_vars[j].X > 0.5:
                representative_vector_result[j] = 1

        print("\nOptimization Results:")
        print(f"  - Objective Value: {model.objVal}")

        optimized_assignment_dict = {}
        representatives = [j for j in range(num_nodes) if representative_vector_result[j] == 1]
        print(f"  - Representatives: {representatives}")
        for j in representatives:
            mapped_nodes = [i for i in range(num_nodes) if assignment_matrix_result[i, j] == 1]
            optimized_assignment_dict[j] = mapped_nodes
            print(f"    - Representative node {j}: {mapped_nodes}")

        self.optimized_assignment_dict = optimized_assignment_dict

        return optimized_assignment_dict

    def cluster_KMedoids(self, n_repr=None):
        """
        Perform k-medoids clustering on the node features using the precomputed distance matrix.
        """
        distance_metrics = self.distance_metrics
        weights = self.config.weights
        total_distance_matrix = self.compute_distance_matrice(distance_metrics, weights)

        if n_repr is None:
            n_repr = self.config.n_repr

        initial_medoids = np.random.choice(range(self.num_original_nodes), n_repr, replace=False)
        kmedoids_instance = kmedoids(total_distance_matrix, initial_medoids.tolist(), data_type='distance_matrix')
        kmedoids_instance.process()

        kmedoids_clusters = kmedoids_instance.get_clusters()
        cluster_centers = kmedoids_instance.get_medoids()

        cluster_assignment_dict = {}
        for idx, cluster in enumerate(kmedoids_clusters):
            cluster_assignment_dict[cluster_centers[idx]] = cluster

        self.cluster_assignment_dict = cluster_assignment_dict

        return cluster_assignment_dict

    def compute_metrics(self, aggregation_method:str, type="custom"):
        """
        Compute all metrics based on the original and aggregated features.
        :return: Dictionary with metric values.
        """
        if aggregation_method == "kmedoids":
            if self.cluster_assignment_dict is None:
                raise ValueError("Clustering results are not available. Please run the KMedoids clustering first.")
            assignment_dict = self.cluster_assignment_dict

        elif aggregation_method == "optimization":
            if self.optimized_assignment_dict is None:
                raise ValueError("Optimization results are not available. Please run the optimization first.")
            assignment_dict = self.optimized_assignment_dict

        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}. Please choose between 'kmedoids' and 'optimization'.")
        
        # Calculate the error for each node to its cluster representative
        node_errors_list = []
        for node_id, original_features in self.nodes_features.items():
            representative_id = next((representative_id for representative_id, nodes in assignment_dict.items() if node_id in nodes), None)
            if representative_id is not None:
                aggregated_feature = self.nodes_features[representative_id]
                if type == "custom":
                    node_error_values = []
                    errors = self.compute_node_errors(original_features, aggregated_feature)
                    for error in errors.values():
                        node_error_values.append(error[0,0])
                    node_errors_list.append(node_error_values)

                elif type == "literature":
                    REEav = self.compute_reeav(original_features, aggregated_feature)
                    NRMSEav = self.compute_nrmseav(original_features, aggregated_feature)
                    CEav = self.compute_ceav(original_features, aggregated_feature)
                    NRMSERDCav = self.compute_nrmsedcav(original_features, aggregated_feature)
                    node_errors_list.append([REEav, NRMSEav, CEav, NRMSERDCav])

                else:
                    raise ValueError(f"Invalid metric type: {type}. Please choose between 'custom' and 'literature'.")

        # Calculate the mean error for the whole network
        node_errors_list = np.array(node_errors_list)
        if type == "custom":
            for i in range(len(node_errors_list[0])):
                node_errors_list[:, i] = self._normalize_distances(node_errors_list[:, i])
        
        mean_error_values = np.mean(node_errors_list, axis=0)

        if type == "custom":
            weights = self.config.weights
            metrics = {'total': mean_error_values[1] * weights['position'] + mean_error_values[2] * weights['time_series'] + mean_error_values[3] * weights['duration_curves'] + mean_error_values[4] * weights['rdc'] + mean_error_values[5] * weights['intra_correlation'] + mean_error_values[6] * weights['inter_correlation'] + mean_error_values[7] * weights['supply_demand_mismatch'],
                    'position': mean_error_values[1],
                    'time_series': mean_error_values[2],
                    'duration_curves': mean_error_values[3],
                    'rdc': mean_error_values[4],
                    'intra_correlation': mean_error_values[5],
                    'inter_correlation': mean_error_values[6],
                    'supply_demand_mismatch': mean_error_values[7]}
        elif type == "literature":
            metrics = {
                "REEav": mean_error_values[0],
                "NRMSEav": mean_error_values[1],
                "CEav": mean_error_values[2],
                "NRMSERDCav": mean_error_values[3]
            }
        return metrics

    def compute_node_errors(self, original_features, aggregated_feature):
        """
        Compute the error for a node to its cluster representative using the computed distance metrics.
        """
        # Create temporary node features dictionaries for distance calculation
        temp_original_features = {0: original_features}
        temp_aggregated_features = {0: aggregated_feature}
        distance_metrics = self.compute_distance_metrics(temp_original_features, temp_aggregated_features, normalize=False)
    
        return distance_metrics

    @staticmethod
    def compute_reeav(original_features, aggregated_features):
        """
        Compute Relative Energy Error Average (REEav).
        """
        for key in original_features["duration_curves"]:
            orig_dc = original_features["duration_curves"][key]
            agg_dc = aggregated_features["duration_curves"][key]
            total_energy_orig = np.sum(orig_dc)
            total_energy_agg = np.sum(agg_dc)
            relative_error = abs(total_energy_orig -
                                    total_energy_agg) / total_energy_orig
        return relative_error

    @staticmethod
    def compute_nrmseav(original_features, aggregated_features):
        """
        Compute Normalized Root Mean Square Error Average (NRMSEav).
        """
        for key in original_features["duration_curves"]:
            orig_dc = original_features["duration_curves"][key]
            agg_dc = aggregated_features["duration_curves"][key]
            mse = np.mean((orig_dc - agg_dc) ** 2)
            nrmse = np.sqrt(mse) / (np.max(orig_dc) - np.min(orig_dc))
        return nrmse

    @staticmethod
    def compute_ceav(original_features, aggregated_features):
        """
        Compute Correlation Error Average (CEav).
        """

        for key in original_features["correlation"]:
            orig_corr = original_features["correlation"][key]
            agg_corr = aggregated_features["correlation"][key]
            error = abs(orig_corr - agg_corr)
        return error

    @staticmethod
    def compute_nrmsedcav(original_features, aggregated_features):
        """
        Compute Normalized RMS Error of Ramp Duration Curve Average (NRMSERDCav).
        """
        for key in original_features["ramp_duration_curves"]:
            orig_rdc = original_features["ramp_duration_curves"][key]
            agg_rdc = aggregated_features["ramp_duration_curves"][key]
            mse = np.mean((orig_rdc - agg_rdc) ** 2)
            nrmse = np.sqrt(mse) / (np.max(orig_rdc) - np.min(orig_rdc))
        return nrmse
    
    def update_config(self, new_config: utils.Config):
        if self.config != new_config:
            print("Config has been updated.")
            self.config = new_config
        else:
            print("Config has not changed.")




class TemporalAggregation:
    def __init__(self, spatialaggregator: SpatialAggregation, assignment_dict, spatial_aggregation_method = "kmedoids"):

        self.aggregator = spatialaggregator
        self.config = self.aggregator.config
        self.spatial_aggregation_method = spatial_aggregation_method
        self.nodes_features = self.aggregator.nodes_features

        self.assignment_dict = assignment_dict
        # if self.spatial_aggregation_method == "kmedoids":
        #     if self.aggregator.cluster_assignment_dict is None:
        #         raise ValueError("Clustering results are not available. Please run the KMedoids spatial clustering first.")
        #     self.assignment_dict = self.aggregator.cluster_assignment_dict

        # elif self.spatial_aggregation_method == "optimization":
        #     if self.aggregator.optimized_assignment_dict is None:
        #         raise ValueError("Optimization results are not available. Please run the spatial optimization first.")
        #     self.assignment_dict = self.aggregator.optimized_assignment_dict
        # else:
        #     raise ValueError(f"Invalid spatial aggregation method: {self.spatial_aggregation_method}. Please choose between 'kmedoids' and 'optimization'.")

    def aggregate(self):
        """
        Aggregate the time series data of the nodes based on the spatial aggregation results.
        """
        sampled_nodes_time_serie = self._sampling()
        day_array = self._create_day_array(sampled_nodes_time_serie)
        representative_days = self._get_representative_days(day_array)
        filtered_dict = {representative_node: {key: self._filter_time_series(value, representative_days) for key, value in time_series.items()} for representative_node, time_series in sampled_nodes_time_serie.items()}
        return filtered_dict
        

    def _sampling(self):
        sampled_nodes_time_serie = {}
        for representative, nodes in self.assignment_dict.items():
            sampled_node = np.random.choice(nodes)
            sampled_nodes_time_serie[representative] = self.nodes_features[sampled_node]['time_series']
        return sampled_nodes_time_serie

    @staticmethod
    def _create_day_array(nodes_time_serie_dict):

        nodes_ts_list = []
        total_hours = list(list(nodes_time_serie_dict.values())[0].values())[0].shape[0]
        number_of_days = total_hours // 24

        for representative, time_series_dict in nodes_time_serie_dict.items():
            time_series = np.array(list(time_series_dict.values()))
            reshaped_time_series = time_series[:, :number_of_days * 24].reshape(time_series.shape[0], number_of_days, 24)
            transposed_time_series = reshaped_time_series.transpose(1, 0, 2)
            flattened_days = transposed_time_series.reshape(number_of_days, -1)
            nodes_ts_list.append(flattened_days)

        nodes_ts_array = np.array(nodes_ts_list)
        reshaped_data = nodes_ts_array.transpose(1, 0, 2)
        final_data = reshaped_data.reshape(number_of_days, -1)
       
        return final_data
    
    def _get_representative_days(self, data):

        number_of_days = data.shape[0]
        k_representative_days = self.config.k_representative_days
        distance_matrix = cdist(data, data, metric='euclidean')

        initial_medoids = np.random.choice(number_of_days, k_representative_days, replace=False)
        kmedoids_instance = kmedoids(distance_matrix, initial_index_medoids=initial_medoids)

        kmedoids_instance.process()
        representative_days = kmedoids_instance.get_medoids()

        return representative_days
    
    @staticmethod
    def _filter_time_series(time_series, representative_days):
        filtered_data = []
        for day in representative_days:
            start_index = day * 24
            end_index = start_index + 24
            filtered_data.extend(time_series[start_index:end_index])
        return filtered_data

    