import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
import utils
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.utils import read_sample


class AggregationOptimizer:
    def __init__(self, node_features, n_repr):
        """
        Initialize the optimizer with nodes, time series, and aggregation parameters.
        """
        self.nodes_features = node_features
        self.n_repr = n_repr
        self.num_nodes = len(node_features)

    def distance_matrices(self, weights=None):
        """
        Compute the distance matrix between node features.
        """
        if weights is None:
            weights = {
                'position': 1.0,
                'time_series': 1.0,
                'duration_curves': 1.0,
                'rdc': 1.0,
                'correlation': 1.0
            }

        num_nodes = self.num_nodes
        dist_matrix = np.zeros((num_nodes, num_nodes))
        position_matrix = np.zeros((num_nodes, num_nodes))
        time_series_matrix = np.zeros((num_nodes, num_nodes))
        duration_curves_matrix = np.zeros((num_nodes, num_nodes))
        rdc_matrix = np.zeros((num_nodes, num_nodes))
        correlation_matrix = np.zeros((num_nodes, num_nodes))

        # Initialize min and max values for normalization
        min_position_distance = float('inf')
        max_position_distance = float('-inf')
        min_time_series_distance = float('inf')
        max_time_series_distance = float('-inf')
        min_duration_curves_distance = float('inf')
        max_duration_curves_distance = float('-inf')
        min_rdc_distance = float('inf')
        max_rdc_distance = float('-inf')
        min_correlation_distance = float('inf')
        max_correlation_distance = float('-inf')

        # First pass to find min and max values for normalization
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Position distance using Haversine formula
                lat1, lon1 = self.nodes_features[i]['position']
                lat2, lon2 = self.nodes_features[j]['position']
                position_distance = utils.haversine(lat1, lon1, lat2, lon2)
                min_position_distance = min(
                    min_position_distance, position_distance)
                max_position_distance = max(
                    max_position_distance, position_distance)

                # Time series distance using Euclidean norm
                time_series_distance = 0
                for key in self.nodes_features[i]['time_series'].keys():
                    ts1 = self.nodes_features[i]['time_series'][key]
                    ts2 = self.nodes_features[j]['time_series'][key]
                    time_series_distance += np.linalg.norm(ts1 - ts2)
                min_time_series_distance = min(
                    min_time_series_distance, time_series_distance)
                max_time_series_distance = max(
                    max_time_series_distance, time_series_distance)

                # Duration curves distance using Euclidean norm
                duration_curves_distance = 0
                for key in self.nodes_features[i]['duration_curves'].keys():
                    dc1 = self.nodes_features[i]['duration_curves'][key]
                    dc2 = self.nodes_features[j]['duration_curves'][key]
                    duration_curves_distance += np.linalg.norm(dc1 - dc2)
                min_duration_curves_distance = min(
                    min_duration_curves_distance, duration_curves_distance)
                max_duration_curves_distance = max(
                    max_duration_curves_distance, duration_curves_distance)

                # RDC distance using Euclidean norm
                rdc_distance = 0
                for key in self.nodes_features[i]['ramp_duration_curves'].keys():
                    rdc1 = self.nodes_features[i]['ramp_duration_curves'][key]
                    rdc2 = self.nodes_features[j]['ramp_duration_curves'][key]
                    rdc_distance += np.linalg.norm(rdc1 - rdc2)
                min_rdc_distance = min(min_rdc_distance, rdc_distance)
                max_rdc_distance = max(max_rdc_distance, rdc_distance)

                # Correlation distance using absolute difference
                correlation_distance = 0
                for key in self.nodes_features[i]['correlation'].keys():
                    corr1 = self.nodes_features[i]['correlation'][key]
                    corr2 = self.nodes_features[j]['correlation'][key]
                    correlation_distance += abs(corr1 - corr2)
                min_correlation_distance = min(
                    min_correlation_distance, correlation_distance)
                max_correlation_distance = max(
                    max_correlation_distance, correlation_distance)

        # Second pass to compute normalized and weighted distances
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Position distance using Haversine formula
                lat1, lon1 = self.nodes_features[i]['position']
                lat2, lon2 = self.nodes_features[j]['position']
                position_distance = utils.haversine(lat1, lon1, lat2, lon2)
                normalized_position_distance = (
                    position_distance - min_position_distance) / (max_position_distance - min_position_distance)
                position_matrix[i, j] = normalized_position_distance
                position_matrix[j, i] = normalized_position_distance

                # Time series distance using Euclidean norm
                time_series_distance = 0
                for key in self.nodes_features[i]['time_series'].keys():
                    ts1 = self.nodes_features[i]['time_series'][key]
                    ts2 = self.nodes_features[j]['time_series'][key]
                    time_series_distance += np.linalg.norm(ts1 - ts2)
                normalized_time_series_distance = (time_series_distance - min_time_series_distance) / (
                    max_time_series_distance - min_time_series_distance)
                time_series_matrix[i, j] = normalized_time_series_distance
                time_series_matrix[j, i] = normalized_time_series_distance

                # Duration curves distance using Euclidean norm
                duration_curves_distance = 0
                for key in self.nodes_features[i]['duration_curves'].keys():
                    dc1 = self.nodes_features[i]['duration_curves'][key]
                    dc2 = self.nodes_features[j]['duration_curves'][key]
                    duration_curves_distance += np.linalg.norm(dc1 - dc2)
                normalized_duration_curves_distance = (duration_curves_distance - min_duration_curves_distance) / (
                    max_duration_curves_distance - min_duration_curves_distance)
                duration_curves_matrix[i,
                                       j] = normalized_duration_curves_distance
                duration_curves_matrix[j,
                                       i] = normalized_duration_curves_distance

                # RDC distance using Euclidean norm
                rdc_distance = 0
                for key in self.nodes_features[i]['ramp_duration_curves'].keys():
                    rdc1 = self.nodes_features[i]['ramp_duration_curves'][key]
                    rdc2 = self.nodes_features[j]['ramp_duration_curves'][key]
                    rdc_distance += np.linalg.norm(rdc1 - rdc2)
                normalized_rdc_distance = (
                    rdc_distance - min_rdc_distance) / (max_rdc_distance - min_rdc_distance)
                rdc_matrix[i, j] = normalized_rdc_distance
                rdc_matrix[j, i] = normalized_rdc_distance

                # Correlation distance using absolute difference
                correlation_distance = 0
                for key in self.nodes_features[i]['correlation'].keys():
                    corr1 = self.nodes_features[i]['correlation'][key]
                    corr2 = self.nodes_features[j]['correlation'][key]
                    correlation_distance += abs(corr1 - corr2)
                normalized_correlation_distance = (correlation_distance - min_correlation_distance) / (
                    max_correlation_distance - min_correlation_distance)
                correlation_matrix[i, j] = normalized_correlation_distance
                correlation_matrix[j, i] = normalized_correlation_distance

                # Sum all normalized and weighted distances to get the total distance
                total_distance = (weights['position'] * normalized_position_distance +
                                  weights['time_series'] * normalized_time_series_distance +
                                  weights['duration_curves'] * normalized_duration_curves_distance +
                                  weights['rdc'] * normalized_rdc_distance +
                                  weights['correlation'] * normalized_correlation_distance)

                dist_matrix[i, j] = total_distance
                dist_matrix[j, i] = total_distance

        return {
            'total_distance': dist_matrix,
            'position_distance': position_matrix,
            'time_series_distance': time_series_matrix,
            'duration_curves_distance': duration_curves_matrix,
            'rdc_distance': rdc_matrix,
            'correlation_distance': correlation_matrix
        }

    def optimize(self, weights=None):
        """
        Formulate and solve the optimization model.
        """
        # Compute the distance matrix
        n_repr = self.n_repr
        distance_matrices = self.distance_matrices(weights=weights)
        total_distance_matrix = distance_matrices['total_distance']

        num_nodes = self.num_nodes

        # Initialize the model
        model = gp.Model("node_aggregation")

        # Define binary variables
        u = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY,
                          name="u")  # Assignment variables
        # Representative selection variables
        z = model.addVars(num_nodes, vtype=GRB.BINARY, name="z")

        # Objective: Minimize the total distance
        model.setObjective(
            gp.quicksum(u[i, j] * total_distance_matrix[i, j]
                        for i in range(num_nodes) for j in range(num_nodes)),
            GRB.MINIMIZE
        )

        # Constraint 1: Each node is assigned to exactly one representative
        model.addConstrs(
            (gp.quicksum(u[i, j] for j in range(num_nodes))
             == 1 for i in range(num_nodes)),
            "assignment"
        )

        # Constraint 2: A node can only be assigned to an active representative
        model.addConstrs(
            (u[i, j] <= z[j] for i in range(num_nodes)
             for j in range(num_nodes)),
            "representative_assignment"
        )

        # Constraint 3: Exactly n_repr representatives are chosen
        model.addConstr(
            gp.quicksum(z[j] for j in range(num_nodes)) == n_repr,
            "num_representatives"
        )

        # Optimize the model
        model.optimize()

        # Extract the results
        u_result = np.zeros((num_nodes, num_nodes))
        z_result = np.zeros(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if u[i, j].X > 0.5:
                    u_result[i, j] = 1
        for j in range(num_nodes):
            if z[j].X > 0.5:
                z_result[j] = 1

        # Print the optimization results
        print("\nOptimization Results:")
        print(f"  - Objective Value: {model.objVal}")

        # Print the representatives and their mappings
        representatives = [j for j in range(num_nodes) if z_result[j] == 1]
        print(f"  - Representatives: {representatives}")
        for j in representatives:
            mapped_nodes = [i for i in range(num_nodes) if u_result[i, j] == 1]
            print(f"    - Representative node {j}: {mapped_nodes}")

        return u_result, z_result


class AggregationClustering:
    def __init__(self, node_features, n_repr):
        """
        Initialize the k-means aggregator with node features and the number of representative nodes.
        """
        self.node_features = node_features
        self.n_repr = n_repr
        self.num_nodes = len(node_features)

    def compute_feature_matrix(self):
        """
        Combine node features into a single feature matrix for clustering.
        """
        # Example: Flattening the position and other features into a feature vector
        feature_matrix = []
        for node in self.node_features:
            position = self.node_features[node]['position']
            time_series = np.concatenate(
                [ts for ts in self.node_features[node]['time_series'].values()])
            duration_curves = np.concatenate(
                [dc for dc in self.node_features[node]['duration_curves'].values()])
            rdc = np.concatenate(
                [rdc for rdc in self.node_features[node]['ramp_duration_curves'].values()])
            correlation = np.array(
                list(self.node_features[node]['correlation'].values()))
            feature_vector = np.concatenate(
                [position, time_series, duration_curves, rdc, correlation])
            feature_matrix.append(feature_vector)

        return np.array(feature_matrix)

    def cluster_KMedoids(self):
        """
        Perform k-medoids clustering on the node features.
        """
        feature_matrix = self.compute_feature_matrix()
        initial_medoids = np.random.choice(
            range(len(feature_matrix)), self.n_repr, replace=False)

        # Use K-Medoids clustering
        kmedoids_instance = kmedoids(
            feature_matrix,
            initial_medoids.tolist(),
            metric=distance_metric(type_metric.EUCLIDEAN)
        )
        kmedoids_instance.process()

        # Cluster results
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        # Create a mapping of clusters
        cluster_mapping = {i: [] for i in range(len(medoids))}
        for idx, cluster in enumerate(clusters):
            cluster_mapping[idx] = cluster

        return cluster_mapping, medoids

    def cluster_KMeans(self):
        """
        Perform k-means clustering on the node features.
        """
        feature_matrix = self.compute_feature_matrix()

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_repr, random_state=0)
        kmeans.fit(feature_matrix)

        # Cluster labels for each node
        labels = kmeans.labels_

        # Mapping each cluster to its nodes
        clusters = {i: [] for i in range(self.n_repr)}
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        return clusters, kmeans.cluster_centers_, labels

#     def cluster_KMedoids(self):
#         """
#         Perform k-medoids clustering on the node features.
#         """
#         feature_matrix = self.compute_feature_matrix()

#         # Apply k-medoids clustering
#         kmedoids = KMedoids(n_clusters=self.n_repr, random_state=0)
#         kmedoids.fit(feature_matrix)

#         # Cluster labels for each node
#         labels = kmedoids.labels_

#         # Mapping each cluster to its nodes
#         clusters = {i: [] for i in range(self.n_repr)}
#         for idx, label in enumerate(labels):
#             clusters[label].append(idx)

#         # Medoids (representative nodes)
#         medoids = kmedoids.medoid_indices_

#         return clusters, medoids, labels

#     def cluster(self, method='KMeans'):
