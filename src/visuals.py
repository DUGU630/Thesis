import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import models as models

class Visualization:
    def __init__(self, spatialaggregator: models.SpatialAggregation, save_fig=False, save_dir=None):
        """
        Initialize the visualization with the network, optimizer, and aggregation results.
        """
        self.aggregator = spatialaggregator

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

    def plot_map(self, aggregation_method='optimization'):
        """
        Plot the map of nodes with representative nodes highlighted.

        Parameters:
        method (str): The method used for clustering ('optimization', 'kmeans', 'kmedoids').
        """
        nodes_features = self.aggregator.nodes_features
        original_num_nodes = len(nodes_features)
        n_repr = self.aggregator.config.n_repr

        # Generate a colormap for the clusters
        colormap = plt.get_cmap('viridis', n_repr)

        # Plot the nodes
        fig = plt.figure(figsize=(7, 7))

        if aggregation_method == 'optimization':
            if self.aggregator.optimized_assignment_dict is None:
                raise ValueError("Optimization results are not available. Please run the optimization first.")
            assignment_dict = self.aggregator.optimized_assignment_dict
                

        elif aggregation_method == 'kmedoids':
            if self.aggregator.cluster_assignment_dict is None:
                raise ValueError("Clustering results are not available. Please run the KMedoids clustering first.")
            assignment_dict = self.aggregator.cluster_assignment_dict

        else:
            raise ValueError("Invalid aggregation method. Please choose 'optimization', or 'kmedoids'.")

        cluster_idx = 0
        for representative_id, nodes in assignment_dict.items():
            cluster_color = colormap(cluster_idx)
            for node_id in nodes:
                if node_id != representative_id:
                    node_coords_lat, node_coords_lon = nodes_features[node_id]['position']
                    plt.plot(-node_coords_lon, node_coords_lat, 'o', color=cluster_color)
            rep_node_coords_lat, rep_node_coords_lon = nodes_features[representative_id]['position']
            plt.plot(-rep_node_coords_lon, rep_node_coords_lat, 'kx')
            cluster_idx += 1

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Node Aggregation Map")
        weights_str = ', '.join([f"{key}: {value}" for key, value in self.aggregator.config.weights.items()])
        plt.suptitle(f"Weights: {weights_str}", fontsize=10)
        plt.tight_layout()
        plt.show()

        if self.save_fig:
            if aggregation_method == 'optimization':
                self.save_figure(fig, f"opti_agg_map_{original_num_nodes}_to_{n_repr}.png")
            elif aggregation_method == 'kmedoids':
                self.save_figure(fig, f"kmedoids_agg_map_{original_num_nodes}_to_{n_repr}.png")