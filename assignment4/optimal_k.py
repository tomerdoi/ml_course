from logger_utils import LoggerUtils
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class OptimalK:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='optimal_k.log')

    # todo: Need to run 1 time on 1 K value and get results, and not iterating over multiple K values
    def elbow_method_metric(self, k, clustering_model, data, labels):
        try:
            clustering_model.n_clusters = k
            if clustering_model.__class__.__name__ == 'DBSCAN':
                # Get the core points and their coordinates
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[clustering_model.core_sample_indices_] = True
                core_points = data[core_samples_mask]
                # Ensure that core_points is 2D even if it contains only one core point
                if len(core_points.shape) == 1:
                    core_points = core_points.reshape(1, -1)
                # Calculate distances between all points and core points
                distances = cdist(data, core_points)
                # Find the nearest core point for each non-core point
                nearest_core_indices = np.min(distances, axis=1)
                metric_value = np.sum([dist ** 2 for dist in nearest_core_indices])
            elif clustering_model.__class__.__name__ == 'AgglomerativeClustering':
                linkage_matrix = clustering_model.children_
                # Calculate the number of data points in each merged cluster
                n_samples = data.shape[0]
                cluster_sizes = np.zeros(2 * n_samples - 1)
                cluster_sizes[:n_samples] = 1
                for i in range(n_samples - 1, 2 * n_samples - 1):
                    child_1, child_2 = linkage_matrix[i - n_samples]
                    cluster_sizes[i] = cluster_sizes[child_1] + cluster_sizes[child_2]
                # Calculate the SSE value for the current k
                centers = np.array([data[labels == j].mean(axis=0) for j in range(k)])
                distances = cdist(data, centers, 'euclidean')
                sse_per_cluster = np.array([np.sum(distances[labels == j] ** 2) for j in range(k)])
                metric_value = np.sum(sse_per_cluster)
            elif clustering_model.__class__.__name__ == 'OPTICS':
                # Calculate the OPTICS SSE variant considering only finite reachability distances
                reachability_distances = clustering_model.reachability_
                finite_reachability_distances = reachability_distances[np.isfinite(reachability_distances)]
                metric_value = np.sum(finite_reachability_distances ** 2)
            else:
                metric_value = clustering_model.inertia_  # SSE value for the current k
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during elbow_method_metric.' % e)

    def variance_ratio_criterion_metric(self, k, clustering_model, data, labels):
        try:
            clustering_model.n_clusters = k
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = calinski_harabasz_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during variance_ratio_criterion_metric.' % e)

    def davies_bouldin_metric(self, k, clustering_model, data, labels):
        try:
            clustering_model.n_clusters = k
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = davies_bouldin_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during davies_bouldin_metric.' % e)

    def silhouette_metric(self, k, clustering_model, data, labels):
        try:
            clustering_model.n_clusters = k
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = silhouette_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during silhouette_metric.' % e)

    def custom_clustering_validity_metric(self, k, clustering_model, data, labels, true_labels):
        try:
            # Calculate the maximum possible Davies-Bouldin Index for single-sample clusters
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            true_labels = np.array(true_labels).reshape(-1, 1)
            max_dbi = davies_bouldin_score(data, true_labels)
            clustering_model.n_clusters = k
            db_score = davies_bouldin_score(data, labels)
            # Normalize the Davies-Bouldin Index by dividing by the maximum possible score
            db_normalized = db_score / max_dbi
            metric_value = db_normalized
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during custom_clustering_validity_metric.' % e)

    def underscore_to_capital_space(self, s):
        words = s.split('_')
        capitalized_words = [word.capitalize() for word in words]
        return ' '.join(capitalized_words)

    def plot_optimal_k_figure(self, dataset_name, metric_name, dataset_results):
        try:
            K = list(dataset_results.keys())
            scores = [dataset_results[k][metric_name] for k in dataset_results]
            plt.figure(figsize=(8, 5))
            plt.plot(K, scores, 'bx-')
            metric_name = self.underscore_to_capital_space(metric_name)
            dataset_name = self.underscore_to_capital_space(dataset_name)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('%s Score' % metric_name)
            plt.title('%s Metric for Optimal k for dataset %s' % (metric_name, dataset_name))
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during plot_optimal_k_figure.' % e)


if __name__ == '__main__':
    optimal_k = OptimalK()
    pass
