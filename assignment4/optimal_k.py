from typing import Dict
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS, KMeans
from logger_utils import LoggerUtils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class OptimalK:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')

    # todo: Need to run 1 time on 1 K value and get results, and not iterating over multiple K values
    def elbow_method_metric(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin, data: pd.DataFrame,
                            labels: list) -> float:
        try:
            if len(set(labels)) == 1 and list(set(labels))[0] == -1:
                self.logger.error('Exception occurred during elbow_method_metric, '
                                  'len(set(labels)) == 1 and list(set(labels))[0] == -1')
                return None
            if isinstance(clustering_model, DBSCAN) or isinstance(clustering_model, AgglomerativeClustering) or \
                    isinstance(clustering_model, OPTICS) or isinstance(clustering_model, KMeans):
                metric_value = self.calculate_sse(data, labels)
            else:
                return None
            if metric_value:
                num_of_clusters = len(set(labels))
                print(f'{hp_name}: {hp_value}, num_of_clusters: {num_of_clusters}, SSE: {metric_value}')
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during elbow_method_metric.' % e)

    def calculate_sse(self, dataset, labels):
        try:
            unique_labels = np.unique(labels)
            sse = 0
            for label in unique_labels:
                cluster_points = dataset[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                squared_distances = np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
                sse += squared_distances
            return sse
        except Exception as e:
            self.logger.error('Exception %s occurred during calculate_agglomerative_sse.' % e)

    def variance_ratio_criterion_metric(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin,
                                        data: pd.DataFrame, labels: list):
        try:
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = calinski_harabasz_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during variance_ratio_criterion_metric.' % e)

    def davies_bouldin_metric(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin, data: pd.DataFrame,
                              labels: list):
        try:
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = davies_bouldin_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during davies_bouldin_metric.' % e)

    def silhouette_metric(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin, data: pd.DataFrame,
                          labels: list):
        try:
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            metric_value = silhouette_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during silhouette_metric.' % e)

    def custom_clustering_validity_metric(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin,
                                          data: pd.DataFrame, labels: list, true_labels: list):
        try:
            # Calculate the maximum possible Davies-Bouldin Index for single-sample clusters
            unique_labels = len(set(labels))
            if unique_labels < 2 or unique_labels >= len(data):
                return None  # Return None to indicate inability to evaluate clustering
            true_labels = np.array(true_labels).reshape(-1, 1)
            max_dbi = davies_bouldin_score(data, true_labels)
            clustering_model.n_clusters = hp_value
            db_score = davies_bouldin_score(data, labels)
            # Normalize the Davies-Bouldin Index by dividing by the maximum possible score
            db_normalized = db_score / max_dbi
            metric_value = db_normalized
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during custom_clustering_validity_metric.' % e)

    @staticmethod
    def _underscore_to_capital_space(s):
        words = s.split('_')
        capitalized_words = [word.capitalize() for word in words]
        return ' '.join(capitalized_words)

    def plot_optimal_k_figure(self, algo: str, dataset_name: str, metric_name: str,
                              dataset_results: Dict[int, Dict[str, float]]):
        try:
            self.logger.info('Plotting Elbow figure for dataset %s and metric %s.' % (dataset_name, metric_name))
            K = [result['num_of_clusters'] for result in dataset_results.values()]
            scores = [result[metric_name] for result in dataset_results.values()]
            none_score_indices = [i for i in range(len(scores)) if scores[i] is None]
            self.logger.info('none_score_indices for algo %s and metric %s is: %d' % (algo, metric_name,
                                                                                      len(none_score_indices)))
            K = [K[i] for i in range(len(K)) if i not in none_score_indices]
            scores = [scores[i] for i in range(len(scores)) if i not in none_score_indices]
            plt.figure(figsize=(8, 5))
            plt.plot(K, scores, 'bx-')
            dataset_name = self._underscore_to_capital_space(dataset_name)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('%s Score' % metric_name)
            plt.title('%s Algo %s Metric for Optimal k for dataset %s' % (algo, metric_name, dataset_name))
            plt.savefig(f'{algo} {dataset_name} {metric_name}.png')
            plt.show()
            # Clean up and reset Matplotlib state
            plt.close('all')
            plt.clf()
            plt.cla()
            plt.close()
            # Optional: Reset Matplotlib's interactive mode
            plt.ioff()
        except Exception as e:
            self.logger.error('Exception %s occurred during plot_optimal_k_figure.' % e)


if __name__ == '__main__':
    optimal_k = OptimalK()
    pass
