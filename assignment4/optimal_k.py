from logger_utils import LoggerUtils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class OptimalK:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='optimal_k.log')

    # todo: Need to run 1 time on 1 K value and get results, and not iterating over multiple K values
    def elbow_method_metric(self, k, clustering_model, data):
        try:
            clustering_model.n_clusters = k
            labels = clustering_model.fit_predict(data)
            metric_value = clustering_model.inertia_  # SSE value for the current k
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during elbow_method_metric.' % e)

    def variance_ratio_criterion_metric(self, k, clustering_model, data):
        try:
            clustering_model.n_clusters = k
            labels = clustering_model.fit_predict(data)
            metric_value = calinski_harabasz_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during variance_ratio_criterion_metric.' % e)

    def davies_bouldin_metric(self, k, clustering_model, data):
        try:
            clustering_model.n_clusters = k
            labels = clustering_model.fit_predict(data)
            metric_value = davies_bouldin_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during davies_bouldin_metric.' % e)

    def silhouette_metric(self, k, clustering_model, data):
        try:
            clustering_model.n_clusters = k
            labels = clustering_model.fit_predict(data)
            metric_value = silhouette_score(data, labels)
            return metric_value
        except Exception as e:
            self.logger.error('Exception %s occurred during silhouette.' % e)

    def custom_clustering_validity_metric(self, k, clustering_model, data):
        try:
            # Calculate the maximum possible Davies-Bouldin Index for single-sample clusters
            max_dbi = davies_bouldin_score(data.iloc[:, :-1], data.iloc[:, -1].values.reshape(-1, 1))
            clustering_model.n_clusters = k
            labels = clustering_model.fit_predict(data)
            db_score = davies_bouldin_score(data.iloc[:, :-1], labels)
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
