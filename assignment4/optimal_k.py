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
    def elbow_method(self, clustering_model, data):
        try:
            sse = []  # List to store the SSE values for each k
            K = range(1, clustering_model.n_clusters + 1)
            for k in K:
                clustering_model.n_clusters = k
                labels = clustering_model.fit_predict(data)
                sse.append(clustering_model.inertia_)  # SSE value for the current k
            # Plot the SSE values against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, sse, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during elbow_method.' % e)

    def variance_ratio_criterion(self, clustering_model, data):
        try:
            calinski_scores = []
            K = range(2, clustering_model.n_clusters + 1)
            for k in K:
                clustering_model.n_clusters = k
                labels = clustering_model.fit_predict(data)
                calinski_scores.append(calinski_harabasz_score(data, labels))
            # Plot the Calinski and Harabasz scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, calinski_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski and Harabasz Score')
            plt.title('Variance Ratio Criterion for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during variance_ratio_criterion.' % e)

    def davies_bouldin(self, clustering_model, data):
        try:
            db_scores = []
            K = range(2, clustering_model.n_clusters + 1)
            for k in K:
                clustering_model.n_clusters = k
                labels = clustering_model.fit_predict(data)
                db_scores.append(davies_bouldin_score(data, labels))
            # Plot the Davies-Bouldin scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, db_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Davies-Bouldin Score')
            plt.title('Davies-Bouldin Score for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during davies_bouldin.' % e)

    def silhouette(self, clustering_model, data):
        try:
            silhouette_scores = []
            K = range(2, clustering_model.n_clusters + 1)
            for k in K:
                clustering_model.n_clusters = k
                labels = clustering_model.fit_predict(data)
                silhouette_scores.append(silhouette_score(data, labels))
            # Plot the Silhouette scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, silhouette_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during silhouette.' % e)

    # custom method
    def custom_clustering_validity(self, clustering_model, data):
        try:
            db_normalized_scores = []
            K = range(2, clustering_model.n_clusters + 1)
            # Calculate the maximum possible Davies-Bouldin Index for single-sample clusters
            max_dbi = davies_bouldin_score(data.iloc[:, :-1], data.iloc[:, -1].values.reshape(-1, 1))
            for k in K:
                clustering_model.n_clusters = k
                labels = clustering_model.fit_predict(data)
                db_score = davies_bouldin_score(data.iloc[:, :-1], labels)
                # Normalize the Davies-Bouldin Index by dividing by the maximum possible score
                db_normalized = db_score / max_dbi
                db_normalized_scores.append(db_normalized)
            # Plot the normalized Davies-Bouldin scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, db_normalized_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Normalized Davies-Bouldin Score')
            plt.title('Custom Clustering Validity Metric for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during custom_clustering_validity.' % e)


if __name__ == '__main__':
    optimal_k = OptimalK()
    pass
