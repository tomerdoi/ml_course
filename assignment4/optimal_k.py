from logger_utils import LoggerUtils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class OptimalK:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='optimal_k.log')

    def elbow_method(self, df, max_clusters=10):
        try:
            sse = []  # List to store the SSE values for each k
            K = range(1, max_clusters + 1)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df.iloc[:, :-1])  # Fit K-Means to the numeric features (excluding the last column)
                sse.append(kmeans.inertia_)  # SSE value for the current k
            # Plot the SSE values against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, sse, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during elbow_method.' % e)

    def variance_ratio_criterion(self, df, max_clusters=10):
        try:
            calinski_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df.iloc[:, :-1])  # Fit K-Means to the numeric features (excluding the last column)
                labels = kmeans.labels_
                calinski_scores.append(calinski_harabasz_score(df.iloc[:, :-1], labels))
            # Plot the Calinski and Harabasz scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, calinski_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski and Harabasz Score')
            plt.title('Variance Ratio Criterion for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during variance_ratio_criterion.' % e)

    def davies_bouldin(self, df, max_clusters=10):
        try:
            db_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df.iloc[:, :-1])  # Fit K-Means to the numeric features (excluding the last column)
                labels = kmeans.labels_
                db_scores.append(davies_bouldin_score(df.iloc[:, :-1], labels))
            # Plot the Davies-Bouldin scores against different k values
            plt.figure(figsize=(8, 5))
            plt.plot(K, db_scores, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Davies-Bouldin Score')
            plt.title('Davies-Bouldin Score for Optimal k')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during davies_bouldin.' % e)

    def silhouette(self, df, max_clusters=10):
        try:
            silhouette_scores = []
            K = range(2, max_clusters + 1)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df.iloc[:, :-1])  # Fit K-Means to the numeric features (excluding the last column)
                labels = kmeans.labels_
                silhouette_scores.append(silhouette_score(df.iloc[:, :-1], labels))
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
    def custom_clustering_validity(self, df, max_clusters=10):
        try:
            db_normalized_scores = []
            K = range(2, max_clusters + 1)
            # Calculate the maximum possible Davies-Bouldin Index for single-sample clusters
            max_dbi = davies_bouldin_score(df.iloc[:, :-1], df.iloc[:, -1].values.reshape(-1, 1))
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df.iloc[:, :-1])  # Fit K-Means to the numeric features (excluding the last column)
                labels = kmeans.labels_
                db_score = davies_bouldin_score(df.iloc[:, :-1], labels)
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
