import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from logger_utils import LoggerUtils
from matplotlib import pyplot as plt
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from dataset_handler import DatasetHandler  # Make sure to import the DatasetHandler class
import wandb

os.environ['WANDB_API_KEY'] = '2cfad53cf20bc18b3968eafc92a4aedfefbb7af8'


class ClusteringEnsemble:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')
        self.ds_handler = DatasetHandler()
        wandb.init(project="ML2023")
        self.algs = {
            'KMeans': KMeans(n_clusters=2),
            'DBSCAN': DBSCAN(min_samples=7),
            'OPTICS': OPTICS(min_samples=7),
            'Agglomerative': AgglomerativeClustering(n_clusters=2)
        }
        self.figure_ext = '4_algos'

    def compare_ari_heatmap(self, labels, algs_names):
        try:
            ari_matrix = np.zeros((len(algs_names), len(algs_names)))
            for i, (alg1_name, alg1_labels) in enumerate(zip(algs_names, labels)):
                for j, (alg2_name, alg2_labels) in enumerate(zip(algs_names, labels)):
                    ari_matrix[i, j] = adjusted_rand_score(alg1_labels, alg2_labels)
            ari_df = pd.DataFrame(ari_matrix, columns=algs_names)
            ari_df.index = algs_names
            plt.figure(figsize=(8, 6))
            sns.heatmap(ari_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
            plt.title("Clustering Comparison Heat-Map (Adjusted Rand Index)")
            # Ensure all elements are visible in the saved image
            plt.tight_layout()
            # Save the figure
            figure_name = 'Clustering Comparison Heat-Map (Adjusted Rand Index)_%s' % self.figure_ext
            figure_path = './figures/%s.png' % figure_name
            plt.savefig(figure_path)
            # Log the figure to wandb
            wandb.log({figure_name: wandb.Image(figure_path)})
            if self.figure_ext == '4_algos':
                self.figure_ext = '5_algos_plus_meta'
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during compare_ari_heatmap.' % e)

    def create_co_occurrence_matrix(self, labels):
        try:
            num_samples = len(labels)
            co_occurrence_matrix = np.zeros((num_samples, num_samples))
            for sample1, sample2 in itertools.product(range(num_samples), repeat=2):
                if labels[sample1] == labels[sample2]:
                    co_occurrence_matrix[sample1, sample2] += 1
            return co_occurrence_matrix
        except Exception as e:
            self.logger.error('Exception %s occurred during create_co_occurrence_matrix.' % e)

    def meta_clustering(self, affinity_matrix):
        try:
            spectral_cluster = SpectralClustering(n_clusters=2, affinity='precomputed')
            meta_labels = spectral_cluster.fit_predict(affinity_matrix)
            return meta_labels
        except Exception as e:
            self.logger.error('Exception %s occurred during meta_clustering.' % e)

    def run(self):
        try:
            dataset = self.ds_handler.load_south_german_credit()
            dataset.drop(columns=[dataset.columns[-1]], inplace=True)
            # Select the best hyper-parameter settings for other algorithms based on optimal k
            selected_labels = []
            for alg_name, alg in self.algs.items():
                selected_labels.append(alg.fit_predict(dataset))  # You need to select proper hyper-parameters here
            # Compare clustering results
            all_labels = selected_labels
            algs_names = list(self.algs.keys())
            self.compare_ari_heatmap(all_labels, algs_names)
            # Create co-occurrence matrix for each algorithm
            co_occurrence_matrices = [self.create_co_occurrence_matrix(labels) for labels in all_labels]
            # Aggregate co-occurrence matrices into one affinity matrix
            affinity_matrix = sum(co_occurrence_matrices)
            # Perform meta-clustering
            meta_labels = self.meta_clustering(affinity_matrix)
            # Compare clustering results again
            all_labels.append(meta_labels)
            algs_names.append('Meta-Clustering')
            self.compare_ari_heatmap(all_labels, algs_names)
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    clustering_ensemble = ClusteringEnsemble()
    clustering_ensemble.run()
