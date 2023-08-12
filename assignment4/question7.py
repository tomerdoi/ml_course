import itertools
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

from dataset_handler import DatasetHandler


def compare_ari_heatmap(labels: list, algs_names: List[str]):
    ari_matrix = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            ari_matrix[i, j] = adjusted_rand_score(labels[i], labels[j])

    # Create a comparison matrix (ARI values)
    ari_df = pd.DataFrame(ari_matrix, columns=algs_names)
    ari_df.index = algs_names

    # Create a clustering heat-map using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(ari_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
    plt.title("Clustering Comparison Heat-Map (Adjusted Rand Index)")
    plt.show()


def create_co_occurrence_matrix(labels: list):
    num_samples = len(labels[0])
    num_algorithms = len(labels)

    # Initialize an empty co-occurrence matrix
    co_occurrence_matrix = np.zeros((num_samples, num_samples))

    # Calculate the co-occurrence matrix
    for i in range(num_algorithms):
        for j in range(num_algorithms):
            if i == j:  # Avoid comparing the same algorithm
                continue
            for sample1, sample2 in itertools.product(range(num_samples), repeat=2):
                if labels[i][sample1] == labels[j][sample2]:
                    co_occurrence_matrix[sample1, sample2] += 1

    # Print the co-occurrence matrix (you can visualize it as well)
    print("Co-Occurrence Matrix:")
    print(co_occurrence_matrix)


def main():
    ds_handler = DatasetHandler()
    # TODO: add the correct hp
    algs: Dict[str, ClusterMixin] = dict(KMeans=KMeans(n_clusters=30), DBSCAN=DBSCAN(), OPTICS=OPTICS(),
                                         Agglomerative=AgglomerativeClustering())

    dataset = ds_handler.load_parking_birmingham_data_set()
    dataset.drop(columns=[dataset.columns[-1]], inplace=True)

    labels = [alg.fit_predict(dataset) for alg in algs.values()]
    compare_ari_heatmap(labels, list(algs.keys()))
    create_co_occurrence_matrix(labels)


if __name__ == '__main__':
    main()
