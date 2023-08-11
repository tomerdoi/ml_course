from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin

from optimal_k import OptimalK
from logger_utils import LoggerUtils


class Pipeline(ABC):
    def __init__(self):
        self.optimal_k = OptimalK()
        self.logger_util = LoggerUtils()

    @abstractmethod
    def run_pipeline(self, datasets) -> dict:
        raise NotImplemented

    def fill_in_invalid_run_metrics(self):
        try:
            metrics = {
                'num_of_clusters': 0,
                'unique_labels': [],
                'SSE-Elbow': 0.0,
                'VRC': 0.0,
                'DB': 0.0,
                'Silhouette': 0.0,
                'My_clustring_metric': 0.0
            }
            return metrics
        except Exception as e:
            self.logger.error('Exception %s occurred during fill_in_invalid_run_metrics.' % e)

    def measure_clustering_metrics(self, hp_name: str, hp_value: int, clustering_model: ClusterMixin,
                                   dataset: pd.DataFrame):
        try:
            # drop the last column of the dataset
            true_labels = dataset.iloc[:, -1].tolist()
            dataset = dataset.drop(dataset.columns[-1], axis=1)
            labels = clustering_model.fit_predict(dataset)
            unique_labels = np.unique(labels).tolist()
            num_of_clusters = len(set(labels))
            metrics = {
                'num_of_clusters': num_of_clusters,
                'unique_labels': unique_labels,
                'SSE-Elbow': self.optimal_k.elbow_method_metric(hp_name, hp_value, clustering_model, dataset, labels),
                'VRC': self.optimal_k.variance_ratio_criterion_metric(hp_name, hp_value, clustering_model, dataset,
                                                                      labels),
                'DB': self.optimal_k.davies_bouldin_metric(hp_name, hp_value, clustering_model, dataset, labels),
                'Silhouette': self.optimal_k.silhouette_metric(hp_name, hp_value, clustering_model, dataset, labels),
                'My_clustring_metric': self.optimal_k.custom_clustering_validity_metric(hp_name, hp_value,
                                                                                        clustering_model, dataset,
                                                                                        labels, true_labels)
            }
            return metrics
        except Exception as e:
            self.logger.error('Exception %s occurred during measure_clustering_metrics.' % e)
