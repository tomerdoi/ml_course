from optimal_k import OptimalK
from logger_utils import LoggerUtils


class Pipeline:
    def __init__(self):
        self.optimal_k = OptimalK()
        self.logger_util = LoggerUtils()

    def run_pipeline(self, datasets):
       pass

    def measure_clustering_metrics(self, k, clustering_model, dataset):
        try:
            # drop the last column of the dataset
            true_labels = dataset.iloc[:, -1].tolist()
            dataset = dataset.drop(dataset.columns[-1], axis=1)
            labels = clustering_model.fit_predict(dataset)
            unique_labels = len(set(list(labels)))
            metrics = {
                'unique_labels': unique_labels,
                'SSE-Elbow': self.optimal_k.elbow_method_metric(k, clustering_model, dataset, labels),
                'VRC': self.optimal_k.variance_ratio_criterion_metric(k, clustering_model, dataset, labels),
                'DB': self.optimal_k.davies_bouldin_metric(k, clustering_model, dataset, labels),
                'Silhouette': self.optimal_k.silhouette_metric(k, clustering_model, dataset, labels),
                'My_clustring_metric': self.optimal_k.custom_clustering_validity_metric(k, clustering_model,
                                                                                        dataset, labels, true_labels)
            }
            return metrics
        except Exception as e:
            self.logger.error('Exception %s occurred during measure_clustering_metrics.' % e)
