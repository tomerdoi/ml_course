from optimal_k import OptimalK
from sklearn.cluster import KMeans
from logger_utils import LoggerUtils
from dataset_handler import DatasetHandler


class KMeansPipeline:
    def __init__(self):
        self.optimal_k = OptimalK()
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='optimal_k.log')

    def run_pipeline(self, datasets):
        try:
            k_values = list(range(1, 31)) + list(range(35, 100, 5)) + list(range(100, 1001, 25))
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running K-Means pipeline for dataset: {dataset_name}")
                dataset_results = {}
                for k in k_values:
                    print(f"K = {k}")
                    kmeans_result = self.run_clustering_algorithm(KMeans(n_clusters=k), dataset)
                    clustering_metrics = self.measure_clustering_metrics(KMeans(n_clusters=k), dataset, kmeans_result)
                    dataset_results[k] = clustering_metrics
                results[dataset_name] = dataset_results
            return results
        except Exception as e:
            self.logger.error('Exception %s occurred during run_pipeline.' % e)

    def run_clustering_algorithm(self, clustering_model, dataset):
        try:
            clustering_model.fit(dataset.iloc[:, :-1])  # Fit the clustering model to the numeric features (excluding the last column)
            return clustering_model
        except Exception as e:
            self.logger.error('Exception %s occurred during run_clustering_algorithm.' % e)

    def measure_clustering_metrics(self, clustering_model, dataset, clustering_result):
        try:
            labels = clustering_result.labels_
            metrics = {
                'elbow_method': self.optimal_k.elbow_method(clustering_model, dataset),
                'variance_ratio_criterion': self.optimal_k.variance_ratio_criterion(clustering_model, dataset),
                'davies_bouldin': self.optimal_k.davies_bouldin(clustering_model, dataset),
                'silhouette': self.optimal_k.silhouette(clustering_model, dataset),
                'custom_clustering_validity': self.optimal_k.custom_clustering_validity(clustering_model, dataset)
            }
            return metrics
        except Exception as e:
            self.logger.error('Exception %s occurred during measure_clustering_metrics.' % e)


if __name__ == '__main__':
    # Assuming you have a dictionary 'datasets' with the datasets you want to analyze
    # For example: datasets = {'dataset1': df1, 'dataset2': df2, ...}
    ds_handler = DatasetHandler()
    south_german_credit_ds = ds_handler.load_south_german_credit()
    icmla_2014_accepted_papers_ds = ds_handler.load_icmla_2014_accepted_papers_data_set()
    parking_birmingham_ds = ds_handler.load_parking_birmingham_data_set()
    datasets = {'south_german_credit_ds': south_german_credit_ds,
                'icmla_2014_accepted_papers_ds': icmla_2014_accepted_papers_ds,
                'parking_birmingham_ds': parking_birmingham_ds}
    pipeline = KMeansPipeline()
    results = pipeline.run_pipeline(datasets)
    print(results)
