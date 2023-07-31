from optimal_k import OptimalK
from sklearn.cluster import DBSCAN
from logger_utils import LoggerUtils
from dataset_handler import DatasetHandler


class DBSCANPipeline:
    def __init__(self):
        self.optimal_k = OptimalK()
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='dbscan_pipeline.log')

    def run_pipeline(self, datasets):
        try:
            min_samples_list = list(range(2, 22))
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running DBSCAN pipeline for dataset: {dataset_name}")
                dataset_results = {}
                for min_samples in min_samples_list:
                    print(f"min_samples = {min_samples}")
                    algo = DBSCAN(min_samples=min_samples, eps=3.0)
                    clustering_metrics = self.measure_clustering_metrics(algo, dataset)
                    dataset_results[min_samples] = clustering_metrics
                results[dataset_name] = dataset_results
            return results
        except Exception as e:
            self.logger.error('Exception %s occurred during run_pipeline.' % e)

    def measure_clustering_metrics(self, clustering_model, dataset):
        try:
            # drop the last column of the dataset
            dataset = dataset.drop(dataset.columns[-1], axis=1)
            metrics = {
                'variance_ratio_criterion': self.optimal_k.variance_ratio_criterion_metric(None, clustering_model,
                                                                                           dataset),
                'davies_bouldin': self.optimal_k.davies_bouldin_metric(None, clustering_model, dataset),
                'silhouette': self.optimal_k.silhouette_metric(None, clustering_model, dataset),
                'custom_clustering_validity': self.optimal_k.custom_clustering_validity_metric(None, clustering_model,
                                                                                               dataset)
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
    pipeline = DBSCANPipeline()
    results = pipeline.run_pipeline(datasets)
    print(results)
