from sklearn.cluster import DBSCAN
from pipeline import Pipeline
from dataset_handler import DatasetHandler


class DBSCANPipeline(Pipeline):
    def __init__(self):
        super().__init__()  # Call the parent class constructor first
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')

    def run_pipeline(self, datasets):
        try:
            min_samples_dict = {'south_german_credit_ds': list(range(2, 22)),
                                'parking_birmingham_ds': list(range(10, 210, 10)),
                                'icmla_2014_accepted_papers_ds': list(range(1, 100, 5))}
            eps_dict = {'south_german_credit_ds': 3.0,
                        'parking_birmingham_ds': 0.5,
                        'icmla_2014_accepted_papers_ds': 70.0}
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running DBSCAN pipeline for dataset: {dataset_name}")
                dataset_results = {}
                min_samples_list = min_samples_dict[dataset_name]
                min_samples_list = min_samples_list[:5]
                for min_samples in min_samples_list:
                    print(f"min_samples = {min_samples}")
                    algo = DBSCAN(min_samples=min_samples, eps=eps_dict[dataset_name])
                    clustering_metrics = self.measure_clustering_metrics('min_samples', min_samples, algo, dataset)
                    dataset_results[min_samples] = clustering_metrics
                self.optimal_k.plot_optimal_k_figure('DBSCAN', dataset_name, 'SSE-Elbow', dataset_results)
                results[dataset_name] = dataset_results
            return results
        except Exception as e:
            self.logger.error('Exception %s occurred during run_pipeline.' % e)


if __name__ == '__main__':
    # Assuming you have a dictionary 'datasets' with the datasets you want to analyze
    # For example: datasets = {'dataset1': df1, 'dataset2': df2, ...}
    ds_handler = DatasetHandler()
    south_german_credit_ds = ds_handler.load_south_german_credit()
    icmla_2014_accepted_papers_ds = ds_handler.load_icmla_2014_accepted_papers_data_set()
    parking_birmingham_ds = ds_handler.load_parking_birmingham_data_set()
    datasets = {'south_german_credit_ds': south_german_credit_ds,
                'parking_birmingham_ds': parking_birmingham_ds,
                'icmla_2014_accepted_papers_ds': icmla_2014_accepted_papers_ds}
    pipeline = DBSCANPipeline()
    results = pipeline.run_pipeline(datasets)
    print(results)
