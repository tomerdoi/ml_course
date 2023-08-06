from sklearn.cluster import AgglomerativeClustering
from dataset_handler import DatasetHandler
from pipeline import Pipeline


class AgglomerativePipeline(Pipeline):  # 2. Rename the class to AgglomerativePipeline
    def __init__(self):
        super().__init__()  # Call the parent class constructor first
        self.logger = self.logger_util.init_logger(log_file_name='agglomerative_pipeline.log')

    def run_pipeline(self, datasets):
        try:
            # todo: to check ranges, according to the assignments instructions it should include 1 which is impossible
            k_values = list(range(2, 31)) + list(range(35, 100, 5)) + list(range(100, 1001, 25))
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running Agglomerative pipeline for dataset: {dataset_name}")  # Update the message
                dataset_results = {}
                for k in k_values:
                    print(f"K = {k}")
                    algo = AgglomerativeClustering(n_clusters=k)  # Use AgglomerativeClustering
                    clustering_metrics = self.measure_clustering_metrics(k, algo, dataset)
                    dataset_results[k] = clustering_metrics
                self.optimal_k.plot_optimal_k_figure(dataset_name, 'SSE-Elbow', dataset_results)
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
                'icmla_2014_accepted_papers_ds': icmla_2014_accepted_papers_ds,
                'parking_birmingham_ds': parking_birmingham_ds}
    pipeline = AgglomerativePipeline()  # Update the instance to use AgglomerativePipeline
    results = pipeline.run_pipeline(datasets)
    print(results)