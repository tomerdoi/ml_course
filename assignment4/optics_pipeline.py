from sklearn.cluster import OPTICS  # 1. Import OPTICS
from pipeline import Pipeline
from dataset_handler import DatasetHandler


class OPTICSPipeline(Pipeline):  # 2. Rename the class to OPTICSPipeline
    def __init__(self):
        super().__init__()  # Call the parent class constructor first
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')

    def run_pipeline(self, datasets):
        try:
            # todo: to check ranges, according to the assignments instructions it should include 1 which is impossible
            min_samples_values = list(range(1, 21))
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running OPTICS pipeline for dataset: {dataset_name}")  # Update the message
                dataset_results = {}
                for min_samples in min_samples_values:
                    print(f"min_samples = {min_samples}")
                    algo = OPTICS(min_samples=min_samples)  # Use OPTICS and set min_samples=k
                    clustering_metrics = self.measure_clustering_metrics(min_samples, algo, dataset)
                    dataset_results[min_samples] = clustering_metrics
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
    pipeline = OPTICSPipeline()  # Update the instance to use OPTICSPipeline
    results = pipeline.run_pipeline(datasets)
    print(results)
