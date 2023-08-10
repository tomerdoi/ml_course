from typing import Dict

import pandas as pd
from sklearn.cluster import KMeans
from pipeline import Pipeline
from dataset_handler import DatasetHandler


class KMeansPipeline(Pipeline):
    def __init__(self):
        super().__init__()  # Call the parent class constructor first
        self.logger = self.logger_util.init_logger(log_file_name='kmeans_pipeline.log')

    def run_pipeline(self, datasets: Dict[str, pd.DataFrame]) -> dict:
        try:
            # todo: to check ranges, according to the assignments instructions it should include 1 which is impossible
            k_values = list(range(2, 31)) + list(range(35, 100, 5)) + list(range(100, 1001, 25))
            results = {}
            for dataset_name, dataset in datasets.items():
                print(f"Running K-Means pipeline for dataset: {dataset_name}")
                samples = len(dataset)
                dataset_results = {}
                for k in k_values:
                    print(f"K = {k}")
                    if k > samples:
                        self.logger.info(f"Stopped running since the number of clusters ({k}) is bigger than the "
                                         f"number of samples ({samples})")
                        break
                    algo = KMeans(n_clusters=k, random_state=42)
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
    pipeline = KMeansPipeline()
    results = pipeline.run_pipeline(datasets)
    print(results)
