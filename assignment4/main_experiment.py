import pandas as pd

from KMeansPipeline import KMeansPipeline
from DBSCANPipeline import DBSCANPipeline
from agglomerative_pipeline import AgglomerativePipeline
from pipeline import Pipeline
from optics_pipeline import OPTICSPipeline
import global_conf
from logger_utils import LoggerUtils
from dataset_handler import DatasetHandler
import warnings

warnings.filterwarnings('ignore')


class MainExperiment:  # 2. Rename the class to AgglomerativePipeline
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')
        self.report_path = global_conf.report_path
        self.results = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
                                             'Metric name', 'Metric value', 'Num clusters'])

    def load_datasets(self):
        try:
            ds_handler = DatasetHandler()
            south_german_credit_ds = ds_handler.load_south_german_credit()
            icmla_2014_accepted_papers_ds = ds_handler.load_icmla_2014_accepted_papers_data_set()
            parking_birmingham_ds = ds_handler.load_parking_birmingham_data_set()
            datasets = {
                'south_german_credit_ds': south_german_credit_ds,
                'icmla_2014_accepted_papers_ds': icmla_2014_accepted_papers_ds,
                'parking_birmingham_ds': parking_birmingham_ds
            }
            return datasets
        except Exception as e:
            self.logger.error('Exception %s occurred during load_datasets.' % e)

    def algo_experiment(self, algo_name: str, datasets: dict, hyper_param_name: str, pipeline: Pipeline):
        try:
            self.logger.info('Running %s' % algo_name)
            results = pipeline.run_pipeline(datasets)
            final_df = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
                                             'Metric name', 'Metric value', 'Num clusters'])
            for ds_name in results:
                for hp_value in results[ds_name]:
                    unique_labels = results[ds_name][hp_value].pop('unique_labels')
                    if len(unique_labels) == 1 and list(unique_labels)[0] == -1:
                        num_of_clusters = 0
                    else:
                        num_of_clusters = results[ds_name][hp_value].pop('num_of_clusters')
                    for metric_name in results[ds_name][hp_value]:
                        metric_value = results[ds_name][hp_value][metric_name]
                        final_df.loc[len(final_df)] = [algo_name, ds_name, hyper_param_name, hp_value, metric_name,
                                                       metric_value, num_of_clusters]

            self.results = pd.concat([self.results, final_df], ignore_index=True)
            self.results.to_csv(self.report_path, index=False)
        except Exception as e:
            self.logger.error('Exception %s occurred during algo_experiment.' % e)

    def main_experiment(self):
        try:
            self.logger.info('Loading datasets.')
            datasets = self.load_datasets()
            # self.algo_experiment('K-Means', datasets, 'K', KMeansPipeline())
            datasets = {'parking_birmingham_ds': datasets['parking_birmingham_ds']}
            self.algo_experiment('DBSCAN', datasets, 'min_samples', DBSCANPipeline())
            self.algo_experiment('Agglomerative', datasets, 'K', AgglomerativePipeline())
            self.algo_experiment('OPTICS', datasets, 'min_samples', OPTICSPipeline())
        except Exception as e:
            self.logger.error('Exception %s occurred during main_experiment.' % e)


if __name__ == '__main__':
    main_ex = MainExperiment()
    main_ex.main_experiment()
