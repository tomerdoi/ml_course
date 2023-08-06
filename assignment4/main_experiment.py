from KMeansPipeline import KMeansPipeline
from DBSCANPipeline import DBSCANPipeline
from agglomerative_pipeline import AgglomerativePipeline
from optics_pipeline import OPTICSPipeline
import global_conf
from logger_utils import LoggerUtils
from dataset_handler import DatasetHandler
import warnings
warnings.filterwarnings('ignore')


class MainExperiment:  # 2. Rename the class to AgglomerativePipeline
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='main_experiment.log')
        self.report_path = global_conf.report_path

    def load_datasets(self):
        try:
            ds_handler = DatasetHandler()
            south_german_credit_ds = ds_handler.load_south_german_credit()
            icmla_2014_accepted_papers_ds = ds_handler.load_icmla_2014_accepted_papers_data_set()
            parking_birmingham_ds = ds_handler.load_parking_birmingham_data_set()
            datasets = {'south_german_credit_ds': south_german_credit_ds,
                        'icmla_2014_accepted_papers_ds': icmla_2014_accepted_papers_ds,
                        'parking_birmingham_ds': parking_birmingham_ds}
            return datasets
        except Exception as e:
            self.logger.error('Exception %s occurred during load_datasets.' % e)

    def write_header(self):
        try:
            with open(self.report_path, 'w') as fp:
                fp.write('Algorithm,Dataset,Hyper-parameter name,Hyper-parameter value,Metric name,Metric value,'
                         'Num clusters\n')
        except Exception as e:
            self.logger.error('Exception %s occurred during write_header.' % e)

    def write_results_report(self, algo_name, ds_name, hp_name, hp_value, met_name, met_value, num_clusters):
        try:
            with open(self.report_path, 'a') as fp:
                fp.write('%s,' % algo_name)
                fp.write('%s,' % ds_name)
                fp.write('%s,' % hp_name)
                fp.write('%s,' % hp_value)
                fp.write('%s,' % met_name)
                fp.write('%s,' % met_value)
                fp.write('%s\n' % num_clusters)
        except Exception as e:
            self.logger.error('Exception %s occurred during write_results_report.' % e)

    def algo_experiment(self, algo_name, datasets, hyper_param_name, algo_object):
        try:
            self.logger.info('Running %s' % algo_name)
            pipeline = algo_object
            results = pipeline.run_pipeline(datasets)
            for ds_name in results:
                for hp_value in results[ds_name]:
                    unique_labels = results[ds_name][hp_value]['unique_labels']
                    for metric_name in results[ds_name][hp_value]:
                        if metric_name == 'unique_labels':
                            continue
                        metric_value = results[ds_name][hp_value][metric_name]
                        self.write_results_report(algo_name, ds_name, hyper_param_name, hp_value, metric_name,
                                                  metric_value, unique_labels)
        except Exception as e:
            self.logger.error('Exception %s occurred during algo_experiment.' % e)

    def main_experiment(self):
        try:
            self.write_header()
            self.logger.info('Loading datasets.')
            datasets = self.load_datasets()
            self.algo_experiment('K-Means', datasets, 'K', KMeansPipeline())
            self.algo_experiment('DBSCAN', datasets, 'min_samples', DBSCANPipeline())
            self.algo_experiment('Agglomerative', datasets, 'K', AgglomerativePipeline())
            self.algo_experiment('OPTICS', datasets, 'min_samples', OPTICSPipeline())
        except Exception as e:
            self.logger.error('Exception %s occurred during main_experiment.' % e)


if __name__ == '__main__':
    main_ex = MainExperiment()
    main_ex.main_experiment()
