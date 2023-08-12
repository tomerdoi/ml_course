import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from logger_utils import LoggerUtils


os.environ['WANDB_API_KEY'] = '2cfad53cf20bc18b3968eafc92a4aedfefbb7af8'


class Section5KmeansAnalysis:
    def __init__(self, raw_metrics_csv_path):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')
        self.raw_metrics_df = pd.read_csv(raw_metrics_csv_path)
        # Initialize Weights & Biases
        wandb.init(project="ML2023")

    def plot_elbow_and_metrics(self, dataset_name, metric_names):
        try:
            dataset_df = self.raw_metrics_df[self.raw_metrics_df['Dataset'] == dataset_name]
            k_values = dataset_df[dataset_df['Algorithm'] == 'K-Means']['Hyper-parameter value'].unique()
            for metric_name in metric_names:
                plt.figure(figsize=(10, 6))
                plt.title(f'{dataset_name} - {metric_name}')
                metric_values = dataset_df[(dataset_df['Metric name'] == metric_name) & (dataset_df['Algorithm'] ==
                                           'K-Means')]['Metric value']
                plt.plot(k_values, metric_values, marker='o', label=metric_name)
                if metric_name == 'SSE-Elbow':
                    elbow_k = self.find_elbow_k(k_values, metric_values)
                    plt.axvline(x=elbow_k, color='r', linestyle='--', label='Elbow Point')
                else:
                    best_k = self.find_best_k(k_values, metric_values)
                    plt.axvline(x=best_k, color='g', linestyle='--', label='Best K')
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel(metric_name)
                plt.legend()
                # Log the plot to Weights & Biases
                wandb.log({f'{dataset_name}_{metric_name}': plt})
        except Exception as e:
            self.logger.error('Exception %s occurred during plot_elbow_and_metrics.' % e)

    def find_elbow_k(self, k_values, sse_values):
        try:
            # You can implement your elbow finding logic here
            # For simplicity, I'll just return the index of the minimum SSE value
            min_sse_idx = np.argmin(sse_values)
            return k_values[min_sse_idx]
        except Exception as e:
            self.logger.error('Exception %s occurred during find_elbow_k.' % e)

    def find_best_k(self, k_values, metric_values):
        try:
            # You can implement your best K finding logic here
            # For simplicity, I'll just return the index of the maximum metric value
            max_metric_idx = np.argmax(metric_values)
            return k_values[max_metric_idx]
        except Exception as e:
            self.logger.error('Exception %s occurred during find_best_k.' % e)

    def run_analysis(self, dataset_names, metric_names):
        try:
            for dataset_name in dataset_names:
                self.plot_elbow_and_metrics(dataset_name, metric_names)
            # Finish the Weights & Biases run
            wandb.finish()
        except Exception as e:
            self.logger.error('Exception %s occurred during run_analysis.' % e)


if __name__ == '__main__':
    # Provide the path to your raw_metrics.csv file
    raw_metrics_csv_path = './main_experiment.csv'

    # List of dataset names and metric names for analysis
    dataset_names = ['south_german_credit_ds', 'icmla_2014_accepted_papers_ds', 'parking_birmingham_ds']
    metric_names = ['SSE-Elbow', 'VRC', 'DB', 'Silhouette', 'My_clustring_metric']

    # Initialize the analysis class and run the analysis
    section5_analysis = Section5KmeansAnalysis(raw_metrics_csv_path)
    section5_analysis.run_analysis(dataset_names, metric_names)
