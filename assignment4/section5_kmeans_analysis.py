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
                    elbow_k, elbow_sse = self.find_elbow_k(k_values, metric_values)
                    plt.annotate(f'Elbow Point: K = {elbow_k}',
                                 xy=(elbow_k, elbow_sse),
                                 xytext=(elbow_k, 0.8 * max(metric_values)),
                                 arrowprops=dict(facecolor='red', arrowstyle='->'))
                else:
                    best_k, best_metric = self.find_best_k(k_values, metric_values, metric_name)
                    plt.annotate(f'Best K: K = {best_k}',
                                 xy=(best_k, best_metric),
                                 xytext=(best_k, 0.7 * max(metric_values)),
                                 arrowprops=dict(facecolor='green', arrowstyle='->'))
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel(metric_name)
                plt.legend()
                # Log the plot to Weights & Biases
                wandb.log({f'{dataset_name}_{metric_name}': plt})
        except Exception as e:
            self.logger.error('Exception %s occurred during plot_elbow_and_metrics.' % e)

    def find_elbow_k(self, k_values, sse_values):
        try:
            # Check if there is a sequence of trailing zeros in sse_values
            end_index = len(sse_values) - 1
            while end_index >= 0 and list(sse_values)[end_index] == 0:
                end_index -= 1
            if list(sse_values)[0] == 0:
                start_index = 1
            else:
                start_index = 0
            # Calculate the second derivative of SSE
            deltas = np.diff(np.diff(list(sse_values)[start_index:end_index + 1]))
            # Find the index where the second derivative changes significantly
            elbow_index = np.argmin(deltas) + 1  # Adding 1 to account for the double differentiation
            # Return the corresponding K value and its SSE value as the elbow point
            elbow_k = k_values[elbow_index]
            elbow_sse = list(sse_values)[elbow_index]
            return elbow_k, elbow_sse
        except Exception as e:
            self.logger.error('Exception %s occurred during find_elbow_k.' % e)

    def find_best_k(self, k_values, metric_values, metric_name):
        try:
            # Check if there is a sequence of trailing zeros in metric_values
            end_index = len(metric_values) - 1
            while end_index >= 0 and list(metric_values)[end_index] == 0:
                end_index -= 1
            if list(metric_values)[0] == 0:
                start_index = 1
            else:
                start_index = 0

            # Different logic for different metrics
            if metric_name == 'VRC':
                # For VRC, look for the K with the highest metric value
                max_metric_idx = np.argmax(list(metric_values)[start_index:end_index + 1])
                return k_values[start_index + max_metric_idx], list(metric_values)[start_index + max_metric_idx]
            elif metric_name in ('DB', 'My_clustring_metric'):
                # For DB, look for the K with the lowest metric value
                min_metric_idx = np.argmin(list(metric_values)[start_index:end_index + 1])
                return k_values[start_index + min_metric_idx], list(metric_values)[start_index + min_metric_idx]
            elif metric_name == 'Silhouette':
                # For Silhouette, look for the K with the highest metric value
                max_metric_idx = np.argmax(list(metric_values)[start_index:end_index + 1])
                return k_values[start_index + max_metric_idx], list(metric_values)[start_index + max_metric_idx]
            else:
                # Handle other metrics if needed
                return None, None
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
