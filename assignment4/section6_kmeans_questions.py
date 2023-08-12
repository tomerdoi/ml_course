from logger_utils import LoggerUtils
import pandas as pd
import matplotlib.pyplot as plt


class Section6KmeansQuestions:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')

    def plot_optimal_k(self):
        try:
            # Load the optimal_k.csv data
            results = pd.read_csv('optimal_k.csv')
            # Plot SSE values across different k values
            plt.figure(figsize=(10, 6))
            plt.plot(results[results['Metric name'] == 'SSE-Elbow']['Estimated optimal k'],
                     results[results['Metric name'] == 'SSE-Elbow']['Actual K'], marker='o')
            plt.xlabel('Estimated Optimal K')
            plt.ylabel('Actual K')
            plt.title('SSE-Elbow Method: Estimated Optimal K vs. Actual K')
            plt.savefig('./SSE-Elbow Method: Estimated Optimal K.png')
            plt.show()
        except Exception as e:
            self.logger.error('Exception %s occurred during plot_optimal_k.' % e)

    import matplotlib.pyplot as plt

    def find_best_metric(self):
        try:
            # Load the optimal_k.csv data
            results = pd.read_csv('optimal_k.csv')
            # Calculate the mean difference between Estimated optimal k and Actual K for each metric
            results['Mean difference'] = results['Estimated optimal k'] - results['Actual K']
            mean_difference_by_metric = results.groupby('Metric name')['Mean difference'].mean()
            # Plot the mean difference for each metric
            plt.figure(figsize=(10, 6))
            plt.bar(mean_difference_by_metric.index, mean_difference_by_metric)
            plt.xlabel('Metric name')
            plt.ylabel('Mean Difference (Estimated K - Actual K)')
            plt.title('Mean Difference between Estimated K and Actual K by Metric')
            plt.xticks(rotation=45)
            # Save the plot to disk
            plt.savefig('mean_difference_by_metric.png')
            plt.close()
            # Print the metric with the lowest mean difference
            best_metric = mean_difference_by_metric.idxmin()
            worst_metric = mean_difference_by_metric.idxmax()
            print(
                f"The metric with the smallest mean difference between Estimated optimal k and Actual K "
                f"is: {best_metric}")
            print(
                f"The metric with the highest mean difference between Estimated optimal k and Actual K "
                f"is: {worst_metric}")
        except Exception as e:
            self.logger.error('Exception %s occurred during find_best_metric.' % e)


if __name__ == '__main__':
    # Provide the path to your raw_metrics.csv file
    section6_kmeans_questions = Section6KmeansQuestions()
    section6_kmeans_questions.plot_optimal_k()
    section6_kmeans_questions.find_best_metric()
    pass
