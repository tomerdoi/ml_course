import numpy as np
from logger_utils import LoggerUtils
from entropy_calculator import EntropyCalculator


class DataHandler:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass1.log')
        self.entropy_calculator = EntropyCalculator()

    def get_best_split(self, X, y, node_indices):
        try:
            """
            Returns the optimal feature and threshold value
            to split the node data
    
            Args:
                X (ndarray):            Data matrix of shape(n_samples, n_features)
                y (array like):         list or ndarray with n_samples containing the target variable
                node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    
            Returns:
                best_feature (int):     The index of the best feature to split
            """
            best_feature = None
            best_info_gain = -1
            # Compute entropy at node
            node_entropy = self.entropy_calculator.compute_entropy(y[node_indices])
            # Iterate over features
            for feature in range(X.shape[1]):
                # Split dataset
                left_indices, right_indices = self.entropy_calculator.split_dataset(X, node_indices, feature)
                # If split is not meaningful, skip
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                # Compute information gain
                info_gain = self.entropy_calculator.compute_information_gain(X, y, node_indices, feature)
                # Update best feature if necessary
                if info_gain > best_info_gain:
                    best_feature = feature
                    best_info_gain = info_gain
            return best_feature
        except Exception as e:
            self.logger.error('Exception %s occurred during get_best_split.' % e)

    def get_best_split_check(self):
        try:
            # example training data
            X = np.array([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]])
            y = np.array([1, 0, 0, 1, 1, 0])
            # example node indices
            node_indices = [0, 1, 2, 3, 4, 5]
            # call the function
            best_feature = self.get_best_split(X, y, node_indices)
            self.logger.info(best_feature)  # prints the index of the best feature to split on
        except Exception as e:
            self.logger.error('Exception %s occurred during get_best_split_check.' % e)


if __name__ == '__main__':
    data_handle = DataHandler()
    data_handle.get_best_split_check()
