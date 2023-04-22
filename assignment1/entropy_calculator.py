import numpy as np
from logger_utils import LoggerUtils


class EntropyCalculator:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass1.log')

    def compute_entropy(self, y):
        try:
            """
            Computes the entropy for
    
            Args:
               y (ndarray): Numpy array indicating whether each example at a node is
                   positive (`1`) or negative (`0`)
    
            Returns: entropy (float): Entropy at that node
            """
            # Check if y is not empty
            if len(y) == 0:
                return 0
            # Count the number of positive and negative examples in y
            num_positives = np.sum(y)
            num_negatives = len(y) - num_positives
            # Calculate the probability of positive and negative examples
            p_positive = num_positives / len(y)
            p_negative = num_negatives / len(y)
            # Calculate entropy
            entropy = 0
            if p_positive > 0:
                entropy -= p_positive * np.log2(p_positive)
            if p_negative > 0:
                entropy -= p_negative * np.log2(p_negative)
            return entropy
        except Exception as e:
            self.logger.error('Exception %s occurred during compute_entropy for %s.' % (e, str(y)))

    def entropy_check(self):
        try:
            # create a numpy array with 10 elements, 6 positive and 4 negative
            y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
            # call compute_entropy function
            entropy = self.compute_entropy(y)
            # print the entropy value
            self.logger.info("Entropy: %s", entropy)
        except Exception as e:
            self.logger.error('Exception %s occurred during entropy_check.' % e)

    def compute_information_gain(self, X, y, node_indices, feature):
        try:
            """
            Compute the information of splitting the node on a given feature
    
            Args:
                X (ndarray):            Data matrix of shape(n_samples, n_features)
                y (array like):         list or ndarray with n_samples containing the target variable
                node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    
            Returns:
                cost (float):        Cost computed
            """

            # Split the data into left and right branches based on the chosen feature
            left_indices, right_indices = self.split_dataset(X, node_indices, feature)
            # Calculate the entropy at the node
            node_entropy = self.compute_entropy(y[node_indices])

            # Calculate the entropy at the left and right branches
            left_entropy = self.compute_entropy(y[left_indices])
            right_entropy = self.compute_entropy(y[right_indices])

            # Calculate the proportion of examples at the left and right branches
            w_left = len(left_indices) / len(node_indices)
            w_right = len(right_indices) / len(node_indices)

            # Calculate the information gain using the formula
            information_gain = node_entropy - (w_left * left_entropy + w_right * right_entropy)
            return information_gain
        except Exception as e:
            self.logger.error('Exception %s occurred during compute_information_gain.' % e)

    def ig_check(self):
        try:
            X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
            y = np.array([1, 0, 1, 0, 1])
            node_indices = [0, 1, 2, 3, 4]
            feature = 1
            information_gain = self.compute_information_gain(X, y, node_indices, feature)
            print(f"Information gain for feature {feature}: {information_gain}")
        except Exception as e:
            self.logger.error('Exception %s occurred during ig_check.' % e)

    def split_dataset(self, X, node_indices, feature):
        try:
            """
            Splits the data at the given node into
            left and right branches

            Args:
                X (ndarray):             Data matrix of shape(n_samples, n_features)
                node_indices (list):  List containing the active indices. I.e, the samples being considered at this step.
                feature (int):           Index of feature to split on

            Returns:
                left_indices (list): Indices with feature value == 1
                right_indices (list): Indices with feature value == 0
            """
            # Initialize left and right indices
            left_indices = []
            right_indices = []
            # Iterate over node_indices and split based on feature value
            for i in node_indices:
                if np.abs(X[i][feature] - 0.0) < np.abs(X[i][feature] - 1.0):
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            return left_indices, right_indices
        except Exception as e:
            self.logger.error('Exception %s occurred during split_dataset.' % e)

    def split_data_check(self):
        try:
            # create a numpy array with 10 elements, 6 positive and 4 negative
            X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1]])
            node_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            feature = 0
            # call split_dataset function
            left_indices, right_indices = self.split_dataset(X, node_indices, feature)
            # print the left and right indices
            print("Left indices:", left_indices)
            print("Right indices:", right_indices)
        except Exception as e:
            self.logger.error('Exception %s occurred during split_data_check.' % e)


if __name__ == '__main__':
    entropy_calc = EntropyCalculator()
    entropy_calc.entropy_check()
    entropy_calc.ig_check()
    entropy_calc.split_data_check()
