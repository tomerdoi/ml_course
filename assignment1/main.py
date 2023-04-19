import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")


def compute_entropy(y):
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


# create a numpy array with 10 elements, 6 positive and 4 negative
y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1])

# call compute_entropy function
entropy = compute_entropy(y)

# print the entropy value
print("Entropy:", entropy)


def split_dataset(X, node_indices, feature):
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
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices


# create a numpy array with 10 elements, 6 positive and 4 negative
X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1]])
node_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0

# call split_dataset function
left_indices, right_indices = split_dataset(X, node_indices, feature)

# print the left and right indices
print("Left indices:", left_indices)
print("Right indices:", right_indices)


def compute_information_gain(X, y, node_indices, feature):
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
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Calculate the entropy at the node
    node_entropy = compute_entropy(y[node_indices])

    # Calculate the entropy at the left and right branches
    left_entropy = compute_entropy(y[left_indices])
    right_entropy = compute_entropy(y[right_indices])

    # Calculate the proportion of examples at the left and right branches
    w_left = len(left_indices) / len(node_indices)
    w_right = len(right_indices) / len(node_indices)

    # Calculate the information gain using the formula
    information_gain = node_entropy - (w_left * left_entropy + w_right * right_entropy)

    return information_gain

X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
y = np.array([1, 0, 1, 0, 1])
node_indices = [0, 1, 2, 3, 4]
feature = 1

information_gain = compute_information_gain(X, y, node_indices, feature)
print(f"Information gain for feature {feature}: {information_gain}")


def get_best_split(X, y, node_indices):
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
    node_entropy = compute_entropy(y[node_indices])

    # Iterate over features
    for feature in range(X.shape[1]):

        # Split dataset
        left_indices, right_indices = split_dataset(X, node_indices, feature)

        # If split is not meaningful, skip
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue

        # Compute information gain
        info_gain = compute_information_gain(X, y, node_indices, feature)

        # Update best feature if necessary
        if info_gain > best_info_gain:
            best_feature = feature
            best_info_gain = info_gain

    return best_feature


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
best_feature = get_best_split(X, y, node_indices)

print(best_feature)  # prints the index of the best feature to split on


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


check_estimator(LinearSVC())  # passes
check_estimator(TemplateClassifier())
print('Finished program')