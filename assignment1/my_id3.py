import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
from sklearn.exceptions import *
from entropy_calculator import EntropyCalculator
from data_handler import DataHandler
from sklearn.utils.estimator_checks import check_estimator
warnings.filterwarnings("ignore")


class MyID3(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def _more_tags(self):
        return {'binary_only': True, 'multioutput': False, 'poor_score': True}

    def convert_label_predict(self, y_value):
        y_converted = self.binary_to_classes_[y_value]
        return y_converted

    def convert_labels_predict(self, y):
        map_func = lambda label: self.binary_to_classes_[label]
        y = np.vectorize(map_func)(y)
        return y

    def convert_labels_fit(self, y):
        y = check_array(y, ensure_2d=False, dtype=None)
        unique_y_values = np.sort(np.unique(y))
        if len(unique_y_values) > 2:
            raise ValueError("Y vector have more than two labels.")
        self.classes_ = unique_y_values
        self.classes_to_binary_ = {self.classes_[0]: 0, self.classes_[1]: 1}
        self.binary_to_classes_ = {0: self.classes_[0], 1: self.classes_[1]}
        # if any(unique_y_values != desired_y_values) or unique_y_values.dtype != desired_y_values.dtype:
        map_func = lambda label: self.classes_to_binary_[label]
        y = np.vectorize(map_func)(y)
        return y

    def fit(self, X, y):
        try:
            X, y = check_X_y(X, y)
            if len(X.shape) != 2:
                raise ValueError("Reshape your data")
            # Check that X and y have correct shape
            if len(X) != len(y):
                raise Exception("X and y length are not compatible.")
            if not len(X):
                raise ValueError("Empty data rows X.")
            if not X.shape[1]:
                raise ValueError("0 feature(s) (shape=(%d, 0)) while a minimum of 2 is required." % X.shape[0])
            if len(np.unique(y)) == 1:
                raise ValueError("Classifier can't train when only one class is present.")
            if len(np.unique(y)) > 2 and y.dtype == float:
                raise ValueError("Unknown label type: ")
            y = self.convert_labels_fit(y)
            # Store the classes seen during fit
            self.n_features_in_ = X.shape[1]
            self.tree_ = self.build_tree(X, y, depth=0, node_indices=np.arange(len(X)))  # build the decision tree
            # Return the classifier
            return self
        except Exception as e:
            print('Exception %s occurred during fit.' % e)
            raise e

    def build_tree(self, X, y, depth, node_indices):
        try:
            """
            Recursive function to build the decision tree.
    
            Args:
                X (ndarray): Data matrix of shape (n_samples, n_features).
                y (ndarray): Data matrix of shape (n_samples, ).
                depth (int): Current depth of the decision tree.
    
            Returns:
                tree (dict): A dictionary representing the decision tree.
            """
            entropy_calc = EntropyCalculator()
            data_handler = DataHandler()
            # Check if max depth is reached or if all labels are the same
            if depth == self.max_depth or np.unique(y).shape[0] == 1:
                return {"leaf": True, "class": np.argmax(np.bincount(y[node_indices])), 'instances': node_indices}

            # Find the best split
            best_feature = data_handler.get_best_split(X, y, node_indices)

            # If no split found, return a leaf node with the majority class
            if best_feature is None:
                return {"leaf": True, "class": np.argmax(np.bincount(y[node_indices])), 'instances': node_indices}

            # Split the dataset
            left_indices, right_indices = entropy_calc.split_dataset(X, node_indices, best_feature)
            left_X, left_y = X, y
            right_X, right_y = X, y

            # Recursively build left and right subtrees
            left_tree = self.build_tree(left_X, left_y, depth + 1, left_indices)
            right_tree = self.build_tree(right_X, right_y, depth + 1, right_indices)

            # Create a dictionary to represent the current node
            node = {"leaf": False, "feature": best_feature, "left": left_tree, "right": right_tree,
                    'instances': node_indices}
            return node
        except Exception as e:
            print('Exception %s occurred during build_tree.' % e)
            raise e

    def predict_proba(self, X):
        try:
            """
            Predict class probabilities for input data using a trained model.
    
            Args:
                X (ndarray): Data matrix of shape (n_samples, n_features).
    
            Returns:
                Prob (ndarray): Probability matrix of shape (n_samples, n_classes).
            """
            if len(X.shape) != 2:
                raise ValueError("Reshape your data")
            if not hasattr(self, 'classes_'):
                raise NotFittedError("Fit was not called.")
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Number of in features in train is different from number in predict.")
            # Return probability matrix
            Prob = np.zeros((X.shape[0], len(self.classes_)))
            for i, sample in enumerate(X):
                node = self.tree_
                while not node["leaf"]:
                    if np.abs(sample[node["feature"]] - 0.0) < np.abs(sample[node["feature"]] - 1.0):
                        node = node["left"]
                    else:
                        node = node["right"]
                Prob[i, node["class"]] = 1
            return Prob
        except Exception as e:
            print('Exception %s occurred during predict_proba.' % e)
            raise e

    def predict(self, X):
        try:
            if not hasattr(self, 'classes_'):
                raise NotFittedError("Fit was not called.")
            if len(np.unique(self.classes_)) == 1:
                raise ValueError("Classifier can't predict when only one class is present.")
            X = check_array(X)
            if len(X.shape) != 2:
                raise ValueError("Reshape your data")
            # Check if fit has been called
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Number of in features in train is different from number in predict.")
            check_is_fitted(self)
            # Convert input to numpy array
            X = np.array(X)

            # Handle the case where there is only one class in the training data
            if len(self.classes_) == 1:
                return np.array([self.classes_[0]] * len(X))

            # Make predictions using the decision tree
            y_pred = np.array([self._predict_instance(x, self.tree_) for x in X])
            return y_pred
        except Exception as e:
            print('Exception %s occurred during predict.' % e)
            raise e

    def _predict_instance(self, x, node):
        try:
            if not hasattr(self, 'classes_'):
                raise NotFittedError("Fit was not called.")
            if node["leaf"]:
                converted_label = self.convert_label_predict(y_value=node["class"])
                return converted_label
            else:
                feature = node["feature"]
                if np.abs(x[feature] - 0.0) < np.abs(x[feature] - 1.0):
                    return self._predict_instance(x, node["left"])
                else:
                    return self._predict_instance(x, node["right"])
        except Exception as e:
            print('Exception %s occurred during _predict_instance.' % e)
            raise e


def model_check():
    # check_estimator(LinearSVC())  # passes
    # check_estimator(MyID3())  # passes
    test_gen = check_estimator(MyID3(), True)  # passes
    tests_to_skip = []
    tests_to_run = []
    count_passed_tests = 0
    count_total_tests = 0
    for i, t in enumerate(test_gen):
        try:
            count_total_tests += 1
            if i in tests_to_skip:  # or i not in tests_to_run:
                continue
            print('Running test %d.' % i)
            t[1](t[0])
            print('test %d passed successfully.' % i)
            count_passed_tests += 1
        except AssertionError as e:
            print('Test %d failed with %s.' % (i, e))
    if count_passed_tests + len(tests_to_skip) == count_total_tests:
        print('check_estimator for MyID3 has completed successfully.')
    else:
        print('Failed checking check_estimator on MyID3.')


if __name__ == '__main__':
    model_check()
    # Create a sample dataset
    X = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])
    y = np.array([0, 1, 0, 1, 0, 1])

    # Initialize the MyID3 classifier
    clf = MyID3(max_depth=3)

    # Train the classifier on the sample data
    clf.fit(X, y)

    # Predict the labels of the training data
    y_pred = clf.predict(X)
    print("Predicted labels:", y_pred)

    # Predict the probabilities of the training data
    y_prob = clf.predict_proba(X)
    print("Predicted probabilities:", y_prob)
    print('Finished program')
