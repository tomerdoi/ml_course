import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
from my_id3 import MyID3
from sklearn.exceptions import *
from sklearn.utils.estimator_checks import check_estimator

warnings.filterwarnings("ignore")


class MyBaggingID3(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, max_samples=0.5, max_features=0.5, max_depth=3):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
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
            n_samples, n_features = X.shape
            self.features_ = []
            self.estimators_ = []
            for i in range(self.n_estimators):
                indices = np.random.choice(n_samples, size=int(n_samples * self.max_samples), replace=True)
                estimator = MyID3(max_depth=self.max_depth)
                if self.max_features < 1.0:
                    features = np.random.choice(n_features, size=int(n_features * self.max_features), replace=False)
                    self.features_.append(features)
                    X_ = X[:, features]
                else:
                    X_ = X
                estimator.fit(X_[indices], y[indices])
                self.estimators_.append(estimator)
            return self
        except Exception as e:
            print('Exception %s occurred during fit.' % e)
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
            n_samples, n_features = X.shape
            n_estimators = len(self.estimators_)
            y_pred = np.zeros((n_samples, n_estimators))
            for i, estimator in enumerate(self.estimators_):
                if self.max_features < 1.0:
                    features = self.features_[i]
                    X_ = X[:, features]
                else:
                    X_ = X
                for j in range(n_samples):
                    y_pred[j, i] = self._predict_instance(X_[j], estimator)
            return self.convert_labels_predict(np.mean(y_pred, axis=1))
        except Exception as e:
            print('Exception %s occurred during predict.' % e)
            raise e

    def _predict_instance(self, x, estimator):
        try:
            if not hasattr(self, 'classes_'):
                raise NotFittedError("Fit was not called.")
            return estimator._predict_instance(x)
        except Exception as e:
            print('Exception %s occurred during _predict_instance.' % e)
            raise e


def model_check():
    # check_estimator(LinearSVC())  # passes
    check_estimator(MyBaggingID3())  # passes
    test_gen = check_estimator(MyBaggingID3(), True)  # passes
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
    clf = MyBaggingID3(max_depth=3)

    # Train the classifier on the sample data
    clf.fit(X, y)

    # Predict the labels of the training data
    y_pred = clf.predict(X)
    print("Predicted labels:", y_pred)

    # Predict the probabilities of the training data
    y_prob = clf.predict_proba(X)
    print("Predicted probabilities:", y_prob)
    print('Finished program')
