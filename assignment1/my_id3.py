import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import warnings
from sklearn.preprocessing import LabelEncoder
from entropy_calculator import EntropyCalculator
from data_handler import DataHandler
warnings.filterwarnings("ignore")


class MyID3(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def convert_labels(self, y):
        y = check_array(y, ensure_2d=False, dtype=None)
        try:
            y = y.astype('int')
        except:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        return y

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        y = self.convert_labels(y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.tree_ = self.build_tree(X, y, depth=0)  # build the decision tree
        # Return the classifier
        return self

    def build_tree(self, X, y, depth):
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
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        # Find the best split
        node_indices = np.arange(X.shape[0])
        best_feature = data_handler.get_best_split(X, y, node_indices)

        # If no split found, return a leaf node with the majority class
        if best_feature is None:
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        # Split the dataset
        left_indices, right_indices = entropy_calc.split_dataset(X, node_indices, best_feature)
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return {"leaf": True, "class": np.argmax(np.bincount(y))}
        # Recursively build left and right subtrees
        left_tree = self.build_tree(left_X, left_y, depth + 1)
        right_tree = self.build_tree(right_X, right_y, depth + 1)

        # Create a dictionary to represent the current node
        node = {"leaf": False, "feature": best_feature, "left": left_tree, "right": right_tree}
        return node

    def predict_proba(self, X):
        """
        Predict class probabilities for input data using a trained model.

        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            Prob (ndarray): Probability matrix of shape (n_samples, n_classes).
        """
        # Return probability matrix
        Prob = np.zeros((X.shape[0], len(self.classes_)))
        for i, sample in enumerate(X):
            node = self.tree_
            while not node["leaf"]:
                if sample[node["feature"]] == 1:
                    node = node["left"]
                else:
                    node = node["right"]
            Prob[i, node["class"]] = 1
        return Prob

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)
        # Convert input to numpy array
        X = np.array(X)

        # Handle the case where there is only one class in the training data
        if len(self.classes_) == 1:
            return np.array([self.classes_[0]] * len(X))

        # Make predictions using the decision tree
        y_pred = np.array([self._predict_instance(x, self.tree_) for x in X])

        # Convert predicted labels to the original class labels
        label_encoder = LabelEncoder()
        y_pred = label_encoder.fit_transform(y_pred)

        return y_pred

    def _predict_instance(self, x, node):
        if node["leaf"]:
            return node["class"]
        else:
            feature = node["feature"]
            if x[feature] == 0:
                return self._predict_instance(x, node["left"])
            else:
                return self._predict_instance(x, node["right"])


if __name__ == '__main__':
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
    clf = MyID3(max_depth=2)

    # Train the classifier on the sample data
    clf.fit(X, y)

    # Predict the labels of the training data
    y_pred = clf.predict(X)
    print("Predicted labels:", y_pred)

    # Predict the probabilities of the training data
    y_prob = clf.predict_proba(X)
    print("Predicted probabilities:", y_prob)
    print('Finished program')
