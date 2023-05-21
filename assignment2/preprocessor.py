import random
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from logger_utils import LoggerUtils
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


class PreProcessor:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass2.log')

    def smote_train_set(self, X, y):
        try:
            # {'not majority', 'auto', 'not minority', 'minority', 'all'}
            smote = SMOTE(random_state=42)
            # Generate synthetic samples
            for i in range(100):
                # Randomly select a subset of rows from X and y
                random_indices = np.random.choice(len(X), size=random.randint(int(0.5 * len(X)), len(X) - 1),
                                                  replace=False)
                X_subset, y_subset = X[random_indices], y[random_indices]
                # Apply SMOTE on the subset of rows
                X_synthetic, y_synthetic = smote.fit_resample(X_subset, y_subset)
                # Find the indices of rows not selected in the random subset
                non_selected_indices = np.setdiff1d(np.arange(len(X)), random_indices)
                # Concatenate the synthetic samples with the non-selected rows
                X = np.concatenate((X[non_selected_indices], X_synthetic), axis=0)
                y = np.concatenate((y[non_selected_indices], y_synthetic), axis=0)
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during smote_train_set.' % e)

    def preprocess_data(self, strategy='most_frequent', smote=False):
        try:
            train_set = pd.read_csv('train.csv')
            # train_set = pd.concat([train_set] * 20, ignore_index=True)
            test_set = pd.read_csv('validation_and_test.csv')
            X_test = test_set.iloc[:, 1:]
            X = train_set.iloc[:, :-1]
            y = train_set.iloc[:, -1]
            if strategy:
                imp = SimpleImputer(strategy=strategy)
                X_imputed = imp.fit_transform(X)
                X_test_imputed = imp.fit_transform(X_test)
            else:
                X_imputed = X
                X_test_imputed = X_test
            if smote:
                X_imputed, y = self.smote_train_set(X_imputed, y)
            return X_imputed, y, X_test_imputed
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_data.' % e)


if __name__ == '__main__':
    preprocessor = PreProcessor()
    X_imputed, y, X_test_imputed = preprocessor.preprocess_data(smote=True)
    pass
