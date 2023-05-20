import random
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from logger_utils import LoggerUtils
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings("ignore")


class ETPipeline:
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

    def preprocess_data(self, strategy, smote=False):
        try:
            train_set = pd.read_csv('train.csv')
            # train_set = pd.concat([train_set] * 20, ignore_index=True)
            test_set = pd.read_csv('validation_and_test.csv')
            X_test = test_set.iloc[:, 1:]
            X = train_set.iloc[:, :-1]
            y = train_set.iloc[:, -1]
            imp = SimpleImputer(strategy=strategy)
            X_imputed = imp.fit_transform(X)
            X_test_imputed = imp.fit_transform(X_test)
            if smote:
                X_imputed, y = self.smote_train_set(X_imputed, y)
            return X_imputed, y, X_test_imputed
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_data.' % e)

    def run(self):
        try:
            for strategy in ['most_frequent']:  # ['mean', 'median', 'most_frequent', 'constant']:
                try:
                    X_imputed, y, X_test_imputed = self.preprocess_data(strategy=strategy, smote=True)
                    clf = ExtraTreesClassifier(max_features='sqrt', max_leaf_nodes=31,
                                               n_estimators=1000, n_jobs=-1, random_state=12032022, max_depth=1000,
                                               criterion="gini")
                    clf.fit(X_imputed, y)
                    pred_probs = clf.predict_proba(X_test_imputed)
                    # Save the predictions to a CSV file
                    with open('sub_extra_tree_sk_%s.csv' % strategy, 'w') as fp:
                        fp.write("Id,Predicted\n")
                        for i, row in enumerate(pred_probs):
                            # if 0.48 <= row[1] <= 0.52:
                            #     row[1] = random.choice([0.1, 0.9])
                            fp.write(f"{i},{row[1]:.9f}\n")
                except Exception as e:
                    self.logger.error('Exception %s occurred during run for strategy %s.' % (e, strategy))
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    et_pipeline = ETPipeline()
    et_pipeline.run()
    pass
