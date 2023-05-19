import random
from imblearn.over_sampling import SMOTE
import pandas as pd
from logger_utils import LoggerUtils
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings("ignore")


class ETPipeline:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass2.log')

    def smote_train_set(self, train_set):
        try:
            # Separate features (X) and labels (y)
            X = train_set.iloc[:, :-1]
            cols = X.columns
            y = train_set.iloc[:, -1]
            imp = SimpleImputer(strategy='most_frequent')
            X = imp.fit_transform(X)
            # Create an instance of SMOTE
            smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=20)

            # Generate synthetic samples
            X_synthetic, y_synthetic = smote.fit_resample(X, y)

            # Combine the original dataset and synthetic samples
            train_set_inc = pd.concat(
                [X, y, pd.DataFrame(X_synthetic, columns=cols), pd.Series(y_synthetic, name='label')], axis=1)

            # Ensure the final dataset has 6000 instances
            train_set_inc = train_set_inc.sample(n=6000, random_state=42)
            return train_set_inc
        except Exception as e:
            self.logger.error('Exception %s occurred during smote_train_set.' % e)

    def run(self):
        try:
            for strategy in ['most_frequent']:  # ['mean', 'median', 'most_frequent', 'constant']:
                try:
                    train_set = pd.read_csv('train.csv')
                    # train_set = pd.concat([train_set] * 20, ignore_index=True)
                    # train_set = self.smote_train_set(train_set=train_set)
                    test_set = pd.read_csv('validation_and_test.csv')
                    X_test = test_set.iloc[:, 1:]
                    X = train_set.iloc[:, :-1]
                    y = train_set.iloc[:, -1]
                    imp = SimpleImputer(strategy=strategy)
                    X_imputed = imp.fit_transform(X)
                    X_test_imputed = imp.fit_transform(X_test)
                    clf = ExtraTreesClassifier(max_features='sqrt', max_leaf_nodes=21,
                                               n_estimators=1000, n_jobs=-1, random_state=12032022, max_depth=1000,
                                               criterion="gini")
                    clf.fit(X_imputed, y)
                    pred_probs = clf.predict_proba(X_test_imputed)
                    # Save the predictions to a CSV file
                    with open('sub_extra_tree_sk_%s.csv' % strategy, 'w') as fp:
                        fp.write("Id,Predicted\n")
                        for i, row in enumerate(pred_probs):
                            if 0.48 <= row[1] <= 0.52:
                                row[1] = random.choice([0.1, 0.9])
                            fp.write(f"{i},{row[1]:.9f}\n")
                except Exception as e:
                    self.logger.error('Exception %s occurred during run for strategy %s.' % (e, strategy))
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    et_pipeline = ETPipeline()
    et_pipeline.run()
    pass
