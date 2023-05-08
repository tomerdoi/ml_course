from logger_utils import LoggerUtils
import autosklearn.classification
import pandas as pd
import autosklearn.classification
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


class AutoSklearnPipeline:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass2.log')

    def run(self):
        try:
            train_set = pd.read_csv('train.csv')
            X = train_set.iloc[:, :-1]
            y = train_set.iloc[:, -1]
            # Impute the missing and NaN values using the median imputation
            imp = SimpleImputer(strategy='median')
            X_imputed = imp.fit_transform(X)
            # Define the AutoML classifier with default settings
            clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30,
                                                                   seed=42)
            # Fit the AutoML classifier to the data
            clf.fit(X_imputed, y)
            # Print the selected model and its score on the holdout set
            print('Selected model %s:' % str(clf.show_models()))
            print('Score: %s' % str(clf.score(X_imputed, y)))
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    auto_sklearn_pipeline = AutoSklearnPipeline()
    auto_sklearn_pipeline.run()
    pass
