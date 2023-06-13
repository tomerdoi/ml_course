from logger_utils import LoggerUtils
import autosklearn.classification
import autosklearn.classification
from preprocessor import PreProcessor
from autosklearn import metrics
import warnings
warnings.filterwarnings("ignore")


class AutoSklearnPipeline:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass2.log')
        self.preprocessor = PreProcessor()

    def run(self):
        try:
            for strategy in ['most_frequent']:  # , 'median', 'most_frequent', 'constant']:
                try:
                    X, y, X_test = self.preprocessor.preprocess_data(strategy='most_frequent', smote=True,
                                                                     iterations=200)
                    clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600 * 6,
                                                                           per_run_time_limit=2000, seed=42,
                                                                           memory_limit=10 * 1024,
                                                                           include={
                                                                               'classifier': ["random_forest",
                                                                                              "extra_trees"]
                                                                           },
                                                                           metric=autosklearn.metrics.roc_auc
                                                                           )
                    # Fit the AutoML classifier to the data
                    clf.fit(X, y)
                    # Print the selected model and its score on the holdout set
                    print('Selected model %s:' % str(clf.show_models()))
                    print('Score: %s' % str(clf.score(X, y)))
                    probs = clf.predict_proba(X_test)
                    with open('sub_sklearn_%s.csv' % strategy, 'w') as fp:
                        fp.write('Id,Predicted\n')
                        for i in range(len(probs)):
                            fp.write('%d,%0.9f\n' % (i, probs[i][1]))
                except Exception as e:
                    self.logger.error('Exception %s occurred during run for strategy %s.' % (e, strategy))
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    auto_sklearn_pipeline = AutoSklearnPipeline()
    auto_sklearn_pipeline.run()
    pass
