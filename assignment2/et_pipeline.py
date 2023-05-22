import os.path

import numpy as np
from logger_utils import LoggerUtils
from sklearn.ensemble import ExtraTreesClassifier
from preprocessor import PreProcessor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


class ETPipeline:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass2.log')
        self.preprocessor = PreProcessor()

    def run(self):
        try:
            for strategy in ['most_frequent']:  # ['mean', 'median', 'most_frequent', 'constant']:
                try:
                    self.logger.info('Preprocessing data.')
                    if os.path.exists('X_imputed.npy') and os.path.exists('y.npy'):
                        X_imputed = np.load('X_imputed.npy')
                        y = np.load('y.npy')
                        _, _, X_test_imputed = self.preprocessor.preprocess_data(strategy=strategy)
                    else:
                        X_imputed, y, X_test_imputed = self.preprocessor.preprocess_data(strategy=strategy, smote=True,
                                                                                         iterations=400)
                        np.save('X_imputed.npy', X_imputed)
                        np.save('y.npy', y)
                    self.logger.info('X len is: %s' % str(X_imputed.shape))
                    clf = ExtraTreesClassifier(max_features='sqrt', max_leaf_nodes=30,
                                               n_estimators=2000, n_jobs=-1, random_state=12032022, max_depth=2000,
                                               criterion="gini")
                    self.logger.info('Fitting model.')
                    clf.fit(X_imputed, y)
                    # # Perform cross-validation
                    # self.logger.info('Validating model.')
                    # cv_scores = cross_val_score(clf, X_imputed, y, cv=5)  # Change the cv value as desired
                    # # Print the cross-validation scores
                    # self.logger.info("Cross-validation scores: %s" % str(cv_scores))
                    # print("Mean cross-validation score: %s" % str(cv_scores.mean()))
                    # print("Standard deviation of cross-validation scores: %s" % str(cv_scores.std()))

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
