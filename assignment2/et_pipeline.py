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

    def run(self):
        try:
            for strategy in ['most_frequent']:  # ['mean', 'median', 'most_frequent', 'constant']:
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
                    clf = ExtraTreesClassifier(max_features='sqrt', max_leaf_nodes=21,
                                               n_estimators=500, n_jobs=-1, random_state=12032022, max_depth=600,
                                               criterion="gini")
                    clf.fit(X_imputed, y)
                    pred_probs = clf.predict_proba(X_test_imputed)
                    # Save the predictions to a CSV file
                    with open('sub_extra_tree_sk_%s.csv' % strategy, 'w') as fp:
                        fp.write("Id,Predicted\n")
                        for i, row in enumerate(pred_probs):
                            fp.write(f"{i},{row[1]:.9f}\n")
                except Exception as e:
                    self.logger.error('Exception %s occurred during run for strategy %s.' % (e, strategy))
        except Exception as e:
            self.logger.error('Exception %s occurred during run.' % e)


if __name__ == '__main__':
    et_pipeline = ETPipeline()
    et_pipeline.run()
    pass
