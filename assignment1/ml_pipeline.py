import os
import numpy as np
from bagging_id3 import MyBaggingID3
from logger_utils import LoggerUtils
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelBinarizer, KBinsDiscretizer, OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import wandb


os.environ['WANDB_API_KEY'] = '2cfad53cf20bc18b3968eafc92a4aedfefbb7af8'


class MLPipeline:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='ass1.log')

    def preprocess_dataset1_breast_cancer_coimbra_data_set(self):
        try:
            # Load the dataset
            df = pd.read_csv('/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment1/'
                             'datasets/Breast Cancer Coimbra Data Set/dataR2.csv')
            # Preprocess the data
            X = df.drop('Classification', axis=1)
            y = df['Classification']
            le = LabelEncoder()
            y = le.fit_transform(y)
            kb = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
            X = kb.fit_transform(X)
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset1_breast_cancer_coimbra_data_set.' % e)

    def preprocess_dataset2_fertility(self):
        try:
            # Load dataset
            df = pd.read_csv("/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment1/"
                             "datasets/Fertility Data Set/fertility_Diagnosis.txt", header=None)
            n_bins = 2
            encode = 'ordinal'
            strategy = 'quantile'
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            # Identify binary features based on the number of unique values
            bin_feats = np.where(np.apply_along_axis(lambda x: len(np.unique(x)) == 2, 0, X))[0]
            nonbin_feats = np.setdiff1d(np.arange(X.shape[1]), bin_feats)

            # Discretize the non-binary features only
            if len(nonbin_feats) > 0:
                kb = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
                X_binned_nonbin = kb.fit_transform(X.loc[:, nonbin_feats])
                X_binned = np.concatenate((X_binned_nonbin, X.loc[:, bin_feats]), axis=1)
            else:
                X_binned = X
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return X_binned, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset2_fertility.' % e)

    def preprocess_dataset3_heart_failure_clinical_records(self):
        try:
            # Load dataset
            df = pd.read_csv('/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/'
                             'assignment1/datasets/Heart failure clinical records Data Set/'
                             'heart_failure_clinical_records_dataset.csv')
            n_bins = 2
            encode = 'ordinal'
            strategy = 'quantile'
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            # Identify binary features based on the number of unique values
            bin_feats = np.where(np.apply_along_axis(lambda x: len(np.unique(x)) == 2, 0, X))[0]
            bin_feats = [X.columns[i] for i in bin_feats]
            nonbin_feats = [col for col in X.columns if col not in bin_feats]

            # Discretize the non-binary features only
            if len(nonbin_feats) > 0:
                kb = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
                X_binned_nonbin = kb.fit_transform(X[nonbin_feats])
                X_binned = np.concatenate((X_binned_nonbin, X[bin_feats]), axis=1)
            else:
                X_binned = X
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return X_binned, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset3_heart_failure_clinical_records.' % e)

    def preprocess_dataset4_ionosphere(self):
        try:
            df = pd.read_csv('/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment1/'
                             'datasets/Ionosphere Data Set/ionosphere.data', header=None)
            n_bins = 2
            encode = 'ordinal'
            strategy = 'quantile'
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            # Identify binary features based on the number of unique values
            bin_feats = np.where(np.apply_along_axis(lambda x: len(np.unique(x)) == 2, 0, X))[0]
            nonbin_feats = np.setdiff1d(np.arange(X.shape[1]), bin_feats)

            # Discretize the non-binary features only
            if len(nonbin_feats) > 0:
                kb = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
                X_binned_nonbin = kb.fit_transform(X.loc[:, nonbin_feats])
                X_binned = np.concatenate((X_binned_nonbin, X.loc[:, bin_feats]), axis=1)
            else:
                X_binned = X
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return X_binned, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset4_ionosphere.' % e)

    def preprocess_dataset5_spectf(self):
        try:
            df = pd.read_csv('/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment1/'
                             'datasets/SPECTF Heart Data Set/data.csv', header=None)
            n_bins = 2
            encode = 'ordinal'
            strategy = 'quantile'
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
            kb = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            X = kb.fit_transform(X)
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset5_spectf.' % e)

    def evaluate_model(self, X, y):
        try:
            # Define the models
            my_bagging_id3 = MyBaggingID3(n_estimators=10, max_samples=0.8, max_features=0.8, max_depth=20)
            dtc = DecisionTreeClassifier()
            bc = BaggingClassifier(base_estimator=dtc, n_estimators=10)
            # Define the evaluation metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1_score': 'f1',
                'roc_auc_score': 'roc_auc'
            }
            # Define the cross-validation procedure
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            # Evaluate the models
            models = {'MyBaggingID3': my_bagging_id3, 'DecisionTreeClassifier': dtc, 'BaggingClassifier': bc}
            wandb.init(project="my-project", name='Assignment1')
            for name, model in models.items():
                cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
                wandb.log({metric: cv_results['test_%s' % metric].mean() for metric in scoring})
        except Exception as e:
            self.logger.error('Exception %s occurred during evaluate_model.' % e)


if __name__ == '__main__':
    ml_pipeline = MLPipeline()
    X, y = ml_pipeline.preprocess_dataset1_breast_cancer_coimbra_data_set()
    ml_pipeline.evaluate_model(X, y)
    X, y = ml_pipeline.preprocess_dataset2_fertility()
    ml_pipeline.evaluate_model(X, y)
    X, y = ml_pipeline.preprocess_dataset3_heart_failure_clinical_records()
    ml_pipeline.evaluate_model(X, y)
    X, y = ml_pipeline.preprocess_dataset4_ionosphere()
    ml_pipeline.evaluate_model(X, y)
    X, y = ml_pipeline.preprocess_dataset5_spectf()
    ml_pipeline.evaluate_model(X, y)
    pass

