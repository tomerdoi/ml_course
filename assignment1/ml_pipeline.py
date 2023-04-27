import os
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
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset1_breast_cancer_coimbra_data_set.' % e)

    def preprocess_dataset2_fertility(self):
        try:
            # Load dataset
            df = pd.read_csv("/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment1/"
                             "datasets/Fertility Data Set/fertility_Diagnosis.txt", header=None)
            # Convert non-binary features using KBinsDiscretizer with 2 bins
            kb = KBinsDiscretizer(n_bins=2, encode='onehot-dense', strategy='uniform')
            X = kb.fit_transform(df.iloc[:, :-1])
            # Convert the class using LabelBinarizer
            lb = LabelBinarizer()
            y = lb.fit_transform(df.iloc[:, -1])
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset2_fertility.' % e)

    def preprocess_dataset3_heart_failure_clinical_records(self):
        try:
            # Load dataset
            df = pd.read_csv('/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/'
                             'assignment1/datasets/Heart failure clinical records Data Set/'
                             'heart_failure_clinical_records_dataset.csv')
            # Separate the target variable from the input features
            X = df.drop('DEATH_EVENT', axis=1)
            y = df['DEATH_EVENT']
            # Convert non-binary features to binary using LabelBinarizer and OneHotEncoder
            binarizer = LabelBinarizer()
            onehot = OneHotEncoder()
            X['sex'] = binarizer.fit_transform(X['sex'])
            X['smoking'] = binarizer.fit_transform(X['smoking'])
            X['anaemia'] = binarizer.fit_transform(X['anaemia'])
            X['diabetes'] = binarizer.fit_transform(X['diabetes'])
            X['high_blood_pressure'] = binarizer.fit_transform(X['high_blood_pressure'])
            # One-hot encode the age, ejection_fraction, and serum_creatinine columns
            X_encoded = onehot.fit_transform(X[['age', 'ejection_fraction', 'serum_creatinine']]).toarray()
            X.drop(['age', 'ejection_fraction', 'serum_creatinine'], axis=1, inplace=True)
            X = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_encoded)], axis=1)
            # Convert the target variable to binary
            y = binarizer.fit_transform(y)
            return X, y
        except Exception as e:
            self.logger.error('Exception %s occurred during preprocess_dataset3_heart_failure_clinical_records.' % e)

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
    pass

