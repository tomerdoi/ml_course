import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import global_conf
import numpy as np
from sklearn.preprocessing import StandardScaler
from logger_utils import LoggerUtils


class DatasetHandler:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='yolov5.log')

    def load_south_german_credit(self):
        try:
            asc_file_path = '/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment4/' \
                            'datasets/south+german+credit+update/south+german+credit+update/SouthGermanCredit.asc'
            with open(asc_file_path, 'r') as f:
                # Skip the header lines (usually 6 lines in an ASC file)
                for _ in range(1):
                    f.readline()
                # Read the data and store it in a 2D numpy array
                data = np.loadtxt(f)
            # to add column names using codetable file
            desired_columns = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                               'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors',
                               'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
                               'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk']
            data = pd.DataFrame(data, columns=desired_columns)
            data = self.standardize_df(data)
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_south_german_credit.' % e)

    def load_icmla_2014_accepted_papers_data_set(self):
        try:
            csv_file_path = '/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment4/' \
                            'datasets/icmla+2014+accepted+papers+data+set/ICMLA_2014.csv'
            data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            # convert text features to numeric
            # Create a list of the textual columns to convert
            textual_columns = ['paper_title', 'author_keywords', 'abstract']
            # Combine all the textual data into a single column for processing
            data['combined_text'] = data[textual_columns].apply(lambda x: ' '.join(x), axis=1)
            # Initialize the CountVectorizer to convert text to word counts
            vectorizer = CountVectorizer()
            # Fit and transform the combined text data to obtain the word count matrix
            word_count_matrix = vectorizer.fit_transform(data['combined_text'])
            # Convert the word count matrix to a DataFrame
            word_count_df = pd.DataFrame(word_count_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            # Concatenate the original DataFrame with the word count DataFrame
            df_numeric = pd.concat([data.drop(columns=textual_columns), word_count_df], axis=1)
            # Assuming you already have the df_numeric DataFrame
            # Rearrange columns
            df_numeric = df_numeric[[col for col in df_numeric.columns if col != 'session'] + ['session']]
            # Drop the 'combined_text' column
            df_numeric.drop('combined_text', axis=1, inplace=True)
            df_numeric = self.standardize_df(df_numeric)
            return df_numeric
        except Exception as e:
            self.logger.error('Exception %s occurred during load_icmla_2014_accepted_papers_data_set.' % e)

    def load_parking_birmingham_data_set(self):
        try:
            csv_file_path = '/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment4/' \
                            'datasets/parking+birmingham/dataset.csv'
            data = pd.read_csv(csv_file_path)
            # Assuming you already have the DataFrame 'data' with the dataset
            # If 'LastUpdated' is not in datetime format, convert it to datetime first
            data['LastUpdated'] = pd.to_datetime(data['LastUpdated'])
            # Convert 'LastUpdated' to numeric (timestamp) representation
            data['LastUpdated'] = data['LastUpdated'].astype(int)
            # Move the 'SystemCodeNumber' column to the last position
            data = data[[col for col in data.columns if col != 'SystemCodeNumber'] + ['SystemCodeNumber']]
            # Now 'data' has the 'LastUpdated' column converted to numeric and 'SystemCodeNumber'
            # moved to the last position
            data = self.standardize_df(data)
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_parking_birmingham_data_set.' % e)

    def standardize_df(self, df):
        try:
            # Separate the last column from the rest of the DataFrame
            last_column = df.iloc[:, -1]
            data_columns = df.iloc[:, :-1]
            # Initialize the StandardScaler
            scaler = StandardScaler()
            # Standardize the data columns
            standardized_data = scaler.fit_transform(data_columns)
            # Create a new DataFrame with standardized data and the last column
            standardized_df = pd.DataFrame(standardized_data, columns=data_columns.columns)
            standardized_df[df.columns[-1]] = last_column
            return standardized_df
        except Exception as e:
            self.logger.error('Exception %s occurred during standardize_df.' % e)


if __name__ == '__main__':
    dataset_handler = DatasetHandler()
    south_german_credit_data = dataset_handler.load_south_german_credit()
    icmla_2014_accepted_papers_data = dataset_handler.load_icmla_2014_accepted_papers_data_set()
    parking_birmingham_data = dataset_handler.load_parking_birmingham_data_set()
    pass
