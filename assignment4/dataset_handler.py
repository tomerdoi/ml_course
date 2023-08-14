import os.path
import re
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from logger_utils import LoggerUtils


class DatasetHandler:
    def __init__(self):
        self.logger_util = LoggerUtils()
        self.logger = self.logger_util.init_logger(log_file_name='pipeline.log')
        self.raw_south_german_credit_data = None
        self.raw_icmla_2014_accepted_papers_data = None
        self.raw_parking_birmingham_data = None

    def load_south_german_credit(self):
        try:
            asc_file_path = './datasets/south+german+credit+update/SouthGermanCredit.asc'
            with open(asc_file_path, 'r') as f:
                # Skip the header lines (usually 6 lines in an ASC file)
                f.readline()
                # Read the data and store it in a 2D numpy array
                data = np.loadtxt(f)
            # to add column names using codetable file
            desired_columns = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                               'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors',
                               'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
                               'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk']
            data = pd.DataFrame(data, columns=desired_columns)
            self.raw_south_german_credit_data = data.copy(deep=True)
            data = self.standardize_df(data)
            data.to_csv(f'{os.path.dirname(asc_file_path)}/preprocess.csv', index=False)
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_south_german_credit.' % e)

    def load_icmla_2014_accepted_papers_data_set_word2vec(self):
        def clean_keyword(val: str):
            val = ' '.join(val.split(','))
            val = ' '.join(val.split(';'))
            val.replace('-', ' ')
            pattern = re.compile(r'[^a-zA-Z\d ]')
            val = pattern.sub('', val)
            return val.lower()

        def clean_text(val: str):
            val = val.replace('-', ' ')
            val = val.replace("'s", '')
            pattern = re.compile(r'[^a-zA-Z\d ]')
            val = pattern.sub('', val)
            return val.lower()
        try:
            le = LabelEncoder()
            csv_file_path = './datasets/icmla+2014+accepted+papers+data+set/ICMLA_2014.csv'
            data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            self.raw_icmla_2014_accepted_papers_data = data.copy(deep=True)
            data.drop(columns=['paper_id'], inplace=True)
            data.paper_title = data.paper_title.apply(clean_text)
            data.abstract = data.abstract.apply(clean_text)
            data.author_keywords = data.author_keywords.apply(clean_keyword)
            data.session = le.fit_transform(data.session)

            # Combine all the textual data into a single column for processing
            textual_columns = ['paper_title', 'author_keywords', 'abstract']
            data['combined_text'] = data[textual_columns].apply(lambda x: ' '.join(x), axis=1)
            # Train Word2Vec model on the combined text
            sentences = [text.split() for text in data['combined_text']]
            word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1,
                                      sg=0)  # You can adjust parameters
            # Get word embeddings for each document's combined text
            embeddings = []
            for text in sentences:
                text_embeddings = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
                if text_embeddings:
                    embeddings.append(sum(text_embeddings) / len(text_embeddings))
                else:
                    embeddings.append([0] * word2vec_model.vector_size)  # Use a zero vector if no embeddings are found
            # Convert embeddings to DataFrame
            embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(word2vec_model.vector_size)])
            # Concatenate the original DataFrame with the embedding DataFrame
            df_numeric = pd.concat([data.drop(columns=textual_columns), embedding_df], axis=1)
            df_numeric.drop('combined_text', axis=1, inplace=True)
            # Standardize the DataFrame except for the 'session' column
            columns_to_standardize = [col for col in df_numeric.columns if col != 'session']
            df_numeric[columns_to_standardize] = self.standardize_df(df_numeric[columns_to_standardize])
            df_numeric = df_numeric[[col for col in df_numeric.columns if col != 'session'] + ['session']]
            df_numeric.to_csv(f'{os.path.dirname(csv_file_path)}/preprocess.csv', index=False)
            return df_numeric
        except Exception as e:
            self.logger.error('Exception %s occurred during load_icmla_2014_accepted_papers_data_set_word2vec.' % e)

    def load_icmla_2014_accepted_papers_data_set(self):
        try:
            csv_file_path = './datasets/icmla+2014+accepted+papers+data+set/ICMLA_2014.csv'
            data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            data.drop(columns=['paper_id'], inplace=True)
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
            df_numeric.to_csv(f'{os.path.dirname(csv_file_path)}/preprocess.csv', index=False)
            return df_numeric
        except Exception as e:
            self.logger.error('Exception %s occurred during load_icmla_2014_accepted_papers_data_set.' % e)

    def load_parking_birmingham_data_set(self):
        le = LabelEncoder()
        try:
            csv_file_path = './datasets/parking+birmingham/dataset.csv'
            data = pd.read_csv(csv_file_path)
            self.raw_parking_birmingham_data = data.copy(deep=True)
            # Assuming you already have the DataFrame 'data' with the dataset
            # If 'LastUpdated' is not in datetime format, convert it to datetime first
            data['LastUpdated'] = pd.to_datetime(data['LastUpdated'])
            # Convert 'LastUpdated' to numeric (timestamp) representation
            # in some os the default int is int32 which not compatible with the pandas datetime
            data['LastUpdated'] = data['LastUpdated'].astype(np.int64)

            # Transpose the cluster label into numerics
            data['SystemCodeNumber'] = le.fit_transform(data['SystemCodeNumber'])

            # Move the 'SystemCodeNumber' column to the last position
            data = data[[col for col in data.columns if col != 'SystemCodeNumber'] + ['SystemCodeNumber']]
            # Now 'data' has the 'LastUpdated' column converted to numeric and 'SystemCodeNumber'
            # moved to the last position
            data = self.standardize_df(data)
            data.to_csv(f'{os.path.dirname(csv_file_path)}/preprocess.csv', index=False)
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

    def datasets_eda(self):
        try:
            profile = ProfileReport(self.raw_south_german_credit_data)
            profile.to_file("./reports/raw_south_german_credit_data_eda_report.html")
            profile = ProfileReport(self.raw_icmla_2014_accepted_papers_data)
            profile.to_file("./reports/raw_icmla_2014_accepted_papers_data_eda_report.html")
            profile = ProfileReport(self.raw_parking_birmingham_data)
            profile.to_file("./reports/raw_parking_birmingham_data_eda_report.html")
        except Exception as e:
            self.logger.error('Exception %s occurred during datasets_eda.' % e)


if __name__ == '__main__':
    dataset_handler = DatasetHandler()
    south_german_credit_data = dataset_handler.load_south_german_credit()
    icmla_2014_accepted_papers_data = dataset_handler.load_icmla_2014_accepted_papers_data_set_word2vec()
    parking_birmingham_data = dataset_handler.load_parking_birmingham_data_set()
    dataset_handler.datasets_eda()
    pass
