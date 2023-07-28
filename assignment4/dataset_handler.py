import pandas as pd

import global_conf
import numpy as np
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
            data = pd.DataFrame(data)
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_south_german_credit.' % e)

    def load_icmla_2014_accepted_papers_data_set(self):
        try:
            csv_file_path = '/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment4/' \
                            'datasets/icmla+2014+accepted+papers+data+set/ICMLA_2014.csv'
            data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_icmla_2014_accepted_papers_data_set.' % e)

    def load_parking_birmingham_data_set(self):
        try:
            csv_file_path = '/Users/tomerdoitshman/Desktop/D/Courses/ML_course/course_assignments/assignment4/' \
                            'datasets/parking+birmingham/dataset.csv'
            data = pd.read_csv(csv_file_path)
            return data
        except Exception as e:
            self.logger.error('Exception %s occurred during load_parking_birmingham_data_set.' % e)


if __name__ == '__main__':
    dataset_handler = DatasetHandler()
    south_german_credit_data = dataset_handler.load_south_german_credit()
    icmla_2014_accepted_papers_data = dataset_handler.load_icmla_2014_accepted_papers_data_set()
    parking_birmingham_data = dataset_handler.load_parking_birmingham_data_set()
    pass
