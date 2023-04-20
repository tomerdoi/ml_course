from my_id3 import MyID3
from data_handler import DataHandler
from entropy_calculator import EntropyCalculator
from sklearn.utils.estimator_checks import check_estimator


if __name__ == '__main__':
    estimator = MyID3()
    check_estimator(MyID3(estimator))
    data_handle = DataHandler()
    data_handle.get_best_split_check()
    entropy_calc = EntropyCalculator()
    entropy_calc.entropy_check()
    entropy_calc.ig_check()
