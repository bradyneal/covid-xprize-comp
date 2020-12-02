from abc import ABCMeta, abstractmethod
import numpy as np
import os
import pandas as pd
from datetime import datetime, date, timedelta

SEED = 0
TEST_CONFIGS = [
    ('Nov', {'end_month': 11, 'n_test_months': 1}),
    ('Oct-Nov', {'end_month': 11, 'n_test_months': 2}),
    ('Sep-Nov', {'end_month': 11, 'n_test_months': 3}),
    ('Jan', {'start_month': 1, 'end_month': 1}),
    ('Feb', {'start_month': 2, 'end_month': 2}),
    ('Jan-Feb', {'start_month': 1, 'end_month': 2}),
    ('180-day', {'end_month': 11, 'n_test_days': 180}),
]

OUR_FOLDER = 'ongoing'
DATA_FOLDER = 'data'
OXFORD_FILEPATH = os.path.join(OUR_FOLDER, DATA_FOLDER, 'OxCGRT_latest.csv')
OXFORD_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'


def load_train_test(n_test_months=1, end_month=11, n_test_days=None,
                    start_month=None, update_data=False):
    """
    Loads the data and splits it into train and test sets

    Args:
        n_test_months: Number of months in the test set
        end_month: Last month in the test set
        n_test_days: Number of days in the test set
        start_month: First month in the test set
        update_data: Boolean for whether to re-download the Oxford data

    Returns: (train_df, test_df)
    """
    if update_data:
        print('Updating Oxford data...', end=' ')
        df = pd.read_csv(OXFORD_URL,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         dtype={"RegionName": str,
                                "RegionCode": str},
                         error_bad_lines=False)
        df.to_csv(OXFORD_FILEPATH)
        print('DONE')
    else:
        df = pd.read_csv(OXFORD_FILEPATH,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         dtype={"RegionName": str,
                                "RegionCode": str},
                         error_bad_lines=False)
        print('Using existing data up to date {}'.format(str(df.Date[len(df) - 1]).split()[0]))

    assert 1 <= end_month <= 11
    end_test = datetime(2020, end_month + 1, 1)
    if start_month is not None:
        start_test = datetime(2020, start_month, 1)
        test = (start_test <= df['Date']) & (df['Date'] < end_test)
        test_df = df[test]
        train_df = df[~test & (df['Date'] < datetime(2020, 12, 1))]   # Hard-coded end date
    else:
        if n_test_days is not None:
            start_test = end_test - timedelta(n_test_days)
            end_train = start_test
        elif n_test_months is not None:
            assert 0 < n_test_months < 11
            start_test = datetime(2020, end_month - n_test_months + 1, 1)
            end_train = start_test
        else:
            raise ValueError('Either n_test_months, n_test_days, or start_month must not be None')
        test_df = df[(start_test <= df['Date']) & (df['Date'] < end_test)]
        train_df = df[df['Date'] < start_test]

    return train_df, test_df


class BasePredictorMeta(ABCMeta):
    """
    Forces subclasses to implement abstract_attributes
    """

    abstract_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(BasePredictorMeta, cls).__call__(*args, **kwargs)
        missing_attributes = []
        for attr_name in obj.abstract_attributes:
            if not hasattr(obj, attr_name):
                missing_attributes.append(attr_name)
        if len(missing_attributes) == 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attribute '{}'. "
                "You must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes[0], missing_attributes[0]
                )
            )
        elif len(missing_attributes) > 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attributes {}. "
                "For example, you must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes, missing_attributes[0]
                )
            )

        return obj


class BasePredictor(object, metaclass=BasePredictorMeta):
    """
    Abstract class for predictors. Currently provides train/test data splits and
    evaluation for classes that inherit from this class. In the future, this
    class will hold common code for preprocessing, plotting, other evaluations,
    etc. Requires that subclasses implement 2 methods and 2 attributes:

    2 methods:
        fit() - fit the model to the training data
        predict(data) - make predictions on the given data

    2 attributes:
        train_df - train DataFrame
        test_df - test DataFrame
    """

    abstract_attributes = ['train_df', 'test_df']

    def __init__(self, seed=SEED):
        if seed is not None:
            self.set_seed(seed)

        self.train_df = None
        self.test_df = None

    def choose_train_test_split(self, n_test_months=1, end_month=11,
                                n_test_days=None, start_month=None,
                                update_data=False):
        self.train_df, self.test_df = \
            load_train_test(n_test_months=n_test_months, end_month=end_month,
                            n_test_days=n_test_days, start_month=start_month,
                            update_data=update_data)

    def train_df(self):
        return self.train_df

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def evaluate(self):
        results = {}
        for test_name, test_config in TEST_CONFIGS:
            print('Running test:', test_name)
            train_df, test_df = load_train_test(**test_config)
            self.fit(train_df)
            train_mae = np.abs(self.predict(train_df) - train_df['ConfirmedCases']).mean()
            test_mae = np.abs(self.predict(test_df) - train_df['ConfirmedCases']).mean()
            results[test_name + ' Train MAE'] = train_mae
            results[test_name + ' Test MAE'] = test_mae
        print(results)
        return results

    def set_seed(self, seed=SEED):
        np.random.seed(seed)


if __name__ == '__main__':
    # Run and print different train/test splits
    for test_name, test_config in TEST_CONFIGS:
        train_df, test_df = load_train_test(**test_config)
        print(test_name)
        print('test dates')
        print(test_df.Date.unique)
        print('train dates')
        print(train_df.Date.unique)
