from abc import ABCMeta, abstractmethod
import numpy as np
import os
import pandas as pd
from datetime import datetime, date, timedelta

SEED = 0
TEST_CONFIGS = [
    # ('Nov', {'end_month': 11, 'n_test_months': 1}),
    # ('Oct-Nov', {'end_month': 11, 'n_test_months': 2}),
    # ('Sep-Nov', {'end_month': 11, 'n_test_months': 3}),
    # ('Jan', {'start_month': 1, 'end_month': 1}),
    # ('Feb', {'start_month': 2, 'end_month': 2}),
    # ('Jan-Feb', {'start_month': 1, 'end_month': 2}),
    # ('180-day', {'end_month': 11, 'n_test_days': 180}),
    ('last month', {'start_date': pd.to_datetime('2020-11-14', format='%Y-%m-%d'), 'end_date': pd.to_datetime('2020-12-14', format='%Y-%m-%d')})
]

OUR_FOLDER = 'ongoing'
DATA_FOLDER = 'data'
OXFORD_FILEPATH = os.path.join(OUR_FOLDER, DATA_FOLDER, 'OxCGRT_latest.csv')
OXFORD_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
ADDITIONAL_CONTEXT_FILE = os.path.join(OUR_FOLDER, DATA_FOLDER, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(OUR_FOLDER, DATA_FOLDER, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(OUR_FOLDER, DATA_FOLDER, "uk_populations.csv")
US_PREFIX = "United States / "
COUNTRY_LIST = os.path.join(OUR_FOLDER, DATA_FOLDER, 'countries_regions.txt')

NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']

def load_train_test(start_date=None, end_date=None, n_test_months=1, end_month=11, n_test_days=None,
                    start_month=None, window_size=7, dropifnocases=True,
                    dropifnodeaths=False, update_data=True):
    """
    Loads the data and splits it into train and test sets

    Args:
        start_date: First date in the test set (type: datetime)
        end_date: Last date in the test set (type: datetime)
        n_test_months: Number of months in the test set
        end_month: Last month in the test set
        n_test_days: Number of days in the test set
        start_month: First month in the test set
        update_data: Boolean for whether to re-download the Oxford data

        *** If start_date and end_date are given, n_test_months, enf_month,
        n_test_days, and start_month are ignored ***

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

    # Add unique identifier for each location (region + country)
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])

    # Load dataframe with demographics about each country
    context_df = load_additional_context_df()

    # Merge the two dataframes
    df = df.merge(context_df, on=['GeoID'], how='left', suffixes=('', '_y'))

    # Drop countries with no population data
    df.dropna(subset=['Population'], inplace=True)

    # Fill in missing values
    fill_missing_values(df, dropifnocases=dropifnocases, dropifnodeaths=dropifnodeaths)

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)
    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)
    # Compute smoothed versions of new cases and deaths each day
    df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
        window_size, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
        window_size, center=False).mean().fillna(0).reset_index(0, drop=True)
    # Compute percent change in new cases and deaths each day
    df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']
    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

    if (start_date is None) or (end_date is None):
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
    else:
        test_df = df[(start_date <= df['Date']) & (df['Date'] <= end_date)].copy()
        train_df = df[df['Date'] < start_date]

    # Discard countries that will not be evaluated
    # country_df = pd.read_csv(COUNTRY_LIST,
    #                          encoding="ISO-8859-1",
    #                          dtype={"RegionName": str},
    #                          error_bad_lines=False)
    # test_df = test_df.merge(country_df, on=['RegionName','CountryName'], how='right', suffixes=('', '_y'))

    return train_df, test_df


def load_additional_context_df():
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                        usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName']

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                          usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_uk_df)

    return additional_context_df


def fill_missing_values(df, dropifnocases=True, dropifnodeaths=False):
    df.update(df.groupby('GeoID').ConfirmedCases.apply(
        lambda group: group.interpolate(limit_area='inside')))

    df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))

    if dropifnocases:
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
    if dropifnodeaths:
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)

    # if NPI value is not available, set it to 0
    for npi_column in NPI_COLUMNS:
        df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))


def convert_ratio_to_new_cases(ratio,
                               window_size,
                               prev_new_cases_list,
                               prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]


def convert_ratios_to_total_cases(ratios,
                                  window_size,
                                  prev_new_cases,
                                  initial_total_cases,
                                  pop_size):
    total_cases_list, new_cases_list = [], []
    prev_new_cases_list = list(prev_new_cases)
    curr_total_cases = initial_total_cases
    for ratio in ratios:
        new_cases = convert_ratio_to_new_cases(ratio,
                                               window_size,
                                               prev_new_cases_list,
                                               curr_total_cases / pop_size)
        # new_cases can't be negative!
        new_cases = max(0, new_cases)
        # Which means total cases can't go down
        curr_total_cases += new_cases
        total_cases_list.append(curr_total_cases)
        # Update prev_new_cases_list for next iteration of the loop
        prev_new_cases_list.append(new_cases)
        new_cases_list.append(new_cases)
    return total_cases_list, new_cases_list


def convert_ratios_to_total_deaths(ratios,
                                   window_size,
                                   prev_new_deaths,
                                   initial_total_deaths):
    total_deaths_list, new_deaths_list = [], []
    prev_new_deaths_list = list(prev_new_deaths)
    curr_total_deaths = initial_total_deaths
    for ratio in ratios:
        new_deaths = convert_ratio_to_new_cases(ratio,
                                                window_size,
                                                prev_new_deaths_list,
                                                0.)
        # new_deaths can't be negative!
        new_deaths = max(0, new_deaths)
        # Which means total deaths can't go down
        curr_total_deaths += new_deaths
        total_deaths_list.append(curr_total_deaths)
        # Update prev_new_deaths_list for next iteration of the loop
        prev_new_deaths_list.append(new_deaths)
        new_deaths_list.append(new_deaths)
    return total_deaths_list, new_deaths_list



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

    def choose_train_test_split(self, start_date=None, end_date=None,
                                n_test_months=1, end_month=11,
                                n_test_days=None, start_month=None,
                                window_size=7, dropifnocases=True,
                                dropifnodeaths=False, update_data=True):
        self.train_df, self.test_df = \
            load_train_test(start_date=start_date, end_date=end_date,
                            n_test_months=n_test_months, end_month=end_month,
                            n_test_days=n_test_days, start_month=start_month,
                            window_size=window_size, dropifnocases=dropifnocases,
                            dropifnodeaths=dropifnodeaths, update_data=update_data)

    def train_df(self):
        return self.train_df

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def evaluate(self, report_train=False):
        results_cases, results_deaths = {}, {}
        for test_name, test_config in TEST_CONFIGS:
            print('Running test:', test_name)
            self.choose_train_test_split(**test_config, update_data=True)

            start_date = pd.to_datetime(self.train_df.Date.min(), format='%Y-%m-%d')
            end_date = pd.to_datetime(self.train_df.Date.max(), format='%Y-%m-%d')
            print('Training on data from {} up to {}'.format(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
            self.fit()

            start_date = pd.to_datetime(self.test_df.Date.min(), format='%Y-%m-%d')
            end_date = pd.to_datetime(self.test_df.Date.max(), format='%Y-%m-%d')
            print('Testing on data from {} up to {}'.format(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

            if report_train:
                train_preds = self.predict(self.train_df)
                train_preds['PredictedDailyNewCases7DMA'] = self.smoothing(eval_metric='PredictedDailyNewCases',
                                                                           dataset=train_preds)
                train_mae_cases = np.abs(train_preds['PredictedDailyNewCases7DMA'] - self.train_df['SmoothNewCases']).mean()
                results_cases[test_name + ' Train MAE'] = train_mae_cases
                if 'PredictedDailyNewDeaths' in train_preds:
                    train_preds['PredictedDailyNewDeaths7DMA'] = self.smoothing(eval_metric='PredictedDailyNewDeaths',
                                                                                dataset=train_preds)

            test_preds = self.predict(self.test_df)
            test_preds['PredictedDailyNewCases7DMA'] = self.smoothing(eval_metric='PredictedDailyNewCases',
                                                                      dataset=test_preds)
            test_mae_cases = np.abs(test_preds['PredictedDailyNewCases7DMA'] - self.test_df['SmoothNewCases']).mean()
            results_cases[test_name + ' Test MAE'] = test_mae_cases
            if 'PredictedDailyNewDeaths' in test_preds:
                test_preds['PredictedDailyNewDeaths7DMA'] = self.smoothing(eval_metric='PredictedDailyNewDeaths',
                                                                           dataset=test_preds)
                test_mae_deaths = np.abs(test_preds['PredictedDailyNewDeaths7DMA'] - self.test_df['SmoothNewDeaths']).mean()
                results_deaths[test_name + ' Test MAE'] = test_mae_deaths

            # TODO: Add new evaluation metrics to smooth
            # e.g proportion population under quarantine, ICU admissions, hospital admissions

        print()
        print('Prediction of number of cases:')
        print(results_cases)
        print()
        print('Prediction of number of deaths:')
        print(results_deaths)

        return results_cases, results_deaths

    @staticmethod
    def smoothing(eval_metric, dataset, window_size=7):
        """
        Auxiliary class for applying a "win_size" day simple moving average to smooth the evaluation metric.

        eval_metric (str): string representing the metric to be smoothed
        dataset (data frame): pandas data frame representing the dataset
        window_size (int): window size for smoothing function. Default = 7 day smoothing

        return: pandas data frame containing smoothed values
        """
        col_names = list(dataset.columns)  # get all column names of the dataset

        if eval_metric in col_names:  # check if evaluation metric is valid

            # Compute the 7 day moving average for the evaluation metric
            return dataset.groupby(
                "GeoID")[eval_metric].rolling(
                window_size, center=False).mean().reset_index(0, drop=True)
        else:
            raise NameError("The evaluation metric {} is not present in the dataset".format(eval_metric))

    @staticmethod
    def set_seed(seed=SEED):
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
