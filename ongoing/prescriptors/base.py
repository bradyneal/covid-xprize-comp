from abc import ABCMeta, abstractmethod
import numpy as np
import os
import pandas as pd
from datetime import datetime, date, timedelta
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor

SEED = 0
DEFAULT_TEST_COST = 'covid_xprize/validation/data/uniform_random_costs.csv'
TEST_CONFIGS = [
    ('Default', {'start_date': '2020-08-01', 'end_date': '2020-08-05', 'costs': DEFAULT_TEST_COST}),
    ('Jan2021_EC', {'start_date': '2021-01-01', 'end_date': '2021-01-31', 'costs': 'equal'}),
    ('Jan2021_RC', {'start_date': '2021-01-01', 'end_date': '2021-01-31'}),
    ('Jan2021_EC_NoDec', {'start_date': '2021-01-01', 'end_date': '2021-01-31', 'costs': 'equal', 'train_end_date': '2020-11-30'}),
    ('Jan2021_RC_NoDec', {'start_date': '2021-01-01', 'end_date': '2021-01-31', 'train_end_date': '2020-11-30'}),
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
PREDICTOR_PATH = 'covid_xprize/standard_predictor/xprize_predictor/models/trained_model_weights.h5'

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

def gen_test_config(predictor=None, start_date=None, end_date=None, costs='random', train_start_date=None, train_end_date=None, update_data=True):
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
    assert start_date is not None and end_date is not None
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
    fill_missing_values(df, dropifnocases=True, dropifnodeaths=False)

    # Test dataframe consists of NPI values up to start_date-1
    pd_start_date = pd.to_datetime(start_date)
    test_df = df[df['Date'] < pd_start_date].copy()
    test_columns = ['GeoID', 'CountryName', 'RegionName', 'Date'] + NPI_COLUMNS
    test_df = test_df[test_columns]

    # Discard countries that will not be evaluated
    country_df = pd.read_csv(COUNTRY_LIST,
                             encoding="ISO-8859-1",
                             dtype={"RegionName": str},
                             error_bad_lines=False)
    test_df = test_df.merge(country_df, on=['RegionName','CountryName'], how='right', suffixes=('', '_y'))

    if (costs is not None) and (costs not in ['equal', 'random']):
        cost_df = pd.read_csv(costs)
    else:
        cost_df = generate_costs(test_df, mode=costs)

    cost_df["GeoID"] = np.where(cost_df["RegionName"].isnull(),
                                cost_df["CountryName"],
                                cost_df["CountryName"] + ' / ' + cost_df["RegionName"])

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)
    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)
    # Compute smoothed versions of new cases and deaths each day
    window_size = 7
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

    train_df = df
    # forget all historical data starting from start_date
    train_df = train_df[df['Date'] < pd_start_date]
    if predictor is not None:
        predictor.df = predictor.df[predictor.df['Date'] < pd_start_date]

    if train_start_date is not None:
        # forget all historical data before train_start_date
        pd_train_start_date = pd.to_datetime(train_start_date)
        train_df = train_df[pd_train_start_date <= df['Date']]
        if predictor is not None:
            predictor.df = predictor.df[pd_train_start_date <= predictor.df['Date']]

    if train_end_date is not None:
        # forget all historical data after train_end_date
        pd_train_end_date = pd.to_datetime(train_end_date)
        train_df = train_df[train_df['Date'] <= pd_train_end_date]
        if predictor is not None:
            predictor.df = predictor.df[predictor.df['Date'] <= pd_train_end_date]

    return train_df, test_df, cost_df


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


def generate_costs(df, mode='random'):
    """
    Returns df of costs for each IP for each geo according to distribution.

    Costs always sum to #IPS (i.e., len(NPI_COLUMNS)).

    Available distributions:
        - 'ones': cost is 1 for each IP.
        - 'uniform': costs are sampled uniformly across IPs independently
                     for each geo.
    """
    assert mode in ['equal', 'random'], \
           f'Unsupported mode {mode}'

    # reduce df to one row per geo
    df = df.groupby(['CountryName', 'RegionName'], dropna=False).mean().reset_index()

    # reduce to geo id info
    df = df[['CountryName', 'RegionName']]

    if mode == 'equal':
        df[NPI_COLUMNS] = 1

    elif mode == 'random':
        # generate weights uniformly for each geo independently.
        nb_geos = len(df)
        nb_ips = len(NPI_COLUMNS)
        samples = np.random.uniform(size=(nb_ips, nb_geos))
        weights = nb_ips * samples / samples.sum(axis=0)
        df[NPI_COLUMNS] = weights.T

    return df

def weight_prescriptions_by_cost(presc_df, cost_df):
    """
    Weight prescriptions by their costs.
    """
    weighted_df = presc_df.merge(cost_df, how='outer', on=['CountryName', 'RegionName'], suffixes=('_pres', '_cost'))
    for npi_col in NPI_COLUMNS:
        weighted_df[npi_col] = weighted_df[npi_col + '_presc'] * weighted_df[npi_col + '_cost']
    return weighted_df

class BasePrescriptorMeta(ABCMeta):
    """
    Forces subclasses to implement abstract_attributes
    """

    abstract_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(BasePrescriptorMeta, cls).__call__(*args, **kwargs)
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


class BasePrescriptor(object, metaclass=BasePrescriptorMeta):
    """
    Abstract class for predictors. Currently provides train/test data splits and
    evaluation for classes that inherit from this class. In the future, this
    class will hold common code for preprocessing, plotting, other evaluations,
    etc. Requires that subclasses implement 2 methods and 2 attributes:

    2 methods:
        fit(data: pd.DataFrame) - train the model using the standard predictor and some historical real data
        prescribe(start_date_str: str,
                  end_date_str: str,
                  prior_ips: pd.DataFrame
                  costs: pd.DataFrame) -> pd.DataFrame - make prescriptions for the given period

    1 attribute:
        predictor - standard predictor model
    """

    abstract_attributes = ['predictor']

    def __init__(self, seed=SEED):
        if seed is not None:
            self.set_seed(seed)

        self.predictor = XPrizePredictor(PREDICTOR_PATH, OXFORD_FILEPATH)

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def prescribe(self, start_date_str, end_date_str, prior_ips, costs):
        pass

    def evaluate(self, output_file_path=None):
        for test_name, test_config in TEST_CONFIGS:
            print('Running test:', test_name)
            train_df, test_df, cost_df = gen_test_config(predictor=self.predictor, **test_config)
            start_date, end_date = test_config['start_date'], test_config['end_date']

            # train the model
            self.fit(train_df)

            # generate prescriptions
            print('...generating prescriptions')
            presc_df = self.prescribe(start_date_str=test_config['start_date'],
                                      end_date_str=test_config['end_date'],
                                      prior_ips=test_df,
                                      costs=cost_df)

            # check if all required columns are in the returned dataframe
            assert 'Date' in presc_df.columns()
            assert 'CountryName' in presc_df.columns()
            assert 'RegionName' in presc_df.columns()
            assert 'PrescriptionIndex' in presc_df.columns()
            for npi_col in NPI_COLUMNS:
                assert npi_col in presc_df.columns()

            # generate predictions with the given prescriptions
            print('...generating predictions for all prescriptions')
            pred_dfs = []
            for idx in presc_df['PrescriptionIndex'].unique():
                idx_df = presc_df[presc_df['PrescriptionIndex'] == idx]
                idx_df = idx_df.drop(columns='PrescriptionIndex') # Predictor doesn't need this
                ip_file_path = 'prescriptions/prescription_{}.csv'.format(idx)
                os.makedirs(os.path.dirname(ip_file_path), exist_ok=True)
                idx_df.to_csv(ip_file_path)
                pred_df = self.predictor.predict(start_date, end_date, ip_file_path)
                pred_dfs.append(pred_df)
            pred_df = pd.concat(pred_dfs)

            # aggregate cases by prescription index and geo
            agg_pred_df = pred_df.groupby(['CountryName',
                                           'RegionName',
                                           'PrescriptionIndex'], dropna=False).mean().reset_index()

            # only use costs of geos we've predicted for
            cost_df = cost_df[cost_df.CountryName.isin(agg_pred_df.CountryName) &
                      cost_df.RegionName.isin(agg_pred_df.RegionName)]

            # apply weights to prescriptions
            presc_df = weight_prescriptions_by_cost(presc_df, cost_df)

            # aggregate stringency across npis
            presc_df['Stringency'] = presc_df[NPI_COLUMNS].sum(axis=1)

            # aggregate stringency by prescription index and geo
            agg_presc_df = presc_df.groupby(['CountryName',
                                            'RegionName',
                                            'PrescriptionIndex'], dropna=False).mean().reset_index()

            # combine stringency and cases into a single df
            df = agg_presc_df.merge(agg_pred_df, how='outer', on=['CountryName',
                                                                  'RegionName',
                                                                  'PrescriptionIndex'])

            # only keep columns of interest
            df = df[['CountryName',
                     'RegionName',
                     'PrescriptionIndex',
                     'PredictedDailyNewCases',
                     'Stringency']]

            # maybe save the results to a csv
            if output_file_path is not None:
                df.to_csv(output_file_path)

            # show average (stringency, new_cases) values for each PrescriptionIndex
            print(df.groupby('PrescriptionIndex').mean().reset_index())

    @staticmethod
    def set_seed(seed=SEED):
        np.random.seed(seed)


if __name__ == '__main__':
    # Run and print different test configurations
    for test_name, test_config in TEST_CONFIGS:
        train_df, test_df, cost_df = gen_test_config(**test_config)
        print(test_name)
        print('test dates')
        print(test_df.Date.unique)
        print('train dates')
        print(train_df.Date.unique)
        sample_geoID = 'Canada'
        sample_costs = cost_df[cost_df['GeoID'] == sample_geoID][NPI_COLUMNS].head(1)
        print('NPI costs for', sample_geoID)
        for npi_col in NPI_COLUMNS:
            print(npi_col, round(float(sample_costs[npi_col]), 2))
        print('Sum', round(float(sample_costs.sum(axis=1)), 2))
        print()
