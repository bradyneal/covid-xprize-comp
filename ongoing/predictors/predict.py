# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import pandas as pd
import numpy as np
import pickle

from econ.econ_predictor import econ_predictor
# import econ.econ_utils as econ_utils
#
# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # .../covid-xprize-comp/ongoing/predictors
# parentdir = os.path.dirname(currentdir)  # .../covid-xprize-comp/ongoing
# sys.path.insert(0,parentdir)
# print(sys.path)

from tempgeolstm.tempgeolstm_predictor import tempGeoLSTMPredictor
from tempgeolgbm.tempgeolgbm_predictor import tempGeoLGBMPredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # ..../covid-xprize-comp/ongoing/predictors
print(ROOT_DIR)

ALPHA = 0.50  # 0.50 seems to be the optimal value

# LSTM weights
# If you'd like to use a model, copy it to "trained_model_weights.h5"
# or change this MODEL_FILE path to point to your model.
MODEL_WEIGHTS_FILE = os.path.join(ROOT_DIR, "tempgeolstm", "models", "model_alldata.h5")

# LGBM weights
MODEL_FILE = os.path.join(ROOT_DIR, "tempgeolgbm", "models", "model_alldata.pkl")
ECON_MODEL_FILE = os.path.join(ROOT_DIR, 'econ', 'models', 'econ_models_1.pkl')

COUNTRIES_FILE = os.path.join(ROOT_DIR, "models", "countries.txt")
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, 'data')
DATA_FILE = os.path.join(DATA_DIR, "OxCGRT_latest.csv")
# print(os.path.abspath(DATA_FILE))  # sanity check
TEMPERATURE_DATA_FILE = os.path.join(DATA_DIR, "temperature_data.csv")

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

# --start_date 2020-12-01 --end_date 2020-12-31 --interventions_plan data/future_ip.csv --output_file 2020-12-01_2020_12_31.csv
def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!

    # Generate the predictions
    start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
    npis_df = pd.read_csv(path_to_ips_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          dtype={"RegionName": str,
                                 "RegionCode": str},
                          error_bad_lines=False)
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    npis_df["GeoID"] = np.where(npis_df["RegionName"].isnull(),
                                npis_df["CountryName"],
                                npis_df["CountryName"] + ' / ' + npis_df["RegionName"])

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLUMNS:
        npis_df.update(npis_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    predictors = ['econ', "LSTM", "LGBM"]
    for model in predictors:
        if model == "LSTM":
            # predictor = tempGeoLSTMPredictor(path_to_model_weights=MODEL_WEIGHTS_FILE, path_to_geos=COUNTRIES_FILE)
            predictor = tempGeoLSTMPredictor(path_to_model_weights=MODEL_WEIGHTS_FILE, use_embedding=False)
            lstm_predictions_df = get_predictions(predictor, model, npis_df, start_date_dt, end_date_dt, output_file_path)
        elif model == "LGBM":
            # Load LGBM
            predictor = tempGeoLGBMPredictor()
            with open(MODEL_FILE, 'rb') as model_file:
                predictor.predictor = pickle.load(model_file)
            lgbm_predictions_df = get_predictions(predictor, model, npis_df, start_date_dt, end_date_dt, output_file_path)
        elif model == 'econ':
            # econ prediction try-catch loop
            try:
            # get econ_predictions
                econ_df = econ_predictor(
                        start_date_str=start_date, 
                        end_date_str=end_date, 
                        DATA_DIR=DATA_DIR,
                        MODEL_FILE=ECON_MODEL_FILE,
                        path_to_hist_ips_file=os.path.join(DATA_DIR, "2020-09-30_historical_ip.csv"),
                        path_to_future_ips_file=path_to_ips_file)
                print('econ pred success')
            except:
                print('econ pred fail')
                continue

    ensemble_predictions = get_ensemble_pred(ALPHA, lstm_predictions_df, lgbm_predictions_df)
    
    # econ csv try-catch
    try:
        ensemble_predictions['QuarterEnd'] = ensemble_predictions['Date'] +pd.tseries.offsets.QuarterEnd()
        full_df = ensemble_predictions.merge(
            econ_df,
            how='left',
            left_on=['CountryName','RegionName', 'QuarterEnd'],
            right_on=['CountryName','RegionName','Date'],
            suffixes= (None, '_extra')
                                            )
        full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='_extra')))]
        # Create the output path
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        # Save to a csv file
        full_df.to_csv(output_file_path, index=False)
        
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(full_df)
        print('econ merge succeeded')
        print(f"Saved final model predictions to {output_file_path}")
    except:
                # Create the output path
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        # Save to a csv file
        ensemble_predictions.to_csv(output_file_path, index=False)
        
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(full_df)
        print('econ merge failed')
        print(f"Saved final model predictions to {output_file_path}")


def get_ensemble_pred(alpha, lstm_predictions_df, lgbm_predictions_df):
    ensemble_data = pd.DataFrame()
    ensemble_data['CountryName'] = lstm_predictions_df['CountryName']
    ensemble_data['RegionName'] = lstm_predictions_df['RegionName']
    ensemble_data['GeoID'] = lstm_predictions_df['GeoID']
    ensemble_data['Date'] = lstm_predictions_df['Date']

    ensemble_data['PredictedDailyTotalCases'] = alpha * lstm_predictions_df['PredictedDailyTotalCases'] + (1 - alpha) * lgbm_predictions_df['PredictedDailyTotalCases']
    ensemble_data['PredictedDailyNewCases'] = alpha * lstm_predictions_df['PredictedDailyNewCases'] + (1 - alpha) * lgbm_predictions_df['PredictedDailyNewCases']
    ensemble_data['PredictedDailyTotalDeaths'] = alpha * lstm_predictions_df['PredictedDailyTotalDeaths'] + (1 - alpha) * lgbm_predictions_df['PredictedDailyTotalDeaths']
    ensemble_data['PredictedDailyNewDeaths'] = alpha * lstm_predictions_df['PredictedDailyNewDeaths'] + (1 - alpha) * lgbm_predictions_df['PredictedDailyNewDeaths']

    return ensemble_data


def get_predictions(predictor, model, npis_df, start_date_dt, end_date_dt, output_file_path):
    predictor.choose_train_test_split(start_date=start_date_dt,
                                      end_date=end_date_dt,
                                      update_data=False)

    preds_df = predictor.predict(npis_df, start_date=start_date_dt, end_date=end_date_dt)
    return preds_df


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
