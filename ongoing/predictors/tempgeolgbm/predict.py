# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import pandas as pd
import numpy as np
import pickle

from tempgeolgbm_predictor import tempGeoLGBMPredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, "models", "model.pkl")
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir, 'data')
DATA_FILE = os.path.join(DATA_DIR, "OxCGRT_latest.csv")
TEMPERATURE_DATA_FILE = os.path.join(DATA_DIR, "temperature_data.csv")
COUNTRY_LIST = os.path.join(DATA_DIR, 'countries_regions.txt')

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
    # Load model
    predictor = tempGeoLGBMPredictor()
    with open(MODEL_FILE, 'rb') as model_file:
        predictor.predictor = pickle.load(model_file)
    # Generate the predictions
    start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d')

    predictor.choose_train_test_split(start_date=start_date_dt,
                                      end_date=end_date_dt)

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

    # # Discard countries that will not be evaluated
    # country_df = pd.read_csv(COUNTRY_LIST,
    #                          encoding="ISO-8859-1",
    #                          dtype={"RegionName": str},
    #                          error_bad_lines=False)
    # npis_df = npis_df.merge(country_df, on=['RegionName','CountryName'], how='right', suffixes=('', '_y'))

    preds_df = predictor.predict(npis_df, start_date=start_date_dt, end_date=end_date_dt)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")


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
