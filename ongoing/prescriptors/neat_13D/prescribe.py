import os
import argparse
import time

import numpy as np
import pandas as pd

import zipfile
import os.path

from ongoing.prescriptors.neat_13D.neat_prescriptor_many_objective import Neat
import ongoing.prescriptors.base as base

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.
path = os.path.join(ROOT_DIR, 'models/13d_5days/')
# num_checkpoint = 26
PRESCRIPTORS_FILE = os.path.join(path, 'neat-checkpoint-{}')
PRESCRIPTORS_FILE_ZIP = os.path.join(path, 'neat-checkpoint-{}.zip')
CONFIG_FILE = path + 'config-prescriptor-{}'

def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path,
              num_checkpoint=70) -> None:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    npi_df = pd.read_csv(path_to_prior_ips_file,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)
    npi_df = base.add_geo_id(npi_df)

    # Load historical data
    hist_df = base.load_historical_data(update_data=False)

    # Load the IP weights, so that we can use them
    # greedily for each geo.
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df = base.add_geo_id(cost_df)

    # unpack the model if required
    if os.path.isfile(PRESCRIPTORS_FILE.format(num_checkpoint)):
        print("Model file exists")
    else:
        print("Please unzip model file {}".format(PRESCRIPTORS_FILE_ZIP.format(num_checkpoint)))
        exit(-2)

    # Load the trained prescriptor and generate the prescriptions
    prescriptor = Neat(prescriptors_file=PRESCRIPTORS_FILE.format(num_checkpoint), hist_df=hist_df,
                       config_file=CONFIG_FILE.format(num_checkpoint))
    prescription_df = prescriptor.prescribe(start_date_str, end_date_str, npi_df, cost_df)

    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    start = time.time()
    checkpoint_num = 70
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file, checkpoint_num)
    end = time.time()
    print('TIME elapsed:', (end - start) / 60, 'minutes')
    print("Done!")
