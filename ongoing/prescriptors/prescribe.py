import os
import argparse
import numpy as np
import pandas as pd
import time

import zipfile
import os.path

from ongoing.prescriptors.neat_multi.neat_prescriptor_many_objective import Neat as Neat2D
from ongoing.prescriptors.neat_13D.neat_prescriptor_many_objective import Neat as Neat13D
from ongoing.prescriptors.bandit.bandit_prescriptor import Bandit
from ongoing.prescriptors.heuristic.heuristic_prescriptor import Heuristic
from ongoing.prescriptors.blind_greedy.blind_greedy_prescriptor import BlindGreedy
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
import ongoing.prescriptors.base as base

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.

# Neat2D configs
NEAT2D_PATH = os.path.join(ROOT_DIR, 'neat_multi/models/5days-results-2d-1-hidden/')
NEAT2D_CHECKPOINT_0 = 26
NEAT2D_FILE_0 = os.path.join(NEAT2D_PATH, 'neat-checkpoint-{}'.format(NEAT2D_CHECKPOINT_0))
NEAT2D_FILE_ZIP_0 = os.path.join(NEAT2D_PATH, 'neat-checkpoint-{}.zip'.format(NEAT2D_CHECKPOINT_0))
NEAT2D_CONFIG_FILE_0 = NEAT2D_PATH + 'config-prescriptor-{}'.format(NEAT2D_CHECKPOINT_0)
NEAT2D_CHECKPOINT_1 = 45
NEAT2D_FILE_1 = os.path.join(NEAT2D_PATH, 'neat-checkpoint-{}'.format(NEAT2D_CHECKPOINT_1))
NEAT2D_FILE_ZIP_1 = os.path.join(NEAT2D_PATH, 'neat-checkpoint-{}.zip'.format(NEAT2D_CHECKPOINT_1))
NEAT2D_CONFIG_FILE_1 = NEAT2D_PATH + 'config-prescriptor-{}'.format(NEAT2D_CHECKPOINT_1)

# Neat13D configs
NEAT13D_PATH = os.path.join(ROOT_DIR, 'neat_13D/models/13d_5days/')
NEAT13D_CHECKPOINT_0 = 50
NEAT13D_FILE_0 = os.path.join(NEAT13D_PATH, 'neat-checkpoint-{}'.format(NEAT13D_CHECKPOINT_0))
NEAT13D_FILE_ZIP_0 = os.path.join(NEAT13D_PATH, 'neat-checkpoint-{}.zip'.format(NEAT13D_CHECKPOINT_0))
NEAT13D_CONFIG_FILE_0 = NEAT13D_PATH + 'config-prescriptor-{}'.format(NEAT13D_CHECKPOINT_0)
NEAT13D_CHECKPOINT_1 = 70
NEAT13D_FILE_1 = os.path.join(NEAT13D_PATH, 'neat-checkpoint-{}'.format(NEAT13D_CHECKPOINT_1))
NEAT13D_FILE_ZIP_1 = os.path.join(NEAT13D_PATH, 'neat-checkpoint-{}.zip'.format(NEAT13D_CHECKPOINT_1))
NEAT13D_CONFIG_FILE_1 = NEAT13D_PATH + 'config-prescriptor-{}'.format(NEAT13D_CHECKPOINT_1)

PREDICTOR_PATH = 'covid_xprize/standard_predictor/models/trained_model_weights.h5'
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, 'data')
OXFORD_FILEPATH = os.path.join(DATA_DIR, 'OxCGRT_latest.csv')

# number of prescriptions
NUM_REQUIRED = 10

# limits on the allowed values of stringency for our solutions
MIN_STRINGENCY = 0.5
MAX_STRINGENCY = 35

def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

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
    if os.path.isfile(NEAT2D_FILE_0):
        print("Model file exists")
    else:
        print("Un-zipping model file")
        with zipfile.ZipFile(NEAT2D_FILE_ZIP_0, 'r') as zip_ref:
            zip_ref.extractall(NEAT2D_PATH)

    # unpack the model if required
    if os.path.isfile(NEAT2D_FILE_1):
        print("Model file exists")
    else:
        print("Un-zipping model file")
        with zipfile.ZipFile(NEAT2D_FILE_ZIP_1, 'r') as zip_ref:
            zip_ref.extractall(NEAT2D_PATH)

    # # unpack the model if required
    # if os.path.isfile(NEAT13D_FILE_0):
    #     print("Model file exists")
    # else:
    #     print("Please unzip model file {}".format(NEAT13D_FILE_ZIP_0))
    #     exit(-2)

    # # unpack the model if required
    # if os.path.isfile(NEAT13D_FILE_1):
    #     print("Model file exists")
    # else:
    #     print("Please unzip model file {}".format(NEAT13D_FILE_ZIP_1))
    #     exit(-2)

    # instantiate prescriptors
    # bandit = Bandit(load=True, hist_df=hist_df, verbose=False)
    neat2d_ad15_ckp0 = Neat2D(prescriptors_file=NEAT2D_FILE_0, hist_df=hist_df, config_file=NEAT2D_CONFIG_FILE_0, action_duration=15)
    neat2d_ad15_ckp1 = Neat2D(prescriptors_file=NEAT2D_FILE_1, hist_df=hist_df, config_file=NEAT2D_CONFIG_FILE_1, action_duration=15)
    neat2d_ad1 = Neat2D(prescriptors_file=NEAT2D_FILE_0, hist_df=hist_df, config_file=NEAT2D_CONFIG_FILE_0, action_duration=1)
    # neat13d_ckp0 = Neat13D(prescriptors_file=NEAT13D_FILE_0, hist_df=hist_df, config_file=NEAT13D_CONFIG_FILE_0)
    # neat13d_ckp1 = Neat13D(prescriptors_file=NEAT13D_FILE_1, hist_df=hist_df, config_file=NEAT13D_CONFIG_FILE_1)
    heuristic = Heuristic()
    blind_greedy = BlindGreedy()
    prescriptors = {
        # 'Bandit': bandit,
        'Neat2D_AD1': neat2d_ad1,
        'Neat2D_AD15_CKP0': neat2d_ad15_ckp0,
        'Neat2D_AD15_CKP1': neat2d_ad15_ckp1,
        # 'Neat13D_CKP0': neat13d_ckp0,
        # 'Neat13D_CKP1': neat13d_ckp1,
        'Heuristic': heuristic,
        'BlindGreedy': blind_greedy,
    }

    # generate prescriptions and predictions for all prescriptors
    prescriptions = {}
    eval = []
    predictor = XPrizePredictor(PREDICTOR_PATH, OXFORD_FILEPATH)
    for presc_name, presc in prescriptors.items():
        try:  # in case anything goes wrong with this prescriptor, ignore it and move to the next one
            print('Generating prescriptions for {}'.format(presc_name))
            pres_df = presc.prescribe(start_date_str, end_date_str, npi_df, cost_df)
            pres_df = base.add_geo_id(pres_df)
            prescriptions[presc_name] = pres_df

            # generate predictions with the given prescriptions
            print('Generating predictions for {}'.format(presc_name))
            pred_dfs = []
            for idx in pres_df['PrescriptionIndex'].unique():
                idx_df = pres_df[pres_df['PrescriptionIndex'] == idx]
                idx_df = idx_df.drop(columns='PrescriptionIndex') # predictor doesn't need this
                last_known_date = predictor.df['Date'].max()
                if last_known_date < pd.to_datetime(idx_df['Date'].min()) - np.timedelta64(1, 'D'):
                    # append prior NPIs to the prescripted ones because the predictor will need them
                    idx_df = idx_df.append(npi_df[npi_df['Date'] > last_known_date].drop(columns='GeoID'))
                pred_df = predictor.predict(start_date_str, end_date_str, idx_df)
                pred_df['PrescriptionIndex'] = idx
                pred_dfs.append(pred_df)
            pred_df = pd.concat(pred_dfs)
            # aggregate cases by prescription index and geo
            agg_pred_df = pred_df.groupby(['CountryName',
                                           'RegionName',
                                           'PrescriptionIndex'], dropna=False).mean().reset_index()

            # only use costs of geos we've predicted for
            cost_df = cost_df[cost_df['CountryName'].isin(agg_pred_df['CountryName']) &
                      cost_df['RegionName'].isin(agg_pred_df['RegionName'])]

            # apply weights to prescriptions
            pres_df = base.weight_prescriptions_by_cost(pres_df, cost_df)

            # aggregate stringency across npis
            pres_df['Stringency'] = pres_df[base.NPI_COLUMNS].sum(axis=1)

            # aggregate stringency by prescription index and geo
            agg_pres_df = pres_df.groupby(['CountryName',
                                           'RegionName',
                                           'PrescriptionIndex'], dropna=False).mean().reset_index()

            # combine stringency and cases into a single df
            df = agg_pres_df.merge(agg_pred_df, how='outer', on=['CountryName',
                                                                 'RegionName',
                                                                 'PrescriptionIndex'])

            # only keep columns of interest
            df = df[['CountryName',
                     'RegionName',
                     'PrescriptionIndex',
                     'PredictedDailyNewCases',
                     'Stringency']]
            df['TestName'] = 'NA'
            df['source'] = presc_name
            eval.append(df)
        except:
            print('Something went wrong with {}. Skipping it.'.format(presc_name))
            continue

    # run the aggregation to find the best prescriptions
    pareto_presc = aggregate_results(eval)

    agg_df_dict = {'CountryName': [],
                   'RegionName': [],
                   'Date': [],
                   'PrescriptionIndex': []}
    for npi in base.NPI_COLUMNS:
        agg_df_dict[npi] = []

    # get the best prescription for each prescription index and for each geo
    winner = {presc_name: 0 for presc_name in prescriptors}
    for idx in pareto_presc['PrescriptionIndex'].unique():
        for geo in pareto_presc['GeoID'].unique():
            tmp_df = pareto_presc[(pareto_presc['PrescriptionIndex'] == idx) & (pareto_presc['GeoID'] == geo)]
            src_name = tmp_df['source'].tolist()[0]
            src_idx = tmp_df['source-PrescriptionIndex'].tolist()[0]
            pres_df = prescriptions[src_name][(prescriptions[src_name]['PrescriptionIndex'] == src_idx) & (prescriptions[src_name]['GeoID'] == geo)]

            country_name = pres_df['CountryName'].tolist()
            region_name = pres_df['RegionName'].tolist()
            dates = pres_df['Date'].tolist()

            agg_df_dict['CountryName'].extend(country_name)
            agg_df_dict['RegionName'].extend(region_name)
            agg_df_dict['Date'].extend(dates)
            agg_df_dict['PrescriptionIndex'].extend([idx for date in dates])
            for npi in base.NPI_COLUMNS:
                agg_df_dict[npi].extend(pres_df[npi].tolist())

            winner[src_name] += 1

    agg_df = pd.DataFrame(agg_df_dict)

    # Create the directory for writing the output file, if necessary.
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    except:
        pass

    # Save output csv file.
    agg_df.to_csv(output_file_path, index=False)

    print('No. of times each model was chosen:')
    for presc_name, wins in winner.items():
        print('{}: {}'.format(presc_name, wins))

    return

def get_non_dom_ind(vals_arr, verbose=True):
    res_arr = []
    for i in range(0, len(vals_arr)):
        point1 = vals_arr[i]
        is_dominated = False
        for j in range(0, len(vals_arr)):
            if i == j:
                continue
            point2 = vals_arr[j]
            if ((point2[0] < point1[0]) and (point2[1] <= point1[1])) or (
                    (point2[0] <= point1[0]) and (point2[1] < point1[1])):
                if verbose:
                    print('{} is dominated by {}'.format(i, j))
                is_dominated = True
                break
            pass
        if not is_dominated:
            res_arr.append(i)
        pass
    return res_arr


def get_best_n_points_no_curve(n, arr_input):
    arr_list = arr_input.tolist()
    arr_list.sort(reverse=True, key=lambda x: x[0])
    arr_list = np.array(arr_list)

    # 2. find arc length
    arc_len_arr = []
    for pos in range(0, len(arr_input) - 1):
        p1 = np.array([arr_list[pos][0], arr_list[pos][1]])
        p2 = np.array([arr_list[pos + 1][0], arr_list[pos + 1][1]])
        arc_len_arr.append(np.linalg.norm(p2 - p1))
    arc_len_arr = np.array(arc_len_arr)
    # distance delta
    d = sum(arc_len_arr) / (n - 1)
    # cumul_sum of art length
    arc_len_arr_cum = np.cumsum(arc_len_arr)

    # 3. choose ref. points
    # positions of reference points
    points_pos = [0]
    j = 1
    for i in range(0, len(arc_len_arr_cum)):
        if arc_len_arr_cum[i] >= j * d:
            points_pos.append(i + 1)
            j += 1
            if j == n - 1:
                break
        pass
    points_pos.append(len(arr_list) - 1)

    chosen_points = []
    for ref_point_pos in points_pos:
        ref_point = arr_list[ref_point_pos]
        dist = np.linalg.norm((arr_input - ref_point), axis=1)
        pos = np.argmin(dist)
        chosen_points.append(pos)
        pass

    return chosen_points


def aggregate_results(df_arr):

    # get all test
    all_tests = []
    for df in df_arr:
        all_tests.extend(df['TestName'].unique())
        pass
    # remove duplicates
    all_tests = list(set(all_tests))

    # test = all_tests[0]

    final_dfs = []
    for test in all_tests:
        print('Processing {}'.format(test))
        # generate sub_dfs that will contain only the results for the required test
        test_df_arr = []
        for df in df_arr:
            test_df = df.loc[df['TestName'] == test].copy()
            test_df = base.add_geo_id(test_df)
            test_df_arr.append(test_df)
            pass

        # get all geo-s in this test
        all_geos = set()
        for test_df in test_df_arr:
            geos = test_df['GeoID'].unique()
            all_geos.update(set(geos))
            pass
        all_geos = list(all_geos)

        # for every geo
        for curr_geo in all_geos:
            # put all prescriptors for the current geo in the same df
            prescr_df = []
            for test_df in test_df_arr:
                df = test_df.loc[test_df['GeoID'] == curr_geo].copy()
                df['source'] = test_df['source']
                prescr_df.append(df)
                pass
            # put them all together and remove duplicates
            prescr_df = pd.concat(prescr_df).drop_duplicates(['PredictedDailyNewCases', 'Stringency'])

            # remove solutions with (Stringency < min_stringency) or (Stringency > max_stringency)
            prescr_df.reset_index(inplace=True, drop=True)
            idx_to_drop = prescr_df[(prescr_df['Stringency'] < MIN_STRINGENCY) | (prescr_df['Stringency'] > MAX_STRINGENCY)].index
            prescr_df.drop(idx_to_drop, inplace=True)

            # choose non-dominated
            arr = prescr_df[['PredictedDailyNewCases', 'Stringency']].values
            non_dom_idx = get_non_dom_ind(arr, verbose=False)

            # check here if we have at least 10
            if len(non_dom_idx) < NUM_REQUIRED:
                # print('less than 10: {}'.format(curr_geo))
                # just copy the first point the required number of times
                chosen_points_pos = non_dom_idx
                num_found = len(chosen_points_pos)
                for i in range(0, NUM_REQUIRED - num_found):
                    chosen_points_pos.append(chosen_points_pos[0])
                    pass
                chosen_non_dom = chosen_points_pos
                pass
            elif len(non_dom_idx) == NUM_REQUIRED:
                # don't do anything
                chosen_points_pos = non_dom_idx
                chosen_non_dom = chosen_points_pos
                pass
            else:
                # choose points
                chosen_points_pos = get_best_n_points_no_curve(NUM_REQUIRED, arr[non_dom_idx])
                chosen_non_dom = np.array(non_dom_idx)[chosen_points_pos]
                pass
            # create a df with chosen tests for this geo
            geo_df = prescr_df.iloc[chosen_non_dom].copy()
            # geo_df.drop('Unnamed: 0', axis=1, inplace=True)
            tmp = geo_df.copy()
            while len(geo_df) < NUM_REQUIRED:
                geo_df = pd.concat([geo_df, tmp])
            geo_df = geo_df[0: NUM_REQUIRED]
            geo_df['source-PrescriptionIndex'] = geo_df['PrescriptionIndex']
            geo_df['PrescriptionIndex'] = [i for i in range(0, NUM_REQUIRED)]
            final_dfs.append(geo_df)
            pass
        pass
    res_df = pd.concat(final_dfs)
    res_df.reset_index(drop=True, inplace=True)
    return res_df

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
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    end = time.time()
    print('TIME elapsed:', (end - start) / 60, 'minutes')
    print("Done!")
