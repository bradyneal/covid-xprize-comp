import numpy as np
import pandas as pd
import os
from copy import deepcopy
import datetime

from ongoing.prescriptors.base import BasePrescriptor, PRED_CASES_COL, CASES_COL, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base
from bandit import CCTSB

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_PRED_FILE_NAME = os.path.join(ROOT_DIR, 'tmp_predictions_for_prescriptions', 'preds.csv')
TMP_PRESCRIPTION_FILE = os.path.join(ROOT_DIR, 'tmp_prescription.csv')

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
NB_EVAL_COUNTRIES = 10

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 10


class Bandit(BasePrescriptor):
    def __init__(self,
                 seed=base.SEED,
                 eval_start_date=EVAL_START_DATE,
                 eval_end_date=EVAL_END_DATE,
                 nb_eval_countries=NB_EVAL_COUNTRIES,
                 nb_prescriptions=NB_PRESCRIPTIONS,
                 hist_df=None,
                 verbose=True):

        super().__init__(seed=seed)
        self.eval_start_date = pd.to_datetime(eval_start_date, format='%Y-%m-%d')
        self.eval_end_date = pd.to_datetime(eval_end_date, format='%Y-%m-%d')
        self.nb_eval_countries = nb_eval_countries
        # self.nb_lookback_days = nb_lookback_days
        self.nb_prescriptions = nb_prescriptions
        # self.nb_generations = nb_generations
        # self.action_duration = action_duration
        # self.config_file = config_file
        # self.prescriptors_file = prescriptors_file
        self.hist_df = hist_df
        self.verbose = verbose
        self.bandit = None


    def get_predictions(self, start_date_str, end_date_str, pres_df):
        start_date = pd.to_datetime(start_date_str)
        last_known_date = self.predictor.df['Date'].max()
        if last_known_date < pd.to_datetime(self.hist_df['Date'].min()) - np.timedelta64(1, 'D'):
            # append prior NPIs to the prescripted ones because the predictor will need them
            prior_ips_df = self.hist_df[(self.hist_df['Date'] > last_known_date) & (self.hist_df['Date'] < start_date)]
            prior_ips_df = prior_ips_df[pres_df.columns()]
            ips_df = pres_df.append(prior_ips_df)
        else:
            ips_df = pres_df

        # write ips_df to file
        ips_df.to_csv(TMP_PRESCRIPTION_FILE)

        # use full path of the local file passed as ip_file
        ip_file_full_path = os.path.abspath(TMP_PRESCRIPTION_FILE)

        # generate the predictions
        pred_df = self.predictor.predict(start_date_str, end_date_str, ip_file_full_path)

        return pred_df


    def fit(self, hist_df=None):

        if hist_df is not None:
            self.hist_df = hist_df
        # As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
        # so far as the geos for evaluation.
        eval_geos = list(self.hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                         ascending=False).head(self.nb_eval_countries).index)
        if self.verbose:
            print("Bandit will be evaluated on the following geos:", eval_geos)

        # Pull out historical data for all geos
        past_cases = {}
        past_ips = {}
        for geo in eval_geos:
            geo_df = self.hist_df[self.hist_df['GeoID'] == geo]
            past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
            past_ips[geo] = np.array(geo_df[NPI_COLUMNS])

        # Gather values for scaling network output
        ip_max_values_arr = np.array([NPI_MAX_VALUES[ip] for ip in NPI_COLUMNS])

        # Sample a set of costs for all geos
        cost_df = base.generate_costs(self.hist_df, mode='random')
        cost_df = base.add_geo_id(cost_df)

        geo_costs = {}
        for geo in eval_geos:
            costs = cost_df[cost_df['GeoID'] == geo]
            cost_arr = np.array(costs[NPI_COLUMNS])[0]
            geo_costs[geo] = cost_arr

        # # make a bandit for each geo
        # for geo in eval_geos:
        #     self.bandits[geo] = CCTSB(
        #         N=[i + 1 for i in NPI_MAX_VALUES.values()], # possible values for each action,
        #         K=len(NPI_MAX_VALUES), # possible interventions,
        #         C=len(NPI_MAX_VALUES), # dimension of the context,
        #         alpha_p=0.5, #default
        #         nabla_p=0.5 #default
        #     )

        # Set up dictionary to keep track of prescription
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in NPI_COLUMNS:
            df_dict[ip_col] = []

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Compute prescribed stringency incrementally
        stringency = {date : {geo: 0. for geo in eval_geos} 
            for date in pd.date_range(self.eval_start_date, self.eval_end_date)}

        # initialize bandit
        # get context size
        eval_past_ips_len = len(next(iter(eval_past_ips.values()))[-1])
        eval_past_cases_len = len(next(iter(eval_past_cases.values()))[-1])
        geo_costs_len = len(next(iter(geo_costs.values())))

        context_size = eval_past_ips_len + eval_past_cases_len + geo_costs_len

        # fix N: possible half-values
        self.bandit = CCTSB(
            N=[i + 1 for i in NPI_MAX_VALUES.values()], # possible values for each action,
            K=geo_costs_len, # possible interventions,
            C=context_size, # dimension of the context,
            alpha_p=0.5, #default
            nabla_p=0.5 #default
        )

        # Make prescriptions one day at a time, feeding resulting
        # predictions from the predictor back into the prescriptor.
        for date in pd.date_range(self.eval_start_date, self.eval_end_date):
            date_str = date.strftime("%Y-%m-%d")

            # Prescribe for each geo
            for geo in eval_geos:
                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; zmany other approaches are possible.
                X_cases = np.log(eval_past_cases[geo][-1] + 1)
                X_ips = eval_past_ips[geo][-1]
                X_costs = geo_costs[geo]
                # print("X_cases : ", X_cases)
                # print("X_ips : ", X_ips)
                # print("X_costs : ", X_costs)
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten(),
                                    X_costs])

                # Observe context specific to geo
                self.bandit.observe(X)

                # Get prescriptions
                prescribed_ips = self.bandit.act()

                # Add it to prescription dictionary
                country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                df_dict['CountryName'].append(country_name)
                df_dict['RegionName'].append(region_name)
                df_dict['Date'].append(date_str)
                for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                    df_dict[ip_col].append(prescribed_ip)

                # Calculate stringency. This calculation could include division by
                # the number of IPs and/or number of geos, but that would have
                # no effect on the ordering of candidate solutions.
                # print(geo_costs[geo])
                # print(prescribed_ips)
                stringency[date][geo] = np.dot(geo_costs[geo],np.array(list(prescribed_ips.values())))
                # print(stringency[date][geo])
            # Create dataframe from prescriptions
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)

            # Update past data with new day of prescriptions and predictions
            pres_df = base.add_geo_id(pres_df)
            pred_df = base.add_geo_id(pred_df)
            new_pres_df = pres_df[pres_df['Date'] == date_str]
            new_pred_df = pred_df[pred_df['Date'] == date_str]
            for geo in eval_geos:
                geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

                # Append array of prescriptions
                pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in NPI_COLUMNS]).reshape(1,-1)
                eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                # Append predicted cases
                eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                    geo_pred[PRED_CASES_COL].values[0])
                
                #update bandit
                self.bandit.update(geo_pred[PRED_CASES_COL].values[0],
                              stringency[date][geo])

        return



    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips_df,
                  cost_df):

        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        geos = prior_ips_df['GeoID'].unique()

        if self.verbose:
            print("Bandit will be evaluated on the following geos:", geos)

        # Restrict it to dates before the start_date
        df = self.hist_df[self.hist_df['Date'] <= start_date]

        # Create past case data arrays for all geos
        past_cases = {}
        for geo in geos:
            geo_df = df[df['GeoID'] == geo]
            past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))

        # Create past ip data arrays for all geos
        past_ips = {}
        for geo in geos:
            geo_df = prior_ips_df[prior_ips_df['GeoID'] == geo]
            past_ips[geo] = np.array(geo_df[NPI_COLUMNS])

        # Fill in any missing case data before start_date
        # using predictor given past_ips_df.
        # Note that the following assumes that the df returned by prepare_historical_df()
        # has the same final date for all regions. This has been true so far, but relies
        # on it being true for the Oxford data csv loaded by prepare_historical_df().
        last_historical_data_date_str = df['Date'].max()
        last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                                   format='%Y-%m-%d')
        if last_historical_data_date + pd.Timedelta(days=1) < start_date:
            if self.verbose:
                print("Filling in missing data...")
            missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
            missing_data_start_date_str = datetime.strftime(missing_data_start_date,
                                                               format='%Y-%m-%d')
            missing_data_end_date = start_date - pd.Timedelta(days=1)
            missing_data_end_date_str = datetime.strftime(missing_data_end_date,
                                                               format='%Y-%m-%d')
            pred_df = self.get_predictions(missing_data_start_date_str,
                                           missing_data_end_date_str,
                                           prior_ips_df)

            for geo in geos:
                geo_df = pred_df[pred_df['GeoID'] == geo].sort_values(by='Date')
                pred_cases_arr = np.array(geo_df[PRED_CASES_COL])
                past_cases[geo] = np.append(past_cases[geo], pred_cases_arr)
        elif self.verbose:
            print("No missing data.")

        # # Gather values for scaling network output
        # ip_max_values_arr = np.array([NPI_MAX_VALUES[ip] for ip in NPI_COLUMNS])

        # Load IP costs to condition prescriptions
        geo_costs = {}
        for geo in geos:
            costs = cost_df[cost_df['GeoID'] == geo]
            cost_arr = np.array(costs[NPI_COLUMNS])[0]
            geo_costs[geo] = cost_arr

        # Set up dictionary for keeping track of prescription
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in sorted(NPI_MAX_VALUES.keys()):
            df_dict[ip_col] = []

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Generate prescriptions iteratively, feeding resulting
        # predictions from the predictor back into the prescriptor.
        current_date = start_date
        while current_date <= end_date:

            # Get prescription for all regions
            for geo in geos:
                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; many other approaches are possible.
                X_cases = np.log(eval_past_cases[geo][-1] + 1)
                X_ips = eval_past_ips[geo][-1]
                X_costs = geo_costs[geo]
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten(),
                                    X_costs])


                # Observe context specific to geo
                self.bandit.observe(X)

                # Get prescriptions for that day
                prescribed_ips = self.bandit.act()

                # Add it to prescription dictionary
                country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
                if region_name == 'nan':
                    region_name = np.nan
                if current_date > end_date:
                    break
                date_str = current_date.strftime("%Y-%m-%d")
                df_dict['CountryName'].append(country_name)
                df_dict['RegionName'].append(region_name)
                df_dict['Date'].append(date_str)
                for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
                    df_dict[ip_col].append(prescribed_ip)                
                
            # Create dataframe from prescriptions
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pred_df = self.get_predictions(start_date_str, date_str, pres_df)

            # Update past data with new days of prescriptions and predictions
            pres_df = base.add_geo_id(pres_df)
            pred_df = base.add_geo_id(pred_df)
            if current_date > end_date:
                break
            date_str = current_date.strftime("%Y-%m-%d")
            new_pres_df = pres_df[pres_df['Date'] == date_str]
            new_pred_df = pred_df[pred_df['Date'] == date_str]
            for geo in geos:
                geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                # Append array of prescriptions
                pres_arr = np.array([geo_pres[ip_col].values[0] for
                                        ip_col in NPI_COLUMNS]).reshape(1,-1)
                eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                # It is possible that the predictor does not return values for some regions.
                # To make sure we generate full prescriptions, this script continues anyway.
                # This should not happen, but is included here for robustness.
                if len(geo_pred) != 0:
                    eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                        geo_pred[PRED_CASES_COL].values[0])

            # Move on to next date
            current_date += pd.DateOffset(days=1)
        
        prescription_df = pres_df
        # prescription_df= pres_df.drop(columns='GeoID')
        prescription_df['PrescriptionIndex'] = 1
        return prescription_df


if __name__ == '__main__':
    prescriptor = Bandit(seed=42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'prescriptions')
    ofile_path = os.path.abspath(os.path.join(output_dir, 'bandit_evaluate.csv'))
    print(ofile_path)
    print()
    prescriptor.evaluate(output_file_path=ofile_path)
