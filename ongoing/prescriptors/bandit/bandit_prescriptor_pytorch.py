import numpy as np
import pandas as pd
import os
from copy import deepcopy
import datetime
import torch

from ongoing.prescriptors.base import BasePrescriptor, PRED_CASES_COL, CASES_COL, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base
from bandit_tensor import CCTSB

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_PRED_FILE_NAME = os.path.join(ROOT_DIR, 'tmp_predictions_for_prescriptions', 'preds.csv')
TMP_PRESCRIPTION_FILE = os.path.join(ROOT_DIR, 'tmp_prescription.csv')

# Number of iterations of training for the bandit. 
# Each iteration presents the bandit with a new context.
# Each iteration trains the bandit for the entire prediction window.
NB_ITERATIONS = 10

# Number of days the prescriptors will look at in the past.
# Larger values here may make convergence slower, but give
# prescriptors more context. The number of inputs of each neat
# network will be NB_LOOKBACK_DAYS * (NPI_COLUMNS + 1) + NPI_COLUMNS.
# The '1' is for previous case data, and the final NPI_COLUMNS
# is for IP cost information.
# NB_LOOKBACK_DAYS = 14

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
# NB_EVAL_COUNTRIES = 10

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 10

OBJECTIVE_WEIGHTS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]

class Bandit(BasePrescriptor):
    def __init__(self,
                 seed=base.SEED,
                #  eval_start_date=EVAL_START_DATE,
                #  eval_end_date=EVAL_END_DATE,
                #  nb_eval_countries=NB_EVAL_COUNTRIES,
                 nb_prescriptions=NB_PRESCRIPTIONS,
                #  nb_lookback_days=NB_LOOKBACK_DAYS,
                 hist_df=None,
                 verbose=True):

        super().__init__(seed=seed)

        # self.eval_start_date = pd.to_datetime(eval_start_date, format='%Y-%m-%d')
        # self.eval_end_date = pd.to_datetime(eval_end_date, format='%Y-%m-%d')
        # self.nb_eval_countries = nb_eval_countries
        # self.nb_lookback_days = nb_lookback_days
        self.eval_start_date = None
        self.eval_end_date = None
        self.nb_prescriptions = nb_prescriptions
        self.hist_df = hist_df
        self.verbose = verbose
        self.bandits = {}


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

        # generate the predictions
        pred_df = self.predictor.predict(start_date_str, end_date_str, ips_df)

        return pred_df


    def fit(self, hist_df=None, start_date=None, end_date=None):

        if hist_df is not None:
            self.hist_df = hist_df
        
        if start_date is None:
            start_date = self.hist_df['Date'].min()
        
        if end_date is None:
            end_date = self.hist_df['Date'].max()
        self.eval_start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        self.eval_end_date = pd.to_datetime(end_date, format='%Y-%m-%d')

        # eval_geos = self.choose_eval_geos()
        eval_geos = list(self.hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                         ascending=False).index)

        if self.verbose:
            print("Bandit will be evaluated on the following geos:", eval_geos)


        # TODO: add past cases and IPS
        # will be tensor days x geo x weights x k 
        prescriptions = None

        # weight x days x geo
        predictions = None

        # days x geos x weights
        stringency = None
        
        geo_costs = self.prep_geo_costs(eval_geos) #dummy call to get size
        context_size = len(next(iter(geo_costs.values())))

        predictor_df_bkp = self.predictor.df.copy()

        self.bandit = CCTSB(
            N=[i + 1 for i in NPI_MAX_VALUES.values()], #assumed max val + zero
            K=len(NPI_MAX_VALUES),
            C=context_size,
            geos=eval_geos,
            alpha_p=0.99,
            nabla_p=0.8,
            weights = OBJECTIVE_WEIGHTS)

        for t in range(NB_ITERATIONS):
                
            #prepare costs for all geos
            # returns tensor geos x context
            geo_costs = self.prep_geo_costs_tensor(eval_geos)

            for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                date_str = date.strftime("%Y-%m-%d")
            
                # Make prescriptions one day at a time, feeding resulting
                # predictions from the predictor back into the prescriptor.
                
                # geo_costs is a tensor (geos x context)
                X_costs = geo_costs
                self.bandit.observe(X_costs)

                # geo x weights x k
                prescribed_ips = self.bandit.act() # gets prescriptions

                # Add it to prescription tensor
                # prescriptions is a tensor  days x geo x weights x k
                if prescriptions is not None:
                    prescriptions = torch.cat(prescriptions, prescribed_ips.unsqueeze(0))
                else: 
                    prescriptions = prescribed_ips.unsqueeze(0)
                
                # curent_stringency is geo x weights
                current_stringency = torch.einsum("gk, gwk->gw",
                                                  X_costs,
                                                  prescribed_ips.double())
                if stringency is not None:
                    stringency = torch.cat(stringency,
                                           current_stringency.unsqueeze(0))
                else: 
                    stringency = current_stringency.unsqueeze(0)

                current_daterange = pd.date_range(self.eval_start_date,
                                                  date)
                
                predictions = None
                # piece of code that requires vectorizing
                for weight in range(prescriptions.size()[2]):

                    # Not sure in what order it flattens
                    # days x geo x k
                    pres_tensor = prescriptions[:, :, weight, :]
                    # (days, geo) x k
                    pres_tensor = pres_tensor.flatten(0,1)

                    # cartesian product of geos and days
                    geo_and_dates = [(date, geo) for geo in eval_geos \
                        for date in current_daterange]
                    
                    # give tuples containing all the info for the dataframe
                    # (date, geo, *interventions)
                    df_tuples = [(*geodate, *interv) for geodate, interv in \
                        zip(geo_and_dates, pres_tensor.numpy())]

                    # Once predictions are made for all geos, 
                    # Create dataframe from prescriptions
                    # Dataframe should have columns: 
                    # CountryName, RegionName, Date, and 1 column per NPI
                    pres_df = pd.DataFrame(df_tuples,
                                           columns = ['Date', 'GeoID'] + NPI_COLUMNS)
                    countries_regions = pres_df.GeoID.str.split(' / ', expand=True).fillna(value=np.nan)
                    country_names = countries_regions[0]
                    region_names = countries_regions[1]
                    region_names = region_names.where(region_names != 'nan')
                    pres_df['CountryName'] = country_names
                    pres_df['RegionName'] = region_names
                    pres_df = pres_df.drop(columns='GeoID')
                    # Make batch predictions with prescriptions for all geos
                    pred_df = self.get_predictions(
                        self.eval_start_date.strftime("%Y-%m-%d"), date_str, pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    
                    # turn predictions into tensor days x geo
                    pred_tensor = pred_df[['GeoID', 'Date'] + PRED_CASES_COL]
                    pred_tensor = pd.pivot(pred_tensor,
                                           index = 'Date',
                                           columns = 'GeoID',
                                           values = PRED_CASES_COL[0])
                    pred_tensor = torch.tensor(pred_tensor.values)

                    if predictions is not None:
                        # predictions will be weight x days x geo
                        predictions = torch.cat((predictions, pred_tensor.unsqueeze(0)))
                    
                    else: 
                        predictions = pred_tensor.unsqueeze(0)
                
                # geo x weights x k
                # new_pres = prescriptions[-1, :, :, :]
                # weights x geo
                new_pred = predictions[:, -1, :]

                try: # will fail on first day
                    prev_pred = predictions[:, -2, :]
                except:
                    prev_pred = torch.zeros(new_pred.size())

                # calculate reward before appending to df                    
                reward = torch.div(prev_pred, new_pred)

                #update bandit
                self.bandit.update(reward,
                                   current_stringency,
                                   w=OBJECTIVE_WEIGHTS)

                print('Day ' + str(date) + ' done.')

            print('Iteration ' + str(t) + ' done.')

            for geo in eval_geos:
                self.bandits[weight][geo].clear_update_hist()
        
        print('Weight ' + str(weight) + ' done.')

        return


    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips_df,
                  cost_df):

        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        geos = prior_ips_df['GeoID'].unique()

        eval_stringency = {date : {geo: 0. for geo in geos} 
            for date in pd.date_range(start_date, end_date)}

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

        self.fill_missing_data(prior_ips_df, start_date, geos, df, past_cases)

        # Load IP costs to condition prescriptions
        geo_costs = self.prep_geo_costs(eval_geos=geos,
            costs_provided=True, cost_df=cost_df)


        # Set up dictionary for keeping track of prescription
        df_dict = self.prep_prescription_dict()

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Generate prescriptions iteratively, feeding resulting
        # predictions from the predictor back into the prescriptor.
        prescription_dfs = []
        for idx, weight in enumerate(OBJECTIVE_WEIGHTS):
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")

                # Get prescription for all regions
                for geo in geos:
                    bandit = self.bandits[weight][geo]
                    # Prepare input data. Here we use log to place cases
                    # on a reasonable scale; many other approaches are possible.
                    X_costs = geo_costs[geo]
                    X_costs = torch.tensor

                    # Observe context specific to geo
                    bandit.observe(X_costs)

                    # Get prescriptions for that day
                    # returns a tensor geo x weight x K
                    prescribed_ips = bandit.act()

                    # Add it to prescription dictionary
                    if current_date > end_date:
                        break
                    self.add_pres_to_dict(df_dict, date_str, geo, prescribed_ips)              
                    
                    # Add it to prescription dictionary
                    self.add_pres_to_dict(df_dict, date_str, geo, prescribed_ips)
                    
                    # Calculate stringency
                    eval_stringency[current_date][geo] = self.calc_stringency(X_costs,
                                                                prescribed_ips)

                # Create dataframe from prescriptions
                pres_df = pd.DataFrame(df_dict)
                pres_df = base.add_geo_id(pres_df)

                # Make prediction given prescription for all countries
                pred_df = self.get_predictions(start_date_str, date_str, pres_df)
                pred_df = base.add_geo_id(pred_df)

                # make sure we haven't passed the end date
                if current_date > end_date:
                    break

                for geo in geos:
                    bandit = self.bandits[weight][geo]

                    # Get geo specific data
                    geo_pres = pres_df[pres_df['GeoID'] == geo]
                    geo_pred = pred_df[pred_df['GeoID'] == geo]

                    new_pres = geo_pres[geo_pres['Date'] == date_str]
                    new_pred = geo_pred[geo_pred['Date'] == date_str]

                    # calculate reward before appending to df                    
                    reward = eval_past_cases[geo][-1] / new_pred[PRED_CASES_COL].values[0]

                    self.append_pres_pred_to_df(eval_past_cases, eval_past_ips, 
                                                geo, new_pres, geo_pred)

                    print(geo, reward)

                    #update bandit
                    bandit.update(reward, eval_stringency[current_date][geo], weight)

                # Move on to next date
                current_date += pd.DateOffset(days=1)
                print(current_date)
            
            pres_df['PrescriptionIndex'] = idx + 1
            prescription_dfs.append(pres_df)
        
        prescription_df = pd.concat(prescription_dfs)
        prescription_df = prescription_df.drop(columns='GeoID')

        return prescription_df


    def append_pres_pred_to_df(self, eval_past_cases, eval_past_ips, geo,
                               geo_pres, geo_pred):
        """
        Append prescriptions and predictions to eval_past_cases and
        eval_past_ips.
        These apprend pres and pred will be used the next day in the context
        fed to the bandit.
        """
        # Append prescriptions
        pres_arr = np.array([geo_pres[ip_col].values[0] for 
                             ip_col in NPI_COLUMNS]).reshape(1,-1)
        eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

        # Append predicted cases
        if len(geo_pred) != 0:
            eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                            geo_pred[PRED_CASES_COL].values[0])


    def calc_stringency(self, X_costs, prescribed_ips):
        """
        Calculate stringency. This calculation could include division by
        the number of IPs and/or number of geos, but that would have
        no effect on the ordering of candidate solutions.
        Input:
            - X_costs: 
        """
        return np.dot(X_costs,np.array(list(prescribed_ips.values())))


    def add_pres_to_dict(self, df_dict, date_str, geo, prescribed_ips):
        """
        Add prescribed NPIs to the dict of prescriptions.
        Input: 
            - df_dict: a dict of prescriptions, see prep_prescription_dict();
            - date_str: a string representing the date for which a 
                        prescription was made;
            - geo: a GeoID for which the prescription was made;
            - prescribed_ips: An array indicating the intensity of each
              intervention in the prescription (0 to N).
        Output:
            - None. Appends to each list in df_dict.
        """
        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
        if region_name == 'nan':
            region_name = np.nan
        df_dict['CountryName'].append(country_name)
        df_dict['RegionName'].append(region_name)
        df_dict['Date'].append(date_str)
        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
            df_dict[ip_col].append(prescribed_ip)

    def add_pres_to_dict_tensor(self, df_dict, date_str, geo, prescribed_ips):
        """
        Add prescribed NPIs to the dict of prescriptions.
        Input: 
            - df_dict: a dict of prescriptions, see prep_prescription_dict();
            - date_str: a string representing the date for which a 
                        prescription was made;
            - geo: a GeoID for which the prescription was made;
            - prescribed_ips: An array indicating the intensity of each
              intervention in the prescription (0 to N).
        Output:
            - None. Appends to each list in df_dict.
        """
        country_name, region_name = (geo.split(' / ') + [np.nan])[:2]
        if region_name == 'nan':
            region_name = np.nan
        df_dict['CountryName'].append(country_name)
        df_dict['RegionName'].append(region_name)
        df_dict['Date'].append(date_str)
        for ip_col, prescribed_ip in zip(NPI_COLUMNS, prescribed_ips):
            df_dict[ip_col].append(prescribed_ip)


    def get_context_size(self, eval_past_cases, eval_past_ips, geo_costs):
        """
        Artifact function for when context included past cases and ips.
        DO NOT USE.
        Calculates context size needed for Bandit.
        Each of the inputs' first elements is obtained to get length.
        Context currently includes:
            - geo_costs[geo]: an array of costs for NPIs for a specific GeoID;
            - previous day's cases: the last element of eval_past_cases;
            - previous day's IPS: the last element of eval_past_IPS.
        """
        eval_past_ips_len = len(next(iter(eval_past_ips.values()))[-1]) * NB_LOOKBACK_DAYS
        eval_past_cases_len = len(next(iter(eval_past_cases.values()))[-1]) * NB_LOOKBACK_DAYS
        geo_costs_len = len(next(iter(geo_costs.values())))
        context_size = eval_past_ips_len + eval_past_cases_len + geo_costs_len
        return context_size


    def prep_prescription_dict(self):
        """
        Prepares a dict for prescriptions that will be turned into a df
        fed to the BasePredictor's `get_predictions()`.
        Input: None
        Output: a dict where keys are column names and values are lists.
        """
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in NPI_COLUMNS:
            df_dict[ip_col] = []
        return df_dict


    def prep_geo_costs(self, eval_geos, costs_provided=False, cost_df=None):
        """
        Prepares costs for each intervention (the "weights") for each GeoID.
        Input: eval_geos, a list of GeoIDs for which costs are desired.
        Output: geo_costs, a dict:
            - each key is a GeoID
            - each value is an array of size len(NPI_COLUMNS), so 12 usually,
              which represents the stringency cost associated with each 
              Non-Pharmaceutical Intervention (NPI). These values should sum to
              12 (To be verified).
        """
        if costs_provided == False:
            cost_df = base.generate_costs(self.hist_df, mode='random')
            cost_df = base.add_geo_id(cost_df)
        # Separate costs by geo
        geo_costs = {}
        for geo in eval_geos:
            costs = cost_df[cost_df['GeoID'] == geo]
            cost_arr = np.array(costs[NPI_COLUMNS])[0]
            geo_costs[geo] = cost_arr
        return geo_costs

    def prep_geo_costs_tensor(self, eval_geos, costs_provided=False, cost_df=None):
        """
        Prepares costs for each intervention (the "weights") for each GeoID.
        Input: eval_geos, a list of GeoIDs for which costs are desired.
        Output: geo_costs, a dict:
            - each key is a GeoID
            - each value is an array of size len(NPI_COLUMNS), so 12 usually,
              which represents the stringency cost associated with each 
              Non-Pharmaceutical Intervention (NPI). These values should sum to
              12 (To be verified).
        """
        if costs_provided == False:
            cost_df = base.generate_costs(self.hist_df, mode='random')
            cost_df = base.add_geo_id(cost_df)

        # returns tensor 200 x 12
        return torch.tensor(cost_df[NPI_COLUMNS].values)

    def prep_past_ips_cases(self, eval_geos):
        """
        Separate past cases and past ips data for each eval geo.
        Input: eval_geos, a list of GeoIDs used for evaluation.
        Output: past_cases, past_ips, eval_past_cases, eval_past_ips
            Dictionaries where each key is a GeoID and each value is:
                - an array of past case values, or;
                - an array of past interventions plans (IPs) represented by
                  arrays indicating the intensity of each intervention (0 to N).
        """
        past_cases = {}
        past_ips = {}
        for geo in eval_geos:
            geo_df = self.hist_df[self.hist_df['GeoID'] == geo]
            past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
            past_ips[geo] = np.array(geo_df[NPI_COLUMNS])
        
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        return past_cases, past_ips, eval_past_cases, eval_past_ips


    def choose_eval_geos(self):
        """
        As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
        so far as the geos for evaluation.

        Input: None. Uses self.hist, which is part of __init__

        output: a list of GeoIDs.
        """
        eval_geos = list(self.hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                         ascending=False).head(self.nb_eval_countries).index)
        if self.verbose:
            print("Bandit will be evaluated on the following geos:", eval_geos)
        return eval_geos


    def fill_missing_data(self, prior_ips_df, start_date, geos, df, past_cases):
        """     
        Fill in any missing case data before start_date using predictor given 
        past_ips_df. Note that the following assumes that the df returned by 
        prepare_historical_df() has the same final date for all regions. This
        has been true so far, but relies on it being true for the Oxford data
        csv loaded by prepare_historical_df().
        """
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


if __name__ == '__main__':
    prescriptor = Bandit(seed=42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'prescriptions')
    ofile_path = os.path.abspath(os.path.join(output_dir, 'bandit_evaluate.csv'))
    print(ofile_path)
    print()
    prescriptor.evaluate(output_file_path=ofile_path)
