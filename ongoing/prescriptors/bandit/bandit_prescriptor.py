import numpy as np
import pandas as pd
import os
from copy import deepcopy
import datetime
import pickle
import time
import copy

os.system('export PYTHONPATH="$(pwd):$PYTHONPATH"')
from ongoing.prescriptors.base import BasePrescriptor, PRED_CASES_COL, CASES_COL, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base
from bandit import CCTSB

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_PRED_FILE_NAME = os.path.join(ROOT_DIR, 'tmp_predictions_for_prescriptions', 'preds.csv')
TMP_PRESCRIPTION_FILE = os.path.join(ROOT_DIR, 'tmp_prescription.csv')
MODEL_FILE = os.path.join(ROOT_DIR, 'bandits.pkl')

# Number of iterations of training for the bandit.
# Each iteration presents the bandit with a new context.
# Each iteration trains the bandit for the entire prediction window.
NB_ITERATIONS = 2
EXPLORE_ITERATIONS = 1
CHOICE = 'fixed'
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
# EVAL_START_DATE = '2020-08-01'
# EVAL_END_DATE = '2020-08-02'

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 10

# OBJECTIVE_WEIGHTS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]
OBJECTIVE_WEIGHTS = [0.5, 1.0]

LOAD = True
class Bandit(BasePrescriptor):
    def __init__(self,
                 seed=base.SEED,
                #  eval_start_date=EVAL_START_DATE,
                #  eval_end_date=EVAL_END_DATE,
                #  nb_eval_countries=NB_EVAL_COUNTRIES,
                 nb_prescriptions=NB_PRESCRIPTIONS,
                #  nb_lookback_days=NB_LOOKBACK_DAYS,
                 hist_df=None,
                 start_date=None,
                 end_date=None,
                 verbose=True,
                 load=True):

        super().__init__(seed=seed)
        # self.eval_start_date = pd.to_datetime(eval_start_date, format='%Y-%m-%d')
        # self.eval_end_date = pd.to_datetime(eval_end_date, format='%Y-%m-%d')
        self.eval_start_date = None
        self.eval_end_date = None
        self.load = load
        # self.nb_eval_countries = nb_eval_countries
        # self.nb_lookback_days = nb_lookback_days
        self.nb_prescriptions = nb_prescriptions
        # self.action_duration = action_duration
        # self.config_file = config_file
        # self.prescriptors_file = prescriptors_file
        self.hist_df = hist_df
        self.verbose = verbose
        self.bandits = {}
        self.load = load

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

        # print('start_date_str :', start_date_str)
        # print('ips_df : ', ips_df)
        # generate the predictions
        pred_df = self.predictor.predict(start_date_str, end_date_str, ips_df)

        return pred_df


    def fit(self, hist_df=None):
        if hist_df is not None:
            self.hist_df = hist_df
        if self.load == True:
            print('loading bandit')
            with open(MODEL_FILE, 'rb') as f:
                self.bandits = pickle.load(f)
            return
        

        # eval_geos = self.choose_eval_geos()
        # eval_geos = list(self.hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
        #                  ascending=False).index)
        eval_geos = ['Canada']

        if self.verbose:
            print("Bandit will be evaluated on the following geos:", eval_geos)

        past_cases, past_ips = self.prep_past_ips_cases(eval_geos)

        start_date = '2021-01-01'
        end_date = '2021-02-01'

        self.eval_start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        self.eval_end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
        # Compute prescribed stringency incrementally
        stringency = {date : {geo: 0. for geo in eval_geos}
            for date in pd.date_range(self.eval_start_date, self.eval_end_date)}

        geo_costs = self.prep_geo_costs(eval_geos) #dummy call to get size
        context_size = len(next(iter(geo_costs.values())))

        # predictor_df_bkp = self.predictor.df.copy()

        for weight in OBJECTIVE_WEIGHTS:
            # self.bandits is a dict of dicts [weight][geo]
            self.bandits[weight] = {}

            # Initialize a bandit for each weight and geo
            for geo in eval_geos:
                self.bandits[weight][geo] = CCTSB(
                    N=[i + 1 for i in NPI_MAX_VALUES.values()], #assumed max val + zero
                    K=len(NPI_MAX_VALUES),
                    C=context_size,
                    alpha_p=1,
                    nabla_p=1,
                    w = weight,
                    choice=CHOICE)
            rewards = []
            for t in range(NB_ITERATIONS):

                predictor_df_bkp = self.predictor.df.copy()

                eval_past_cases = deepcopy(past_cases)
                eval_past_ips = deepcopy(past_ips)
                # forget all data before eval_start_date
                self.predictor.df = self.predictor.df[
                    (self.predictor.df['Date'] < self.eval_start_date)
                    & (self.predictor.df['GeoID'].isin(eval_geos))]


                #prepare costs for all geos
                df_dict = self.prep_prescription_dict()

                for date in pd.date_range(self.eval_start_date, self.eval_end_date):
                    geo_costs = self.prep_geo_costs(eval_geos)

                    date_str = date.strftime("%Y-%m-%d")

                    # Make prescriptions one day at a time, feeding resulting
                    # predictions from the predictor back into the prescriptor.
                    for geo in eval_geos:

                        bandit = self.bandits[weight][geo]
                        if geo == eval_geos[0]:
                            bandit.verbose = True
                        X_costs = geo_costs[geo]
                        bandit.observe(X_costs)
                        if t < EXPLORE_ITERATIONS:
                            bandit.choice = 'random'
                        else:
                            bandit.choice = 'fixed'
                        prescribed_ips = bandit.act() # gets prescriptions

                        # print(prescribed_ips)
                        # Add it to prescription dictionary
                        self.add_pres_to_dict(df_dict, date_str, geo, prescribed_ips)

                        # Calculate stringency
                        stringency[date][geo] = self.calc_stringency(X_costs,
                                                                    prescribed_ips)

                    # Once predictions are made for all geos,
                    # Create dataframe from prescriptions
                    pres_df = pd.DataFrame(df_dict)
                    pres_df = base.add_geo_id(pres_df)



                    # Make batch predictions with prescriptions for all geos
                    pred_df = self.get_predictions(date_str, date_str, pres_df)
                    pred_df = base.add_geo_id(pred_df)
                    new_pres_df = pres_df[pres_df['Date'] == date_str]
                    new_pred_df = pred_df[pred_df['Date'] == date_str]

                    # update each geo's bandit based on predictions
                    for geo in eval_geos:

                        bandit = self.bandits[weight][geo]
                        geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                        geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

                        # calculate reward before appending to df
                        reward = eval_past_cases[geo][-1] / (np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]]))
                        # reward = 1 if eval_past_cases[geo][-1] > (np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]])) else 0
                        # print('reward : ', reward)
                        # print('eval_past_cases[geo][-1] : ', eval_past_cases[geo][-1])
                        # print('(np.max([0.1,geo_pred[PRED_CASES_COL].values[0]])) : ', np.max([0.1,geo_pred[PRED_CASES_COL].values[0]]))
                        # print('[0.1,geo_pred[PRED_CASES_COL].values[0]] : ', [0.1,geo_pred[PRED_CASES_COL].values[0][0]])

                        if geo == eval_geos[0]:
                            bandit.update(r_past=eval_past_cases[geo][-1], r_present=(np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]])), s=stringency[date][geo], w=weight, verbose=True)
                        else:
                            bandit.update(r_past=eval_past_cases[geo][-1], r_present=(np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]])), s=stringency[date][geo], w=weight)

                        # Append predictions and prescriptions to past data
                        self.append_pres_pred_to_df(eval_past_cases, eval_past_ips, geo,
                                            geo_pres, geo_pred)


                    new_pred_df = new_pred_df.merge(new_pres_df, on=['CountryName', 'RegionName', 'GeoID', 'Date'], how='left')
                    new_pred_df = new_pred_df.rename(columns={'PredictedDailyNewCases': 'NewCases'})
                    new_pred_df['ConfirmedCases'] = np.nan
                    new_pred_df['Population'] = np.nan

                    # for geo in eval_geos:  # INEFFICIENT: there should be a way to avoid this loop using pandas functions
                    #     temp_df = self.predictor.df[(self.predictor.df['GeoID'] == geo) & (self.predictor.df['Date'] == date - np.timedelta64(1,'D'))]
                    #     new_cases = new_pred_df[new_pred_df['GeoID'] == geo]['NewCases'].to_numpy()
                    #     new_pred_df.loc[new_pred_df['GeoID'] == geo, 'ConfirmedCases'] = np.cumsum(new_cases) + temp_df['ConfirmedCases'].to_numpy()
                    #     new_pred_df.loc[new_pred_df['GeoID'] == geo, 'Population'] = temp_df.iloc[0]['Population']
                    #     if geo == eval_geos[0]:
                    #         print('temp_df : ', temp_df)
                    #         print('New Cases : ', new_pred_df[new_pred_df['GeoID'] == geo]['NewCases'])
                    #         print('New pred df : ', new_pred_df)
                    #         print('self.predictor.df : ', self.predictor.df)

                    temp_df = self.predictor.df[self.predictor.df['Date'] == date - np.timedelta64(1,'D')]
                    new_cases = new_pred_df['NewCases']
                    new_pred_df.loc['ConfirmedCases'] = new_cases + temp_df['ConfirmedCases']
                    new_pred_df.loc['Population'] = temp_df['Population']
                    self.predictor.df = self.predictor.df.append(new_pred_df, ignore_index=True)

                    # we need to compute the PredictionRatio since this is used as input for the predictor
                    # INNEFICIENT: there should be a way to compute these quantities only for the new dates
                    self.predictor.df['SmoothNewCases'] = self.predictor.df.groupby('GeoID')['NewCases'].rolling(
                        7, center=False).mean().fillna(0).reset_index(0, drop=True)
                    self.predictor.df['CaseRatio'] = self.predictor.df.groupby('GeoID').SmoothNewCases.pct_change(
                        ).fillna(0).replace(np.inf, 0) + 1
                    self.predictor.df['ProportionInfected'] = self.predictor.df['ConfirmedCases'] / self.predictor.df['Population']
                    self.predictor.df['PredictionRatio'] = self.predictor.df['CaseRatio'] / (1 - self.predictor.df['ProportionInfected'])

                print('Iteration ' + str(t) + ' done.')

                pres_df = pd.DataFrame(df_dict)

                pres_df.to_csv('inspection.csv')
                for geo in eval_geos:
                    self.bandits[weight][geo].clear_update_hist()
                # restore the predictor historical data after evaluating the genome
                self.predictor.df = predictor_df_bkp

                mean_rewards = np.mean(self.bandits[weight]['Canada'].rewards, axis=0)
                self.bandits[weight]['Canada'].rewards = []

                rewards.append(mean_rewards[0:4])

        #     print('Weight ' + str(weight) + ' done.')
        #     np.savetxt('rewards_cumulative_' + str(weight) + '_' + str(self.bandits[weight]['Canada'].choice),
        #                rewards,
        #                fmt='%1.10f')
        #     indiv_rewards = self.bandits[weight]['Canada'].rewards
        #     np.savetxt('rewards_' + str(weight) + '_' + CHOICE,
        #                indiv_rewards,
        #                fmt='%1.10f')

        # with open('bandits.pkl', 'wb') as f:
        #     pickle.dump(self.bandits, f)
        
        return


    def prescribe(self,
                  start_date_str,
                  end_date_str,
                  prior_ips_df,
                  cost_df):

        if self.load == True:
            with open(MODEL_FILE, 'rb') as f:
                self.bandits = pickle.load(f)

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


        # Generate prescriptions iteratively, feeding resulting
        # predictions from the predictor back into the prescriptor.
        prescription_dfs = []

        for weight in OBJECTIVE_WEIGHTS:
            for geo in geos:
                start_time = time.time()
                self.bandits[weight][geo] = copy.deepcopy(self.bandits[weight]['Canada'])
                print(time.time()-start_time)

        for idx, weight in enumerate(OBJECTIVE_WEIGHTS):
            current_date = start_date
            predictor_df_bkp = self.predictor.df.copy()

            # Set initial data
            eval_past_cases = deepcopy(past_cases)
            eval_past_ips = deepcopy(past_ips)

            # Set up dictionary for keeping track of prescription
            df_dict = self.prep_prescription_dict()
            # forget all data after start_date
            self.predictor.df = self.predictor.df[
                (self.predictor.df['Date'] < start_date)
                & (self.predictor.df['GeoID'].isin(geos))]

            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")

                # Get prescription for all regions
                for geo in geos:
                    # only Canada bandit is trained
                    bandit = self.bandits[weight]['Canada']
                    X_costs = geo_costs[geo]
                    bandit.observe(X_costs)
                    prescribed_ips = bandit.act()

                    # Add it to prescription dictionary
                    if current_date > end_date:
                        break
                    self.add_pres_to_dict(df_dict, date_str, geo, prescribed_ips)

                    # Calculate stringency
                    eval_stringency[current_date][geo] = self.calc_stringency(X_costs,
                                                                prescribed_ips)

                # Create dataframe from prescriptions
                pres_df = pd.DataFrame(df_dict)
                pres_df = base.add_geo_id(pres_df)
                # Make prediction given prescription for all countries
                pred_df = self.get_predictions(date_str, date_str, pres_df)
                pred_df = base.add_geo_id(pred_df)
                new_pres_df = pres_df[pres_df['Date'] == date_str]
                new_pred_df = pred_df[pred_df['Date'] == date_str]

                # make sure we haven't passed the end date
                if current_date > end_date:
                    break

                for geo in geos:
                    bandit = self.bandits[weight]['Canada']
                    geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                    geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

                    # calculate reward before appending to df
                    reward = eval_past_cases[geo][-1] / (np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]]))

                    # print(geo, reward)

                    self.append_pres_pred_to_df(eval_past_cases, eval_past_ips,
                                                geo, geo_pres, geo_pred)

                    bandit.update(eval_past_cases[geo][-1], (np.max([0.1,geo_pred[PRED_CASES_COL].values[0][0]])), eval_stringency[current_date][geo], weight)
                    print('geo ' + str(geo) + ' done.')
                # Move on to next date

                new_pred_df = new_pred_df.merge(new_pres_df, on=['CountryName', 'RegionName', 'GeoID', 'Date'], how='left')
                new_pred_df = new_pred_df.rename(columns={'PredictedDailyNewCases': 'NewCases'})
                new_pred_df['ConfirmedCases'] = np.nan
                new_pred_df['Population'] = np.nan
                for geo in geos:  # INNEFICIENT: there should be a way to avoid this loop using pandas functions
                    temp_df = self.predictor.df[(self.predictor.df['GeoID'] == geo) & (self.predictor.df['Date'] == current_date - np.timedelta64(1,'D'))]
                    new_cases = new_pred_df[new_pred_df['GeoID'] == geo]['NewCases'].to_numpy()
                    # print('New Cases : ', np.cumsum(new_cases))
                    # print('Confirmed Cases : ', temp_df['ConfirmedCases'])
                    new_pred_df.loc[new_pred_df['GeoID'] == geo, 'ConfirmedCases'] = np.cumsum(new_cases) + temp_df['ConfirmedCases'].to_numpy()
                    new_pred_df.loc[new_pred_df['GeoID'] == geo, 'Population'] = temp_df.iloc[0]['Population']
                self.predictor.df = self.predictor.df.append(new_pred_df, ignore_index=True)

                # we need to compute the PredictionRatio since this is used as input for the predictor
                # INNEFICIENT: there should be a way to compute these quantities only for the new dates
                self.predictor.df['SmoothNewCases'] = self.predictor.df.groupby('GeoID')['NewCases'].rolling(
                    7, center=False).mean().fillna(0).reset_index(0, drop=True)
                self.predictor.df['CaseRatio'] = self.predictor.df.groupby('GeoID').SmoothNewCases.pct_change(
                    ).fillna(0).replace(np.inf, 0) + 1
                self.predictor.df['ProportionInfected'] = self.predictor.df['ConfirmedCases'] / self.predictor.df['Population']
                self.predictor.df['PredictionRatio'] = self.predictor.df['CaseRatio'] / (1 - self.predictor.df['ProportionInfected'])

                current_date += pd.DateOffset(days=1)

                print('day ' + str(current_date) + ' done.')

            pres_df['PrescriptionIndex'] = idx
            prescription_dfs.append(pres_df)

            pres_df = pd.DataFrame(df_dict)


            for geo in geos:
                self.bandits[weight][geo].clear_update_hist()
            self.predictor.df = predictor_df_bkp

            print('Weight ' + str(weight) + ' done.')

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
        stringency = np.dot(X_costs,np.array(list(prescribed_ips.values())))
        return stringency


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
            df_dict[ip_col].append(prescribed_ips[prescribed_ip])


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
            cost_arr = costs[NPI_COLUMNS].values[0]
            geo_costs[geo] = cost_arr
        return geo_costs


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

        return past_cases, past_ips


    def choose_eval_geos(self):
        """
        As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
        so far as the geos for evaluation.

        Input: None. Uses self.hist, which is part of __init__

        output: a list of GeoIDs.
        """
        assert self.nb_eval_countries, "need to uncomment nb_eval_countries"
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
            missing_data_start_date_str = datetime.datetime.strftime(missing_data_start_date,
                                                               format='%Y-%m-%d')
            missing_data_end_date = start_date - pd.Timedelta(days=1)
            missing_data_end_date_str = datetime.datetime.strftime(missing_data_end_date,
                                                               format='%Y-%m-%d')
            pred_df = self.get_predictions(missing_data_start_date_str,
                                           missing_data_end_date_str,
                                           prior_ips_df)
            pred_df = base.add_geo_id(pred_df)
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
