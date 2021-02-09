import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime

from ongoing.prescriptors.base import BasePrescriptor, NPI_COLUMNS, NPI_MAX_VALUES
import ongoing.prescriptors.base as base

ALPHA = 0.8
NABLA = 0.997715
CONTEXT_DIM = 3
ITER = 50
EVAL_COUNTRIES = 1
EVAL_START_DATE = '2020-11-01'
EVAL_END_DATE = '2020-12-31'
WEIGHTS = np.linspace(0.1, 0.9, num=10)
ACTION_DURATION = 14

class BanditV2(BasePrescriptor):
    def __init__(self, eval_start_date=EVAL_START_DATE, eval_end_date=EVAL_END_DATE, action_duration=ACTION_DURATION,
                 alpha=ALPHA, nabla=NABLA, weights=WEIGHTS, nb_iter=ITER, nb_eval_countries=EVAL_COUNTRIES, hist_df=None, model_fname=None):
        super().__init__()

        self.eval_start_date = pd.to_datetime(eval_start_date, format='%Y-%m-%d')
        self.eval_end_date = pd.to_datetime(eval_end_date, format='%Y-%m-%d')
        self.action_duration = action_duration
        self.alpha = alpha
        self.nabla = nabla
        self.weights = weights
        self.nb_iter = nb_iter
        self.nb_eval_countries = nb_eval_countries
        self.hist_df = hist_df

        if model_fname is not None:
            self.load(model_fname)
        else:
            self.means = [[[np.zeros(CONTEXT_DIM, dtype='float') for _ in range(NPI_MAX_VALUES[i]+1)] for i in NPI_COLUMNS] for _ in range(len(weights))]
            self.covs = [[[np.eye(CONTEXT_DIM, dtype='float') for _ in range(NPI_MAX_VALUES[i]+1)] for i in NPI_COLUMNS] for _ in range(len(weights))]
            self.precisions = [[[np.eye(CONTEXT_DIM, dtype='float') for _ in range(NPI_MAX_VALUES[i]+1)] for i in NPI_COLUMNS] for _ in range(len(weights))]
            self.z = [[[np.zeros(CONTEXT_DIM, dtype='float') for _ in range(NPI_MAX_VALUES[i]+1)] for i in NPI_COLUMNS] for _ in range(len(weights))]

    @staticmethod
    def reward_fn(weight, case_ratio, stringency):
        return weight*case_ratio + (1-weight)*1/max(0.1, stringency)

    def fit(self, hist_df):
        self.hist_df = hist_df
        eval_geos = list(hist_df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                         ascending=False).head(self.nb_eval_countries).index)
        print('Training on data from', eval_geos)
        all_geos_df = hist_df[hist_df['GeoID'].isin(eval_geos)].copy()

        # reward_hist = []
        # fig = plt.gcf()
        # fig.show()
        # fig.canvas.draw()
        for pidx, weight in enumerate(self.weights):
            print('Training prescriptor {} (weight={:.3f})'.format(pidx, weight))
            for _ in range(self.nb_iter):
                for geo in eval_geos:
                    df = all_geos_df[all_geos_df['GeoID'] == geo]
                    npi_df = pd.DataFrame({
                        'CountryName': df['CountryName'].to_list(),
                        'RegionName': df['RegionName'].to_list(),
                        'Date': df['Date'].to_list(),
                    })
                    for npi in NPI_COLUMNS:
                        npi_df[npi] = df[npi].to_list()
                    country_name = df['CountryName'].to_list()[0]
                    region_name = df['RegionName'].to_list()[0]

                    # reward_avg = 0.
                    # steps = 0
                    one_day_bfr_df = df[df['Date'] == self.eval_start_date - np.timedelta64(self.action_duration, 'D')]
                    two_day_bfr_df = df[df['Date'] == self.eval_start_date - np.timedelta64(2*self.action_duration, 'D')]
                    action_start_date = self.eval_start_date
                    while action_start_date <= self.eval_end_date:
                        print(action_start_date)
                        costs = np.random.uniform(size=len(NPI_COLUMNS))
                        costs = len(NPI_COLUMNS) * costs / costs.sum(axis=0)
                        cases_ctx = float(1e5*(one_day_bfr_df['SmoothNewCases']/one_day_bfr_df['Population']).to_numpy())
                        ratio_ctx = float(two_day_bfr_df['SmoothNewCases'].to_numpy()/max(1., one_day_bfr_df['SmoothNewCases'].to_numpy()))
                        strin_ctx = float(costs.T @ one_day_bfr_df[NPI_COLUMNS].to_numpy().squeeze())
                        context = np.array([cases_ctx, ratio_ctx, strin_ctx])

                        actions = []
                        for i, npi in enumerate(NPI_COLUMNS):
                            samples = []
                            for j in range(NPI_MAX_VALUES[npi]+1):
                                samples.append(np.random.multivariate_normal(self.means[pidx][i][j], self.alpha*self.covs[pidx][i][j]))
                            samples = np.array(samples)
                            utility = samples @ context.T
                            action = np.argmax(utility)
                            actions.append(action)
                        actions = np.array(actions)
                        print('actions:', actions)
                        print()

                        action_end_date = action_start_date + np.timedelta64(self.action_duration-1, 'D')
                        for date in pd.date_range(action_start_date, action_end_date):
                            new_npi_dict = {
                                'CountryName': country_name,
                                'RegionName': region_name,
                                'GeoID': geo,
                                'Date': date.strftime("%Y-%m-%d"),
                            }
                            for i, npi in enumerate(NPI_COLUMNS):
                                new_npi_dict[npi] = actions[i]
                            npi_df = npi_df.append(pd.DataFrame([new_npi_dict])).reset_index(drop=True)

                        pred_df = self.get_predictions(self.eval_start_date.strftime("%Y-%m-%d"), action_end_date.strftime("%Y-%m-%d"), npi_df)

                        new_pred_df = pred_df[pred_df['Date'] == action_end_date]
                        new_pred_df = new_pred_df.rename(columns={'PredictedDailyNewCases': 'SmoothNewCases'})
                        new_pred_df['Population'] = one_day_bfr_df['Population'].to_numpy()
                        for i, npi in enumerate(NPI_COLUMNS):
                            new_pred_df[npi] = npi_df[npi]

                        case_ratio = (one_day_bfr_df['SmoothNewCases'].to_numpy()/max(1., new_pred_df['SmoothNewCases'].to_numpy()))[0]
                        stringency_norm = costs.T @ actions/34.
                        reward = self.reward_fn(weight, case_ratio, stringency_norm)

                        context2d = context[:,np.newaxis]
                        for i, action in enumerate(actions):
                            self.precisions[pidx][i][action] = self.nabla*self.precisions[pidx][i][action] + context2d @ context2d.T
                            self.covs[pidx][i][action] = np.linalg.pinv(self.precisions[pidx][i][action])
                            self.z[pidx][i][action] += reward*context
                            self.means[pidx][i][action] = self.covs[pidx][i][action] @ self.z[pidx][i][action]

                        two_day_bfr_df = one_day_bfr_df
                        one_day_bfr_df = new_pred_df
                        action_start_date += np.timedelta64(self.action_duration, 'D')


    def prescribe(self, start_date_str, end_date_str, prior_ips_df, cost_df):
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        geos = prior_ips_df['GeoID'].unique()

        # restrict to dates before the start_date
        all_geos_df = self.hist_df[(self.hist_df['Date'] < start_date) & (self.hist_df['GeoID'].isin(geos))].copy()

        # fill missing data, if any
        last_historical_data_date_str = all_geos_df['Date'].max()
        last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                                   format='%Y-%m-%d')
        if last_historical_data_date + pd.Timedelta(days=1) < start_date:
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
            pred_df = pred_df.rename(columns={'PredictedDailyNewCases': 'SmoothNewCases'})
            pred_df = pred_df.merge(prior_ips_df, on=['RegionName', 'CountryName', 'GeoID', 'Date'], how='left').reset_index(drop=True)
            all_geos_df = all_geos_df.append(pred_df).reset_index(drop=True)

        costs_dict = {geo: cost_df[cost_df['GeoID'] == geo][NPI_COLUMNS].to_numpy().squeeze() for geo in geos}
        df_dict = {geo: all_geos_df[all_geos_df['GeoID'] == geo] for geo in geos}
        all_pres_dfs = []
        for pidx in range(len(self.weights)):
            print('Generating prescription {} ...'.format(pidx))

            one_step_bfr_df_dict = {geo: df_dict[geo][df_dict[geo]['Date'] == start_date - np.timedelta64(self.action_duration, 'D')] for geo in geos}
            two_step_bfr_df_dict = {geo: df_dict[geo][df_dict[geo]['Date'] == start_date - np.timedelta64(2*self.action_duration, 'D')] for geo in geos}
            npi_dict = {'CountryName': [], 'RegionName': [], 'GeoID': [], 'Date': []}
            for npi in NPI_COLUMNS:
                npi_dict[npi] = []
            pres_df = pd.DataFrame()
            actions_dict = {}
            action_start_date = start_date
            while action_start_date <= end_date:
                action_end_date = action_start_date + np.timedelta64(self.action_duration-1, 'D')

                # generate prescriptions for all geos
                for geo in geos:
                    one_step_bfr_df = one_step_bfr_df_dict[geo]
                    two_step_bfr_df = two_step_bfr_df_dict[geo]
                    costs = costs_dict[geo]
                    country_name = one_step_bfr_df['CountryName'].to_list()[0]
                    region_name = one_step_bfr_df['RegionName'].to_list()[0]

                    cases_ctx = float(1e5*(one_step_bfr_df['SmoothNewCases']/one_step_bfr_df['Population']).to_numpy())
                    ratio_ctx = float(two_step_bfr_df['SmoothNewCases'].to_numpy()/max(1., one_step_bfr_df['SmoothNewCases'].to_numpy()))
                    strin_ctx = float(costs.T @ one_step_bfr_df[NPI_COLUMNS].to_numpy().squeeze())
                    context = np.array([cases_ctx, ratio_ctx, strin_ctx])

                    actions = []
                    for i, npi in enumerate(NPI_COLUMNS):
                        samples = []
                        for j in range(NPI_MAX_VALUES[npi]+1):
                            samples.append(np.random.multivariate_normal(self.means[pidx][i][j], ALPHA*self.covs[pidx][i][j]))
                        samples = np.array(samples)
                        # samples = np.array(self.means[pidx][i])

                        utility = samples @ context.T
                        actions.append(np.argmax(utility))
                    actions = np.array(actions)
                    actions_dict[geo] = actions

                    for date in pd.date_range(action_start_date, action_end_date):
                        npi_dict['CountryName'].append(country_name)
                        npi_dict['RegionName'].append(region_name)
                        npi_dict['GeoID'].append(geo)
                        npi_dict['Date'].append(date)
                        for i, npi in enumerate(NPI_COLUMNS):
                            npi_dict[npi].append(actions[i])

                pres_df = pd.DataFrame(npi_dict)
                if action_end_date < end_date:
                    # generate predictions for all geos
                    pred_df = self.get_predictions(start_date_str, action_end_date.strftime("%Y-%m-%d"), pres_df)
                    pred_df = base.add_geo_id(pred_df)

                    new_pred_df = pred_df[pred_df['Date'] == action_end_date]
                    new_pred_df = new_pred_df.rename(columns={'PredictedDailyNewCases': 'SmoothNewCases'})
                    for geo in geos:
                        geo_idx = new_pred_df[new_pred_df['GeoID']==geo].index
                        for i, npi in enumerate(NPI_COLUMNS):
                            new_pred_df.loc[geo_idx, npi] = actions_dict[geo][i]
                        new_pred_df.loc[geo_idx,'Population'] = float(one_step_bfr_df_dict[geo]['Population'].to_numpy())
                        two_step_bfr_df_dict[geo] = one_step_bfr_df_dict[geo]
                        one_step_bfr_df_dict[geo] = new_pred_df.loc[geo_idx]

                action_start_date += np.timedelta64(self.action_duration, 'D')

            pres_df['PrescriptionIndex'] = pidx
            all_pres_dfs.append(pres_df.drop(columns='GeoID'))

        return pd.concat(all_pres_dfs)

    def get_predictions(self, start_date_str, end_date_str, pres_df):
        start_date = pd.to_datetime(start_date_str)
        geos = pres_df['GeoID'].unique()
        last_known_date = self.predictor.df[self.predictor.df['GeoID'].isin(geos)]['Date'].max()
        if last_known_date < pd.to_datetime(self.hist_df['Date'].min()) - np.timedelta64(1, 'D'):
            # append prior NPIs to the prescripted ones because the predictor will need them
            prior_ips_df = self.hist_df[(self.hist_df['Date'] > last_known_date) & (self.hist_df['Date'] < start_date) & (self.hist_df['GeoID'].isin(geos))]
            prior_ips_df = prior_ips_df[pres_df.columns()]
            ips_df = pres_df.append(prior_ips_df)
        else:
            ips_df = pres_df

        # generate the predictions
        pred_df = self.predictor.predict(start_date_str, end_date_str, ips_df)

        return pred_df

    def save(self, fname):
        params = {'means': self.means, 'covariances': self.covs, 'precisions': self.precisions, 'z': self.z}
        with open(fname, 'wb') as f:
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        self.means = params['means']
        self.covs = params['covariances']
        self.precisions = params['precisions']
        self.z = params['z']

if __name__ == '__main__':
    hist_df, _, _ = base.gen_test_config(**base.TEST_CONFIGS[0][1])
    prescriptor = BanditV2(hist_df=hist_df, model_fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bandit_params.pkl'))
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'prescriptions')
    ofile_path = os.path.abspath(os.path.join(output_dir, 'bandit_v2_evaluate.csv'))
    prescriptor.evaluate(output_file_path=ofile_path, fit=False)
