import argparse
# from ongoing.predictors.econ.econ_utils import *
from pandas._libs.tslibs import NaT
from econ.econ_utils import *
import statsmodels
import statsmodels.api as sm
import os
import pickle

import numpy as np
import pandas as pd

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
NPI_COLS = ['C1_School closing',
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


def econ_predictor(start_date_str, 
                   end_date_str, 
                   DATA_DIR,
                   MODEL_FILE,
                   path_to_hist_ips_file,
                   path_to_future_ips_file,
                   verbose=False):

    # set dates and quarters
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    starting_quarter = start_date + pd.tseries.offsets.QuarterEnd()
    ending_quarter = end_date + pd.tseries.offsets.QuarterEnd()

    # load gdp data
    gdp = load_quarterly_imf(os.path.join(DATA_DIR, 'gdp.xlsx'))
    gdp = prep_gdp(gdp)

    # load NPI data
    hist_ips_df = pd.read_csv(path_to_hist_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)
    for npi_col in NPI_COLS:
        hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    for npi_col in NPI_COLS:
        hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    hist_ips_df = add_geoID(hist_ips_df)
    hist_ips_df = prep_quarterly_npis(hist_ips_df, gdp.index.min())

    # ips_df = hist_ips_df[(hist_ips_df.index >= starting_quarter) & (hist_ips_df.index <= ending_quarter)]

    future_ips_df = pd.read_csv(path_to_future_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    for npi_col in NPI_COLS:
        future_ips_df.update(future_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    for npi_col in NPI_COLS:
        future_ips_df.update(future_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    # future_ips_df = add_geoID(future_ips_df)
    future_ips_df = prep_quarterly_npis(future_ips_df, gdp.index.min())

    # get rid of old index and empty NewCases columns
    future_ips_df = future_ips_df[hist_ips_df.columns]

    combined_ips_df = pd.concat([hist_ips_df, future_ips_df]).fillna(0)
    combined_ips_df = combined_ips_df.loc[combined_ips_df.index.drop_duplicates()]

    # ips_df = combined_ips_df[(combined_ips_df.index >= starting_quarter) & (combined_ips_df.index <= ending_quarter)]
    # keep NPI data for countries that have econ data
    npis_countries = hist_ips_df['GeoID']

    gdp = rename_imf_countries(gdp)

    # remove data for countries with no NPIs and for regions
    bad_countries = set(gdp['CountryName']).difference(set(npis_countries))
    gdp = gdp[~gdp['CountryName'].isin(bad_countries)]

    hist_ips_df = hist_ips_df.rename({'GeoID': 'CountryName'},axis=1)\

    # keep only countries with econ data
    econ_data = gdp.merge(hist_ips_df, how ='right', on=['Date', 'CountryName'])

    econ_countries = econ_data.loc['2015-03-31'][econ_data.loc['2015-03-31']['GDP growth'].notna()]['CountryName']

    econ_data = econ_data[econ_data['CountryName'].isin(econ_countries)]

    # Load econ models, they are country-specific
    try:
        with open(MODEL_FILE, 'rb') as model_file:
            models = pickle.load(model_file)
    except:
        print('no model file')

    # get predictions for each country with econ data
    geo_pred_dfs = []
    old_geo_pred_dfs = []
    
    for g in combined_ips_df.GeoID.unique():
        g_short = g[:-5] #trim __nan
        if verbose:
            print('\nPredicting for', g)

        # get country-specific model
        try:
            model = models[g_short]

        # exit loop if no model for country
        except:
            if verbose:
                print('No model for ' + str(g_short))
            continue
        
        # get old predictions and append
        old_pred = model.get_prediction()
        old_results = old_pred.conf_int()
        old_results['gdp_growth_pred_mean'] = old_pred.predicted_mean
        old_results['gdp_growth_stderr_mean'] = old_pred.se_mean
        old_results['gdp_growth_var_pred_mean'] = old_pred.var_pred_mean
        old_results['GeoID'] = g
        old_results['CountryName'] = g_short
        old_results['RegionName'] = np.nan
        old_results['RegionName'] = old_results['RegionName'].astype(object)
        old_results['Date'] = old_results.index
        old_geo_pred_dfs.append(old_results)

        # get country-specific NPIs
        combined_ips_df_g = combined_ips_df[combined_ips_df.GeoID == g]
        ips_df_g = combined_ips_df_g[(combined_ips_df_g.index > old_pred.predicted_mean.index.max()) & (combined_ips_df_g.index <= ending_quarter)]

        # ips_df_g = ips_df_g.loc[ips_df_g.index.drop_duplicates()].asfreq('Q')
        ips_df_g = ips_df_g[~ips_df_g.index.duplicated(keep='first')]
        # prep exogenous data for prediction
        exog = add_freq(sm.add_constant(ips_df_g.drop('GeoID', axis=1), has_constant='add'), 'Q')

        # print(exog)
        # print(exog[exog.index> old_results['pred_mean'].index.max()])
        # print(old_results['pred_mean'].index.max())
        # print(exog[exog.index> old_results['pred_mean'].index.max()].index.min())
        # print(ending_quarter)

        # get new predictions
        pred = model.get_prediction(
            exog=exog[exog.index> old_results['gdp_growth_pred_mean'].index.max()], # only predict on new datapoints
            start=exog[exog.index> old_results['gdp_growth_pred_mean'].index.max()].index.min(),
            end=ending_quarter)
        
        # append results for GDP prediction
        results = pred.conf_int()
        results['gdp_growth_pred_mean'] = pred.predicted_mean
        results['gdp_growth_stderr_mean'] = pred.se_mean
        results['gdp_growth_var_pred_mean'] = pred.var_pred_mean
        results['GeoID'] = g
        results['CountryName'] = g_short
        results['RegionName'] = np.nan
        results['RegionName'] = results['RegionName'].astype(object)
        results['Date'] = results.index
        geo_pred_dfs.append(results)

    # concat old and new predictions
    predictions = pd.concat(old_geo_pred_dfs + geo_pred_dfs)

    # drop duplicate predictions 
    predictions = predictions.loc[predictions.index.drop_duplicates()]
    return predictions.loc[starting_quarter:, :]