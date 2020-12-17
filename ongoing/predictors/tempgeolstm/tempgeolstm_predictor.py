# Uses a learnable embedding for the countries which is fed as input to both encoders (at the respective dense layers).

# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import urllib.request

from keras.layers.core import RepeatVector

# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# noinspection PyPep8Naming
import keras.backend as K
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Lambda
from keras.models import Model

import ongoing.predictors.base as base
from ongoing.predictors.base import BasePredictor

# See https://github.com/OxCGRT/covid-policy-tracker
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, os.pardir, os.pardir, 'data')
TEMPERATURE_DATA_FILE_PATH = os.path.join(DATA_PATH, "temperature_data.csv")
TEMPERATURE_COLUMN = 'temp,C'
HOLIDAY_COLUMN = 'Holiday'

NB_LOOKBACK_DAYS = 21
NB_TEST_DAYS = 14
WINDOW_SIZE = 7
US_PREFIX = "United States / "
NUM_TRIALS = 10
NUM_EPOCHS = 1000
LSTM_SIZE = 32
EMBED_SIZE = 4
NPI_DELAY = 0
TEMP_SCALE = 20.  # divide temperature values by 20 so they're roughly in the range 0-2
AVG_EARTH_TEMP = 16./TEMP_SCALE  # average temperature on earth (used to predict for locations where temperature data is missing)

HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-05-06")

class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)


# Functions to be used for lambda layers in model
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)


class tempGeoLSTMPredictor(BasePredictor):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, path_to_model_weights=None, path_to_geos=None,
                 use_embedding=True, embed_size=EMBED_SIZE, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS,
                 nb_test_days=NB_TEST_DAYS, window_size=WINDOW_SIZE, npi_delay=NPI_DELAY,
                 num_trials=NUM_TRIALS, num_epochs=NUM_EPOCHS, seed=base.SEED):

        super().__init__(seed=seed)
        self.use_embedding=use_embedding
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.nb_lookback_days = nb_lookback_days
        self.nb_test_days = nb_test_days
        self.window_size = window_size
        self.npi_delay = npi_delay
        self.num_trials = num_trials
        self.num_epochs = num_epochs

        if path_to_model_weights:
            # Load model weights
            nb_context = 4  # time series of new cases rate, death rate, temperature, and holidays are used
            nb_action = len(base.NPI_COLUMNS)
            if self.use_embedding:
                with open(path_to_geos, 'r') as f:
                    self.geos = [g.rstrip() for g in f.readlines()]
            else:
                self.geos = None
            self.predictor, _ = self._construct_model(nb_context=nb_context,
                                                      nb_action=nb_action,
                                                      embed_size=self.embed_size,
                                                      lstm_size=self.lstm_size,
                                                      nb_lookback_days=self.nb_lookback_days)
            self.predictor.load_weights(path_to_model_weights)

        # read and preprocess temperature data
        self.temp_df = pd.read_csv(TEMPERATURE_DATA_FILE_PATH,
                                   parse_dates=['Date'],
                                   encoding="ISO-8859-1",
                                   dtype={"RegionName": str,
                                          "RegionCode": str},
                                   error_bad_lines=False)
        self.temp_df["GeoID"] = np.where(self.temp_df["RegionName"].isnull(),
                                         self.temp_df["CountryName"],
                                         self.temp_df["CountryName"] + ' / ' + self.temp_df["RegionName"])
        self.temp_df[TEMPERATURE_COLUMN] = self.temp_df[TEMPERATURE_COLUMN]/TEMP_SCALE

        self.country_samples = None  # will be set when fit() or predict() are called

    def predict(self, data=None, start_date=None, end_date=None):
        if self.train_df is None:
            raise Exception("train_df must be defined before calling predict()")

        if data is None:
            data = self.test_df
        if start_date is None:
            start_date = pd.to_datetime(data.Date.min(), format='%Y-%m-%d')
        if end_date is None:
            end_date = pd.to_datetime(data.Date.max(), format='%Y-%m-%d')

        # if self.country_samples is None:
        #     # merge the two dataframes (keep only rows where new cases rate and temperature are available)
        #     # self.train_df = pd.merge(self.train_df, self.temp_df, on=['CountryName', 'RegionName', 'GeoID', 'Date'], how='inner')
        #     self.country_samples = self._create_country_samples(self.train_df,
        #                                                         list(self.train_df.GeoID.unique()),
        #                                                         self.nb_lookback_days,
        #                                                         self.npi_delay,
        #                                                         self.nb_test_days)
        train_df = pd.merge(self.train_df, self.temp_df, on=['CountryName', 'RegionName', 'GeoID', 'Date'], how='left')
        train_df[TEMPERATURE_COLUMN] = train_df[TEMPERATURE_COLUMN].fillna(AVG_EARTH_TEMP)
        train_df[HOLIDAY_COLUMN] = train_df[HOLIDAY_COLUMN].fillna(0)
        self.country_samples = self._create_country_samples(train_df,
                                                            list(self.train_df.GeoID.unique()),
                                                            self.nb_lookback_days,
                                                            self.npi_delay,
                                                            self.nb_test_days)

        nb_days = (end_date - start_date).days + 1

        # Prepare the output
        forecast = {"GeoID": [],
                    "CountryName": [],
                    "RegionName": [],
                    "Date": [],
                    "PredictedDailyTotalCases": [],
                    "PredictedDailyNewCases": [],
                    "PredictedDailyTotalDeaths": [],
                    "PredictedDailyNewDeaths": []}

        # For each requested geo
        geos = data.GeoID.unique()
        for g in geos:
            if self.use_embedding and g not in self.geos:
                # the model was not trained for this geo: return zeroes
                print("WARNING: The model was not trained for {}".format(g))
                pred_total_cases = [0] * nb_days
                pred_new_cases = [0] * nb_days
                pred_total_deaths = [0] * nb_days
                pred_new_deaths = [0] * nb_days
                geo_start_date = start_date
            else:
                cdf = self.train_df[self.train_df.GeoID == g]

                if len(cdf) == 0:
                    # we don't have historical data for this geo: return zeroes
                    print("WARNING: No historical data for {}".format(g))
                    pred_total_cases = [0] * nb_days
                    pred_new_cases = [0] * nb_days
                    pred_total_deaths = [0] * nb_days
                    pred_new_deaths = [0] * nb_days
                    geo_start_date = start_date
                else:
                    last_known_date = cdf.Date.max()
                    # Start predicting from start_date, unless there's a gap since last known date
                    geo_start_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
                    npis_gdf = data[(data.Date >= geo_start_date - pd.Timedelta(days=self.npi_delay)) & (data.Date <= end_date - pd.Timedelta(days=self.npi_delay))]
                    temp_gdf = self.temp_df[(self.temp_df.Date >= geo_start_date.replace(year=2020)) & (self.temp_df.Date <= end_date.replace(year=2020))]
                    # if temp_gdf.empty:
                    #     print("WARNING: No temperature data available for {} ({} - {})".format(g, geo_start_date.replace(year=2020).strftime("%Y/%m/%d"), end_date.replace(year=2020).strftime("%Y/%m/%d")))
                    #     temp_gdf = pd.DataFrame.from_dict({TEMPERATURE_COLUMN: AVG_EARTH_TEMP*np.ones((end_date-geo_start_date).days),
                    #                                        HOLIDAY_COLUMN: np.zeros(((end_date-geo_start_date).days))})
                    pred_total_cases, pred_new_cases, pred_total_deaths, pred_new_deaths = self._get_new_cases_preds(cdf, g, npis_gdf, temp_gdf)

            # Append forecast data to results to return
            country = data[data.GeoID == g].iloc[0].CountryName
            region = data[data.GeoID == g].iloc[0].RegionName
            for i, (ptot_cases, pnew_cases, ptot_deaths, pnew_deaths) in enumerate(zip(pred_total_cases, pred_new_cases, pred_total_deaths, pred_new_deaths)):
                forecast["GeoID"].append(g)
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyTotalCases"].append(ptot_cases)
                forecast["PredictedDailyNewCases"].append(pnew_cases)
                forecast["PredictedDailyTotalDeaths"].append(ptot_deaths)
                forecast["PredictedDailyNewDeaths"].append(pnew_deaths)

        forecast_df = pd.DataFrame.from_dict(forecast)
        # Return only the requested predictions
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]

    def _get_new_cases_preds(self, c_df, g, npis_df, temp_df):
        cdf = c_df[c_df.ConfirmedCases.notnull()]
        initial_context_input = self.country_samples[g]['X_test_context'][-1]
        initial_action_input = self.country_samples[g]['X_test_action'][-1]
        if self.use_embedding:
            country_id = np.array([self.geos.index(g)])
        else:
            country_id = np.array([0.])
        # Predictions with passed npis
        cnpis_df = npis_df[npis_df.GeoID == g]
        npis_sequence = np.array(cnpis_df[base.NPI_COLUMNS])
        ctemp_df = temp_df[temp_df.GeoID == g]
        if ctemp_df.empty:
            print("WARNING: No temperature data available for {}".format(g))
            temp_sequence = AVG_EARTH_TEMP*np.ones(npis_sequence.shape[0])
            holiday_sequence = np.zeros(npis_sequence.shape[0])
        else:
            temp_sequence = np.array(ctemp_df[TEMPERATURE_COLUMN])
            holiday_sequence = np.array(ctemp_df[HOLIDAY_COLUMN])
        # Get the predictions with the passed NPIs
        preds = self._roll_out_predictions(self.predictor,
                                           initial_context_input,
                                           initial_action_input,
                                           country_id,
                                           npis_sequence,
                                           temp_sequence,
                                           holiday_sequence)
        preds_cases = preds[:,0]
        preds_deaths = preds[:,1]
        # Gather info to convert to total cases
        prev_confirmed_cases = np.array(cdf.ConfirmedCases)
        prev_new_cases = np.array(cdf.NewCases)
        initial_total_cases = prev_confirmed_cases[-1]
        pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time
        prev_confirmed_deaths = np.array(cdf.ConfirmedDeaths)
        prev_new_deaths = np.array(cdf.NewDeaths)
        initial_total_deaths = prev_confirmed_deaths[-1]

        # Compute predictor's forecast
        pred_total_cases, pred_new_cases = base.convert_ratios_to_total_cases(
            preds_cases,
            self.window_size,
            prev_new_cases,
            initial_total_cases,
            pop_size)

        # Compute predictor's deaths forecast
        pred_total_deaths, pred_new_deaths = base.convert_ratios_to_total_deaths(
            preds_deaths,
            self.window_size,
            prev_new_deaths,
            initial_total_deaths)

        return pred_total_cases, pred_new_cases, pred_total_deaths, pred_new_deaths

    @staticmethod
    def _create_country_samples(df: pd.DataFrame, geos: list, nb_lookback_days: int, npi_delay: int, nb_test_days: int) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :return: a dictionary of train and test sets, for each specified country
        """
        context_column = ['PredictionRatio', 'DeathRatio', TEMPERATURE_COLUMN, HOLIDAY_COLUMN]
        action_columns = base.NPI_COLUMNS
        outcome_column = ['PredictionRatio', 'DeathRatio']
        country_samples = {}
        for i, g in enumerate(geos):
            cdf = df[df.GeoID == g]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            context_data = np.array(cdf[context_column])
            action_data = np.array(cdf[action_columns])
            outcome_data = np.array(cdf[outcome_column])
            context_samples = []
            action_samples = []
            outcome_samples = []
            nb_total_days = outcome_data.shape[0]
            for d in range(nb_lookback_days+npi_delay, nb_total_days):
                context_samples.append(context_data[d - nb_lookback_days: d])
                action_samples.append(action_data[d-npi_delay - nb_lookback_days: d-npi_delay])
                outcome_samples.append(outcome_data[d])
            if len(outcome_samples) > 0:
                X_context = np.stack(context_samples, axis=0)
                X_action = np.stack(action_samples, axis=0)
                X_country = i*np.ones(X_context.shape[0])
                y = np.stack(outcome_samples, axis=0)
                country_samples[g] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'X_country': X_country,
                    'y': y,
                    'X_test_context': X_context[-nb_test_days:],
                    'X_test_action': X_action[-nb_test_days:],
                    'X_test_country': X_country[-nb_test_days:],
                    'y_test': y[-nb_test_days:],
                }
        return country_samples

    # Function for performing roll outs into the future
    def _roll_out_predictions(self, predictor, initial_context_input, initial_action_input, country_id, future_action_sequence, future_temperature_sequence, future_holiday_sequence):
        nb_roll_out_days = future_action_sequence.shape[0]
        pred_output = np.zeros((nb_roll_out_days, 2))
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        country_input = np.expand_dims(np.copy(country_id), axis=0)
        for d in range(nb_roll_out_days):
            action_input[:, :-1] = action_input[:, 1:]
            # Use the passed actions
            action_sequence = future_action_sequence[d]
            action_input[:, -1] = action_sequence
            if self.use_embedding:
                inputs = [context_input, action_input, country_input]
            else:
                inputs = [context_input, action_input]

            pred = predictor.predict(inputs)
            pred_output[d] = pred[-1]
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1, 0:2] = pred[-1]
            context_input[:, -1, 2] = future_temperature_sequence[d]
            context_input[:, -1, 3] = future_holiday_sequence[d]
        return pred_output

    def fit(self):
        if self.train_df is None:
            raise Exception("train_df must be defined bfr calling predict()")

        # merge the two dataframes (keep only rows where new cases rate and temperature are available)
        train_df = pd.merge(self.train_df, self.temp_df, on=['CountryName', 'RegionName', 'GeoID', 'Date'], how='inner')
        self.country_samples = self._create_country_samples(train_df,
                                                            list(train_df.GeoID.unique()),
                                                            self.nb_lookback_days,
                                                            self.npi_delay,
                                                            self.nb_test_days)
        self.geos = list(self.country_samples.keys())

        # Aggregate data for training
        all_X_context_list = [self.country_samples[c]['X_context']
                              for c in self.country_samples]
        all_X_action_list = [self.country_samples[c]['X_action']
                             for c in self.country_samples]
        all_X_country_list = [self.country_samples[c]['X_country']
                              for c in self.country_samples]
        all_y_list = [self.country_samples[c]['y']
                      for c in self.country_samples]
        X_context = np.concatenate(all_X_context_list)
        X_action = np.concatenate(all_X_action_list)
        X_country = np.concatenate(all_X_country_list)
        y = np.concatenate(all_y_list)

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)

        X_context, X_action, X_country, y = self._permute_data(X_context, X_action, X_country, y)
        self.predictor, training_model = self._construct_model(nb_context=X_context.shape[-1],
                                                      nb_action=X_action.shape[-1],
                                                      embed_size=self.embed_size,
                                                      lstm_size=self.lstm_size,
                                                      nb_lookback_days=self.nb_lookback_days)
        history = self._train_model(training_model, X_context, X_action, X_country, y, epochs=self.num_epochs, verbose=0)
        top_epoch = np.argmin(history.history['val_loss'])
        train_loss = history.history['loss'][top_epoch]
        val_loss = history.history['val_loss'][top_epoch]
        print('Train Loss:', train_loss)
        print('Val Loss:', val_loss)

    # Shuffling data prior to train/val split
    def _permute_data(self, X_context, X_action, X_country, y):
        p = np.random.permutation(y.shape[0])
        X_context = X_context[p]
        X_action = X_action[p]
        X_country = X_country[p]
        y = y[p]
        return X_context, X_action, X_country, y

    # Construct model
    def _construct_model(self, nb_context, nb_action, embed_size=10, lstm_size=32, nb_lookback_days=21):
        if self.use_embedding:
            # Create country embedding
            country_id = Input(shape=(1,),
                               name='country_id')
            emb = Embedding(len(self.geos), embed_size)(country_id)
            emb = Reshape((embed_size,))(emb)

        # Create context encoder
        context_input = Input(shape=(nb_lookback_days, nb_context),
                              name='context_input')
        x = LSTM(lstm_size,
                 return_sequences=False,
                 name='context_lstm')(context_input)
        if self.use_embedding:
            x = Concatenate(axis=1)([x, emb])  # concatenate the output of the LSTM with the country embedding prior to the dense layer
        context_output = Dense(units=1,
                               activation='softplus',
                               name='context_dense')(x)

        # Create action encoder
        # Every aspect is monotonic and nonnegative except final bias
        action_input = Input(shape=(nb_lookback_days, nb_action),
                             name='action_input')
        x = LSTM(units=lstm_size,
                 kernel_constraint=Positive(),
                 recurrent_constraint=Positive(),
                 bias_constraint=Positive(),
                 return_sequences=False,
                 name='action_lstm')(action_input)
        if self.use_embedding:
            x = Concatenate(axis=1)([x, emb])  # concatenate the output of the LSTM with the country embedding prior to the dense layer
        action_output = Dense(units=2,
                              activation='sigmoid',
                              # kernel_constraint=Positive(),
                              name='action_dense')(x)

        # Create prediction model
        model_output = Lambda(_combine_r_and_d, name='prediction')(
            [context_output, action_output])
        if self.use_embedding:
            inputs = [context_input, action_input, country_id]
        else:
            inputs = [context_input, action_input]

        model = Model(inputs=inputs,
                      outputs=[model_output])
        model.compile(loss='mae', optimizer='adam')

        # Create training model, which includes loss to measure
        # variance of action_output predictions
        training_model = Model(inputs=inputs,
                               outputs=[model_output])
        training_model.compile(loss='mae',
                               optimizer='adam')

        return model, training_model

    # Train model
    def _train_model(self, training_model, X_context, X_action, X_country, y, epochs=1, verbose=0):
        early_stopping = EarlyStopping(patience=20,
                                       restore_best_weights=True)
        if self.use_embedding:
            inputs = [X_context, X_action, X_country]
        else:
            inputs = [X_context, X_action]
        history = training_model.fit(inputs, [y],
                                     epochs=epochs,
                                     batch_size=32,
                                     validation_split=0.1,
                                     callbacks=[early_stopping],
                                     verbose=verbose)
        return history

    def save_model(self, path_to_weights, path_to_country_list=None):
        self.predictor.save_weights(path_to_weights)
        if self.use_embedding and path_to_country_list is not None:
            with open(path_to_country_list, 'w') as f:
                f.writelines("{}\n".format(g) for g in self.geos)


if __name__ == '__main__':
    # Run all test cases
    # model = tempGeoLSTMPredictor('./ongoing/predictors/tempgeolstm/models/model.h5', './ongoing/predictors/tempgeolstm/models/countries.txt')
    model = tempGeoLSTMPredictor(use_embedding=False)
    model.evaluate()
    model.save_model('./ongoing/predictors/tempgeolstm/models/model_no_embed.h5', './ongoing/predictors/tempgeolstm/models/countries.txt')
