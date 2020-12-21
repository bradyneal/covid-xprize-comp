# Note: don't run this in PyCharm as the default output will
# be in SciView and the plot will not be interactive
# Run this script from the terminal instead so the output
# of the plot will open in a new window so the plot will
# be interactive.

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # ..../covid-xprize-comp/ongoing/predictors

# Load predictions
lstm_predictions = os.path.join(ROOT_DIR, "tempgeolstm", "predictions", "predictions_future_lstm.csv")
lgbm_predictions = os.path.join(ROOT_DIR, "tempgeolgbm", "predictions", "predictions_future_lgbm.csv")

lstm_predictions_df = pd.read_csv(lstm_predictions,
                      parse_dates=['Date'],
                      encoding="ISO-8859-1",
                      dtype={"RegionName": str,
                             "RegionCode": str},
                      error_bad_lines=False)

lgbm_predictions_df = pd.read_csv(lgbm_predictions,
                                  parse_dates=['Date'],
                                  encoding="ISO-8859-1",
                                  dtype={"RegionName": str,
                                         "RegionCode": str},
                                  error_bad_lines=False)

num_regions = lstm_predictions_df.groupby(['GeoID']).ngroups

num_days = lstm_predictions_df.groupby(['Date']).ngroups

min_pred = min(lstm_predictions_df['PredictedDailyTotalCases'].min(), lgbm_predictions_df['PredictedDailyTotalCases'].min())
max_pred = min(lstm_predictions_df['PredictedDailyTotalCases'].max(), lgbm_predictions_df['PredictedDailyTotalCases'].max())

x = np.array(list(range(0, num_days)))  # lstm_predictions_df['Date']

y = np.array([10] * num_days)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)
plt.xlabel("Day")
plt.ylabel("Predicted Daily Total Cases")
p, = plt.plot(x, y, linewidth=2, color='blue')
plt.axis([0, num_days, min_pred, 2e6])


d = {}  # map integer to region name
for i, region in enumerate(lstm_predictions_df["GeoID"].unique()):
    d[i] = region


def get_pred(alpha, lstm_predictions_df, lgbm_predictions_df):
    lstm_pred = lstm_predictions_df['PredictedDailyTotalCases']
    lgbm_pred = lgbm_predictions_df['PredictedDailyTotalCases']

    return alpha * lstm_pred + (1 - alpha) * lgbm_pred


axSlider = plt.axes([0.1, 0.2, 0.8, 0.05])
slider1 = Slider(ax=axSlider,
                 label='alpha',
                 valmin=0,
                 valmax=1,
                 valinit=0.5,
                 valfmt='%1.2f',
                 valstep=0.01,
                 closedmax=True,
                 color='blue')

axSlider2 = plt.axes([0.1, 0.1, 0.8, 0.05])
slider2 = Slider(ax=axSlider2,
                 label='Region',
                 valmin=0,
                 valmax=num_regions,
                 valinit=30,
                 # valfmt='%1.2f',
                 valstep=1,
                 closedmax=True,
                 color='green')


def val_update(val):
    ensemble_data = pd.DataFrame()
    ensemble_data['GeoID'] = lstm_predictions_df['GeoID']
    ensemble_data['Date'] = lstm_predictions_df['Date']
    ensemble_data['PredictedDailyTotalCases'] = get_pred(slider1.val, lstm_predictions_df, lgbm_predictions_df)
    ensemble_data = ensemble_data[ensemble_data['GeoID'] == d[slider2.val]]
    ensemble_data['Day'] = range(ensemble_data.shape[0])
    plt.text(120, 1.75e6, 'fsdfds')  # set to region_name

    p.set_ydata(ensemble_data['PredictedDailyTotalCases'])
    print(slider1.val)
    print(ensemble_data)
    plt.draw()


slider2.on_changed(val_update)
slider1.on_changed(val_update)

axButton1 = plt.axes([0.25, 0.9, 0.1, 0.1])
btn1 = Button(axButton1, 'Reset')

axButton2 = plt.axes([0.4, 0.9, 0.2, 0.1])
btn2 = Button(axButton2, 'Set Val')


def reset_sliders(event):
    slider1.reset()
    slider2.reset()


btn1.on_clicked(reset_sliders)


def set_value(val):
    slider2.set_val(50)


btn2.on_clicked(set_value)

plt.show()



