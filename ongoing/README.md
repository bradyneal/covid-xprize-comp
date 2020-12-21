## Evaluating models

You can evaluate our models (LGBM and LSTM) by doing the following steps:
    1- Set the test configs. you want to evaluate by defining the list `TEST_CONFIGS`in `<repo_root>/ongoing/predictors/base.py`.
    2- From the root of the repo, run `python ongoing/predictors/tempgeolgbm/tempgeolgbm_predictor.py` or `python ongoing/predictors/tempgeolstm/tempgeolstm_predictor.py`.

## Visualizing future predictions

Run the following jupyter notebook: `<repo_root>/ongoing/predictors/look_into_future.ipynb`.

## Generating predictions

You can find `predict.py` scripts under each model directory. This script follows the official API of the competition.
