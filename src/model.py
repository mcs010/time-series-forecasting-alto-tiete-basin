"""
Author: Matheus Cardoso da Silva
"""
#%%
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from skforecast.preprocessing import series_long_to_dict
from skforecast.preprocessing import exog_long_to_dict
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries

from sklearn.metrics import r2_score

#%%
def train_model(algorithm, features_list, number_of_features, lags, steps, series_dict_train, exog_dict_train, seed):
    """
    Training model
    """

    features = features_list[:number_of_features]

    if algorithm == "LGBM":
        regressor = LGBMRegressor(random_state=seed, verbose=-1, max_depth=5)
    elif algorithm == "ExtraTrees":
        regressor = ExtraTreesRegressor(random_state=seed)
    elif algorithm == "RF":
        regressor = RandomForestRegressor(random_state=seed)

    forecaster = ForecasterRecursiveMultiSeries(
                    regressor          = regressor, 
                    lags               = lags, 
                    encoding           = "ordinal", 
                    dropna_from_series = False
                )
    
    forecaster.fit(series=series_dict_train, exog=exog_dict_train, suppress_warnings=True)

    return forecaster

#%%
def predict(forecaster, features_list, number_of_features, lags, steps, series_dict_train, series_dict, exog_dict, seed):

    cv = TimeSeriesFold(
        steps                 = steps,
        initial_train_size    = len(series_dict_train["EMMI02900"]),
        #refit                 = False,
        refit                 = True,
        allow_incomplete_fold = True,
        fixed_train_size      = False,
        gap                   = 0,
    )

    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster,
        series                = series_dict,
        exog                  = exog_dict,
        cv                    = cv,
        levels                = None,
        #metric                = "mean_absolute_error",
        #metric = ['mean_squared_error', 'mean_absolute_error',
        #'mean_absolute_percentage_error', 'mean_squared_log_error',
        #'mean_absolute_scaled_error', 'root_mean_squared_scaled_error', r2_score],
        metric                = r2_score,
        add_aggregated_metric = True,
        #n_jobs                ="auto",
        n_jobs                = -1,
        verbose               = True,
        show_progress         = True,
        suppress_warnings     = True
    )