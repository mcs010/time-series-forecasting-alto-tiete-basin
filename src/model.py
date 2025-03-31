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

def train_model(algorithm, features_list, lags, steps, series_dict_train, exog_dict_train, seed):
    """
    Training model
    """
    if algorithm == "LGBM":
        regressor = LGBMRegressor(random_state=seed, verbose=-1, max_depth=5)
    elif algorithm == "ExtraTrees":
        regressor = ExtraTreesRegressor(random_state=seed)
    elif algorithm == "RF":
        regressor = RandomForestRegressor(random_state=seed)