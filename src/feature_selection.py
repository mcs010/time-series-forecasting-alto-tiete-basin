from skforecast.feature_selection import select_features_multiseries
from sklearn.feature_selection import SelectFromModel

from skforecast.recursive import ForecasterRecursiveMultiSeries

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

#%%
def select_features(algorithm:str, lags:int, series, exog, subsample:float, seed:int):

    if algorithm == "LGBM":
        regressor = LGBMRegressor(random_state=seed, verbose=-1, max_depth=5)
    elif algorithm == "ExtraTrees":
        regressor = ExtraTreesRegressor(random_state=seed)
    elif algorithm == "RF":
        regressor = RandomForestRegressor(random_state=seed)

    forecaster = ForecasterRecursiveMultiSeries(
                    regressor = regressor,
                    lags      = lags,
                    encoding  = 'ordinal'
                )


    selected_lags, selected_window_features, selected_exog = select_features_multiseries(
        forecaster=forecaster,
        selector=SelectFromModel(estimator=regressor),
        series=series,
        exog=exog,
        subsample=subsample,
        random_state=seed,
        verbose=True
    )

    return selected_exog # List of selected exogenous features