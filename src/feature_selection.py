from skforecast.feature_selection import select_features_multiseries
from sklearn.feature_selection import SelectFromModel

from skforecast.recursive import ForecasterRecursiveMultiSeries

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

def loop_feature_selection(algorithms: list, lags: list, series_dict, exog_dict, actual_date_and_time, subsample:float, seed: int):
    df_best_features = pd.DataFrame(columns=["algorithm", "lags", "important_features"])
    for algorithm in algorithms:
        for lag in lags:
            selected_exog_features = select_features(algorithm, lag, series_dict, exog_dict, subsample, seed)
            new_df_row = {"algorithm":algorithm, "lags":lag, "important_features":selected_exog_features}
            df_best_features = pd.concat([df_best_features, pd.DataFrame(new_df_row)], ignore_index=True)

    #df_best_features
    df_best_features.to_excel(f"../reports/files/{actual_date_and_time}/best_features.xlsx")
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