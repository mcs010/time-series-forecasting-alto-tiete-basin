"""
Author: Matheus Cardoso da Silva
"""
#%%
# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from skforecast.preprocessing import series_long_to_dict
from skforecast.preprocessing import exog_long_to_dict
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster_multiseries

from custom import get_series, get_exog, converte_df, concat_all_dfs
from model import train_predict_model, tunning_predict

seed = 120

#%%
# Load all data
# ==============================================================================
data = pd.read_csv("../data/processed/tabela_completa.csv")

data.drop("Unnamed: 0", inplace=True, axis=1)

data.head()

#%%
pontos = ["BILL02100", "BILL02500", "BILL02900", "BITQ00100", "RGDE02200", "RGDE02900", "GUAR00100", "EMGU00800", "EMMI02900", "GUAR00900"]
pontos_series = {}
pontos_exog = {}
for ponto_str in pontos:
    pontos_series[ponto_str] = get_series(data, ponto_str)
    pontos_exog[ponto_str] = get_exog(data, ponto_str)

print(pontos_series[pontos[0]].head(5))
print("")
print(pontos_exog[pontos[0]].head(5))

#%%
pontos_series_freq = {}
pontos_exog_freq = {}
for ponto_str in pontos:
    pontos_series_freq[ponto_str] = converte_df(pontos_series[ponto_str])
    pontos_exog_freq[ponto_str] = converte_df(pontos_exog[ponto_str])
    
print(pontos_series_freq["EMMI02900"].head(5))
print(pontos_exog_freq["EMMI02900"].head(5))

#%%
full_series_df = concat_all_dfs(pontos, pontos_series_freq)
full_series_df['Data Coluna'] = pd.to_datetime(full_series_df.index)

full_exog_df = concat_all_dfs(pontos, pontos_exog_freq)
full_exog_df['Data Coluna'] = pd.to_datetime(full_exog_df.index)

full_df = pd.concat([full_exog_df, full_series_df['WQI']], axis=1)
full_df.drop(full_df.columns[[10]], axis=1, inplace=True)
#full_df["Data Coleta"] = pd.to_datetime(full_df["Data Coleta"], format="%d/%m/%Y")
full_export_df = full_df.sort_values("Data Coleta")
#full_export_df.to_excel("../output/full_df.xlsx")

#%%:

print(full_series_df.head(5))
print(full_series_df.tail(5))
print(full_exog_df.head(5))
print(full_exog_df.tail(5))

#%%
# Transform series and exog to dictionaries
# ==============================================================================
series_dict = series_long_to_dict(
    data      = full_series_df,
    series_id = 'Código Ponto',
    index     = 'Data Coluna',
    values    = 'WQI',
    freq      = '2MS'
)

exog_dict = exog_long_to_dict(
    data      = full_exog_df,
    series_id = 'Código Ponto',
    index     = 'Data Coluna',
    freq      = '2MS'
)

#%%
# Partition data in train and test
# Partition created with date slicing
# ==============================================================================
end_train = '2013-12-30'

series_dict_train = {k: v.loc[: end_train,] for k, v in series_dict.items()}
exog_dict_train   = {k: v.loc[: end_train,] for k, v in exog_dict.items()}
series_dict_test  = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test    = {k: v.loc[end_train:,] for k, v in exog_dict.items()}

#%%
# Description of each partition
# ==============================================================================
for k in series_dict.keys():
    print(f"{k}:")
    try:
        print(
            f"\tTrain: len={len(series_dict_train[k])}, {series_dict_train[k].index[0]}"
            f" --- {series_dict_train[k].index[-1]}"
        )
    except:
        print("\tTrain: len=0")
    try:
        print(
            f"\tTest : len={len(series_dict_test[k])}, {series_dict_test[k].index[0]}"
            f" --- {series_dict_test[k].index[-1]}"
        )
    except:
        print("\tTest : len=0")

#%%
# Exogenous variables for each series
# ==============================================================================
print("\n>>> As variaveis exogenas sao diferentes para cada serie:")
for k in series_dict.keys():
    print(f"{k}:")
    try:
        print(f"\t{exog_dict[k].columns.to_list()}")
    except:
        print("\tNo exogenous variables")

#%%
results, backtest_predictions = train_predict_model("LGBM", "", "", 6, 6, series_dict, series_dict_train, exog_dict, exog_dict_train, seed)

print(type(results))

#%%
results = tunning_predict("LGBM", "", "", 6, series_dict_train, series_dict, exog_dict, seed)

print(type(results))

#%%