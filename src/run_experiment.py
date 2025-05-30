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
#from skforecast.model_selection import backtesting_forecaster_multiseries
#from skforecast.model_selection import bayesian_search_forecaster_multiseries

from custom import get_series, get_exog, converte_df, concat_all_dfs, create_folder
from model import train_predict_model, tunning_predict
from feature_selection import loop_feature_selection

from datetime import date, datetime
import time
import pathlib
from tqdm import tqdm

seed = 10 # sets seed, mainly for feature selection
seeds = [value for value in range(1, 11)]
#%%
# Sets actual date and time values for later use
actual_datetime = datetime.now()
actual_date = date.today()
actual_date_and_time = f"{actual_date}_{actual_datetime.hour}-{actual_datetime.minute}-{actual_datetime.second}"
#%%
# Marks the beginning of the experiment
start_time = time.process_time()
print(f"---------------- Experiment beginning at {actual_datetime} ----------------")
#%%
# Load all data
# ==============================================================================
data = pd.read_csv("../data/processed/tabela_completa.csv", sep=";")

#data.drop("Unnamed: 0", inplace=True, axis=1)

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
# Create folder for saving results files
create_folder(actual_date_and_time)

#%%
# Setting global variables for feature selection, training and prediction
algorithms = ["LGBM", "ExtraTrees", "RF"]
lags = [1, 6] #12
steps = [1, 6] #6
n_features = 9
subsample = 1.0

BILL02100_list = []
BILL02500_list = []
BILL02900_list = []
BITQ00100_list = []
EMGU00800_list = []
EMMI02900_list = []
GUAR00100_list = []
GUAR00900_list = []
RGDE02200_list = []
RGDE02900_list = []
average_results_list = []
weighted_average_results_list = []
algorithm_results_list = []
features_list_results = []
number_of_features_results_list = []
lag_result_list = []
step_result_list = []
seed_result_list = []
#%%
# Feature selection process
loop_feature_selection(algorithms, lags, series_dict, exog_dict, actual_date_and_time, subsample, seed)
#%%
# Running training and prediction algorithm
step=1
for algorithm in tqdm(algorithms, "Algorithms"):
    for lag in tqdm(lags, "Lags"):
        results, _, algorithm, features_list, number_of_features, lag, step, seed = train_predict_model(algorithm, "", "", lag, step, series_dict, series_dict_train, exog_dict, exog_dict_train, seed)
        #df_results = pd.DataFrame(results).set_index("levels").transpose(copy=True)
        df_results_temp = pd.DataFrame(results).transpose(copy=True).reset_index(drop=True).rename_axis(columns=None)
        column_names = df_results_temp.iloc[0]
        df_results_temp = df_results_temp[1:]
        df_results_temp.columns = column_names
        print(df_results_temp["BILL02100"])
        
        BILL02100_list.append(df_results_temp["BILL02100"].values[0])
        BILL02500_list.append(df_results_temp["BILL02500"].values[0])
        BILL02900_list.append(df_results_temp["BILL02900"].values[0])
        BITQ00100_list.append(df_results_temp["BITQ00100"].values[0])
        EMGU00800_list.append(df_results_temp["EMGU00800"].values[0])
        EMMI02900_list.append(df_results_temp["EMMI02900"].values[0])
        GUAR00100_list.append(df_results_temp["GUAR00100"].values[0])
        GUAR00900_list.append(df_results_temp["GUAR00900"].values[0])
        RGDE02200_list.append(df_results_temp["RGDE02200"].values[0])
        RGDE02900_list.append(df_results_temp["RGDE02900"].values[0])
        average_results_list.append(df_results_temp["average"].values[0])
        weighted_average_results_list.append(df_results_temp["weighted_average"].values[0])
        algorithm_results_list.append(algorithm)
        features_list_results.append(str(features_list))
        number_of_features_results_list.append(number_of_features)
        lag_result_list.append(lag)
        step_result_list.append(step)
        seed_result_list.append(seed)

        #results.to_excel(f"../reports/files/{actual_date_and_time}/{algorithm}_{n_features}_{lags}_{steps[1]}_predictions.xlsx")
df_results = pd.DataFrame(data=zip(BILL02100_list, BILL02500_list, BILL02900_list, BITQ00100_list, EMGU00800_list, EMMI02900_list, GUAR00100_list, GUAR00900_list, RGDE02200_list, RGDE02900_list, average_results_list, weighted_average_results_list, algorithm_results_list, features_list_results, number_of_features_results_list, lag_result_list, step_result_list, seed_result_list), 
                          columns=["BILL02100", "BILL02500", "BILL02900", "BITQ00100", "EMGU00800", "EMMI02900", 
                                   "GUAR00100", "GUAR00900", "RGDE02200", "RGDE02900", "average", "weighted_average",
                                   "algorithm", "features_list", "number_of_features", "lag", "step", "seed"])

print(df_results.head(5))
df_results.to_excel(f"../reports/files/{actual_date_and_time}/scores.xlsx")
#%%
# Hyperparameter Tunning with Grid Search
#results = tunning_predict("LGBM", "", "", 6, series_dict_train, series_dict, exog_dict, seed)

tunning_results = tunning_predict(algorithm, "", "", steps, series_dict_train, series_dict, exog_dict, seed, actual_date_and_time)
#print(results)
tunning_results["algorithm"] = algorithm
tunning_results["steps"] = steps
tunning_results.to_excel(f"../reports/files/{actual_date_and_time}/tunning_{algorithm}_{steps}.xlsx")

#%%
# Marks the end of the experiment
end_time = time.process_time()
print(f"---------------- Experiment finishing with {end_time - start_time} ----------------")
# %%
