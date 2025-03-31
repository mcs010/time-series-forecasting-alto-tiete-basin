"""
Author: Matheus Cardoso da Silva

This file contains custom modules
"""
import math
import pandas as pd

# Functions to calculate dissolved oxygen saturation (uses air temperature)

def saturated_DO_concentration(temperature, altitude):
  """
  First step to calculate saturated DO
  Receives the temperature and altitude of the water body
  Returns the concentration of DO saturation
  """
  return (14.62-0.3898*temperature+0.006969*pow(temperature, 2)-0.00005896*pow(temperature, 3))*pow(1-0.0000228675*altitude, 5.167)


def DO_percentage(DO, temperature, altitude):
  """
  Receives DO and the concentration of DO saturation
  Return percentage of DO saturation
  """
  return (100*DO)/saturated_DO_concentration(temperature, altitude)

"""
Function to calculate quality value (q_i) of nth parameter (between 0 and 100) obtained from the respective "average curve of quality variation"
in function of the parameter concentration
"""

def q_i_coliforms(coliforms):
  if(coliforms<=0):
    return 100
  log_value = math.log10(coliforms)
  if(log_value<=0):
    return 100
  elif(log_value>=0 and log_value<=1):
    return 100-33.5*log_value
  elif(log_value>1 and log_value<=5):
    return 100-37.2*log_value+3.60743*math.pow(log_value, 2)
  elif(log_value>5):
    return 3

def q_i_pH(ph):
  if(ph>0.0 and ph<=2.0):
    return 2
  elif(ph>2.0 and ph<=4.0):
    return 13.6-10.64*ph+2.4364*pow(ph, 2)
  elif(ph>4.0 and ph<=6.2):
    return 155.5-77.36*ph+10.2481*pow(ph, 2)
  elif(ph>6.2 and ph<=7.0):
    return -657.2+197.38*ph-12.9167*pow(ph, 2)
  elif(ph>7.0 and ph<=8.0):
    return -427.8+142.05*ph-9.695*pow(ph, 2)
  elif(ph>8.0 and ph<=8.5):
    return 216-16*ph
  elif(ph>8.5 and ph<=9.0):
    return 1415823*math.exp(-1.1507*ph)
  elif(ph>9.0 and ph<=10.0):
    return 288-27*ph
  elif(ph>10.0 and ph<=12.0):
    return 633-106.5*ph+4.5*pow(ph, 2)
  elif(ph>12.0 and ph<=14.0):
    return 3

def q_i_DBO(dbo):
  if(dbo>0 and dbo<=5):
    return 99.96*math.exp(-0.1232728*dbo)
  elif(dbo>5 and dbo<=15):
    return 104.67-31.5463*math.log(dbo)
  elif(dbo>15 and dbo<=30):
    return 4394.91*math.pow(dbo, -1.99809)
  elif(dbo>30):
    return 2

def q_i_nitrogen(nitrogen):
  if(nitrogen>0 and nitrogen<=10):
    return 100-8.169*nitrogen+0.3059*math.pow(nitrogen, 2)
  elif(nitrogen>10 and nitrogen<=60):
    return 101.9-23.1023*math.log(nitrogen)
  elif(nitrogen>60 and nitrogen<=100):
    return 159.3148*math.exp(-0.0512842*nitrogen)
  elif(nitrogen>100):
    return 1

def q_i_phosphorus(phosphorus):
  phosphate = phosphorus*3.066 # Adjust to phosphate
  if(phosphate>0 and phosphate<=1):
    return 99.9*math.exp(-0.91629*phosphate)
  elif(phosphate>1 and phosphate<=5):
    return 57.6-20.178*phosphate+2.1326*math.pow(phosphate, 2)
  elif(phosphate>5 and phosphate<=10):
    return 19.08*math.exp(-0.13544*phosphate)
  elif(phosphate>10):
    return 5

def q_i_temperature():
  return 94

def q_i_turbidity(turbidity):
  if turbidity <= 0:
    return 100
  if(turbidity>0 and turbidity<=25):
    return 100.17-2.67*turbidity+0.03775*math.pow(turbidity, 2)
  elif(turbidity>25 and turbidity<=100):
    return 84.96*math.exp(-0.016206*turbidity)
  elif(turbidity>100):
    return 5

def q_i_total_residue(residue):
  if(residue>0 and residue<=150):
    return 79.75+0.166*residue-0.001088*math.pow(residue, 2)
  elif(residue>150 and residue<=500):
    return 101.67-0.13917*residue
  elif(residue>500):
    return 32

def q_i_dissolved_oxigen_saturation_percentage(do, temperature, code_point):
  """
  Altitudes of the points
  BILL02030 = 745m
  BILL02100 = 743m
  BILL02500 = 743m
  BILL02900 = 740m
  BIRP00500 = 743m
  BITQ00100 = 744m
  RGDE02030 = 745m
  RGDE02200 = 744m
  RGDE02900 = 745m
  """
  if(code_point == 'BILL02030'):
    altitude = 745
  elif(code_point == 'BILL02100'):
    altitude = 743
  elif(code_point == 'BILL02500'):
    altitude = 743
  elif(code_point == 'BILL02900'):
    altitude = 740
  elif(code_point == 'BIRP00500'):
    altitude = 743
  elif(code_point == 'BITQ00100'):
    altitude = 744
  elif(code_point == 'RGDE02030'):
    altitude = 745
  elif(code_point == 'RGDE02200'):
    altitude = 744
  elif(code_point == 'RGDE02900'):
    altitude = 745
  elif(code_point == 'GUAR00900'):
    altitude = 735
  elif(code_point == 'GUAR00100'):
    altitude = 735
  elif(code_point == 'EMMI02900'):
    altitude = 730
  elif(code_point == 'EMGU00800'):
    altitude = 730

  do_percentage = DO_percentage(do, temperature, altitude)

  if(do_percentage>0 and do_percentage<=50):
    return 3+0.34*do_percentage+0.008095*math.pow(do_percentage, 2)+1.35252*0.00001*math.pow(do_percentage, 3)
  elif(do_percentage>50 and do_percentage<=85):
    return 3-1.166*do_percentage+0.058*math.pow(do_percentage, 2)-3.803435*0.0001*math.pow(do_percentage, 3)
  elif(do_percentage>85 and do_percentage<=100):
    return 3+3.7745*math.pow(do_percentage, 0.704889)
  elif(do_percentage>100 and do_percentage<=140):
    return 3+2.9*do_percentage-0.02496*math.pow(do_percentage, 2)+5.60919*0.00001*math.pow(do_percentage, 3)
  elif(do_percentage>140):
    return 3+47
  
# Load time series of multiple lengths and exogenous variables
# ==============================================================================

def get_series(data, ponto_str):
    """
    Receives a dataframe and the category string
    Returns the series dataframe (datetime index column + target feature column)
    """
    df_ponto = data[data['Código Ponto'] == ponto_str]
    df_series = df_ponto[["Código Ponto", "Data Coleta", "WQI"]].copy()
    return df_series

def get_exog(data, ponto_str):
    """
    Receives the dataframe grouped by series (category) and the series string
    Return the dataframe with exogenous variables values
    """
    df_ponto = data[data['Código Ponto'] == ponto_str]
    df_exog = df_ponto[["Código Ponto", "Data Coleta", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]].copy()
    return df_exog

def converte_df(df_ponto):
    """
    Receives the category string
    Returns a dataframe for that category, transforming de datetime index column to the 
    """
    codigo_ponto = df_ponto["Código Ponto"].iloc[0]
    df_ponto['Data Coleta'] = pd.to_datetime(df_ponto['Data Coleta'])
    df_ponto['Data Coleta'] = df_ponto['Data Coleta'].apply(lambda x: x.strftime('%Y-%m-01 00:00:00'))
    df_ponto_freq = df_ponto.set_index(pd.DatetimeIndex(df_ponto['Data Coleta']))
    df_ponto_freq = df_ponto_freq.asfreq("2MS")
    df_ponto_freq.drop('Data Coleta', axis=1, inplace=True)
    df_ponto_freq['Código Ponto'] = codigo_ponto
    return df_ponto_freq

def concat_all_dfs(pontos, lista_freq):
  """
  Concat all series dataframes
  Receives the category strings list and the dataframe for each category
  Returns the concatenated dataframe with each category
  """  
  full_df = None
  for ponto_str in pontos:
      if full_df is None:
          full_df = lista_freq[ponto_str]
      else:
          full_df = pd.concat([full_df, lista_freq[ponto_str]], axis=0)
  return full_df