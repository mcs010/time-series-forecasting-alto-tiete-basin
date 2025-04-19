#%%
"""
Author: Matheus Cardoso da Silva
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
import custom

#%%
"""
Appends all excel files into one dataframe
"""

all_files = glob.glob(os.path.join("../data/raw", "*.xlsx")) # system path with all excel files

dfs = list() # temporary dataframe

# loop to get all data together within one dataframe
for file in all_files:
  data = pd.read_excel(file, na_values=["NA"])
  dfs.append(data)

df_source = pd.concat(dfs, ignore_index=True) # dataframe with source data

df_source.head() # preview of first 5 lines

df_source.info() # some info about the dataset

# %%
# Pivots the 'Parametro' column lines in order for each parameter to be a column

df_source.pivot_table(values='Valor', index=['Código Ponto', 'Data Coleta'], columns='Parametro', aggfunc='first')

#%%
# Block of code to transform dataset into ABT (Analytics Base Table)

# List of dataset columns
new_columns = ['Código Ponto', 'Data Coleta', 'Hora Coleta', 'pH', 'Temperatura da Água', 'Temperatura do Ar', 'Alumínio Dissolvido', 'Cobre Dissolvido', 'Vanádio Total', 'Estireno', 'm,p-Xileno', 'o-Xileno', 'Sólidos Sedimentáveis', 'Manganês Dissolvido', 'Cor Aparente', 'Condutividade', 'Turbidez', 'Boro Total', 'Óleos e Graxas', 'Fenóis Totais', 'Cianeto Livre', 'Tolueno', 'Etilbenzeno', 'Nitrogênio Kjeldahl', 'Nitrogênio-Nitrito', 'Nitrogênio-Nitrato', 'Nitrogênio Amoniacal', 'Nitrogênio Total', 'Oxigênio Dissolvido', 'DBO (5, 20)', 'DQO', 'Fluoreto Total', 'Sólido Dissolvido Total', 'Sólido Total', 'Alumínio Total', 'Fósforo-Ortofosfato', 'Fósforo Total', 'Sulfato Total', 'Sulfeto Total', 'Cloreto Total', 'Crômio Total', 'Crômio Hexavalente', 'Manganês Total', 'Ferro Total', 'Ferro Dissolvido', 'Níquel Total', 'Cobre Total', 'Zinco Total', 'Arsênio Total', 'Selênio Total', 'Ens. Ecotoxic. C/ Ceriodaphnia dubia', 'TA98 - S9', 'TA98 + S9', 'TA100 - S9', 'TA100 + S9', 'Coliformes Termotolerantes', 'Microcistinas', 'Prata Total', 'Cádmio Total', 'Estanho Total', 'Antimônio', 'Bário Total', 'Mercúrio Total', 'Chumbo Total', 'Urânio Total', 'Coloração', 'Chuvas nas últimas 24h', 'Lítio Total', 'Cobalto Total', 'Benzeno', 'Berílio Total', 'Clorofila-a', 'Feofitina-a', 'Ens. Ecotoxic. c/ Vibrio fischeri', 'Transparência', 'Número de Células de Cianobactérias', 'Escherichia coli**', 'Fósforo-Ortofosfato']
df = df_source.pivot_table(values='Valor', index=['Código Ponto', 'Data Coleta', 'Hora Coleta'], columns='Parametro', aggfunc='first')

#%%
# Clean unnecessary columns

df_param = df
for column in df_param.columns.values:
  if column not in new_columns:
    df_param.drop(column, axis=1, inplace=True)

df_param

#%%
"""
Filter columns needed for WQI prediction
"""

df_wqi = df_param.loc[:,['Coliformes Termotolerantes', 'Escherichia coli**', 'pH', 'DBO (5, 20)', 'Nitrogênio Kjeldahl', 'Nitrogênio-Nitrito', 'Nitrogênio-Nitrato', 'Nitrogênio Amoniacal', 'Nitrogênio Total', 'Fósforo Total', 'Temperatura da Água', 'Turbidez', 'Sólido Total', 'Oxigênio Dissolvido']]

df_wqi

#%%
# Format data

df_wqi = df_wqi.replace(regex={',': '.'}).astype(float) # Replace ',' into '.' and converts object data type into float

df_wqi.info()

#%%
# Drop Null rows because WQI parameters is not measured with null values

for index, row in df_wqi.iterrows():
  null_row = row.isnull().sum()
  if(null_row == 14):
    remove = index
    df_wqi.drop(remove, inplace=True)

df_wqi.info()

#%%
# Consolidate columns that depends of values in other columns

for index, row in df_wqi.iterrows():
  if(pd.isna(row['Coliformes Termotolerantes'])):
    row['Coliformes Termotolerantes'] = 1.25*row['Escherichia coli**']
  if(pd.isna(row['Nitrogênio Total'])):
    row['Nitrogênio Total'] = np.nansum([row['Nitrogênio Kjeldahl'], row['Nitrogênio-Nitrito'], row['Nitrogênio-Nitrato']])

df_wqi.info()

#%%
# Drop Null rows where WQI parameters were not measured
# ***Create a fucntion for this process***

for index, row in df_wqi.iterrows():
  null_row = row.isnull().sum()
  if(null_row == 13):
    remove = index
    df_wqi.drop(remove, inplace=True)

df_wqi.info()

#%%
# Drop columns not needed anymore after consolidating needed columns

df_wqi.drop(['Escherichia coli**', 'Nitrogênio Kjeldahl', 'Nitrogênio-Nitrito', 'Nitrogênio-Nitrato', 'Nitrogênio Amoniacal'], axis=1, inplace=True)

df_wqi.info()

#%%
# Remove rows containing null values all columns

for index, row in df_wqi.iterrows():
  if(pd.isna(row['Coliformes Termotolerantes']) or pd.isna(row['pH']) or pd.isna(row['DBO (5, 20)']) or pd.isna(row['Nitrogênio Total']) or pd.isna(row['Fósforo Total']) or pd.isna(row['Temperatura da Água']) or pd.isna(row['Turbidez']) or pd.isna(row['Sólido Total']) or pd.isna(row['Oxigênio Dissolvido'])):
    df_wqi.drop([index], axis=0, inplace=True)

df_wqi.info()

#%%
# Calculate quality values

"""q_i is the grade obtained according to the parameter concentration in the water"""
q1 = list() # Fecal coliforms parameter quality
q2 = list() # pH parameter quality
q3 = list() # BOD (Biochemical oxygen demand) parameter quality
q4 = list() # Total nitrogen parameter quality
q5 = list() # Total phosphor parameter quality
q6 = list() # Temperature parameter quality
q7 = list() # Turbidity parameter quality
q8 = list() # Total residue parameter quality
q9 = list() # DO (Dissolved Oxygen) parameter quality

"""
Store (Coliforms) values in a list, either Total Coliforms or applying a correction factor to E. Coli when TC is null
"""
for index, row in df_wqi.iterrows():
  if(pd.notna(row['Coliformes Termotolerantes'])):
    q1.append(row['Coliformes Termotolerantes'])
  else:
    q1.append(row['Escherichia coli**']*1.25)

"""
Store (pH) values in a list
"""
q2 = [value for value in df_wqi['pH']]

"""
Store (DBO) values in a list
"""
q3 = [value for value in df_wqi['DBO (5, 20)']]

"""
Store (Total Nitrogen) values in a list
"""
for index, row in df_wqi.iterrows():
  if(pd.isna(row['Nitrogênio Total'])):
    #q4.append(row['Nitrogênio Total'])
    q4.append(np.nansum([row['Nitrogênio Kjeldahl'], row['Nitrogênio-Nitrito'], row['Nitrogênio-Nitrato']]))
  else:
    q4.append(row['Nitrogênio Total'])

"""
Store (Phosphor) values in a list
"""
q5 = [value for value in df_wqi['Fósforo Total']]

"""
Store (Temperature) values in a list
"""
q6 = [value for value in df_wqi['Temperatura da Água']]

"""
Store (Turbidity) values in a list
"""
q7 = [value for value in df_wqi['Turbidez']]

"""
Store (Total Residue) values in a list
"""
q8 = [value for value in df_wqi['Sólido Total']]

"""
Store (Dissolved Oxigen) values in a list
"""
q9_list = list()
for index, row in df_wqi.iterrows():
  q9_list.append(([row['Oxigênio Dissolvido'], row['Temperatura da Água'], index[0]])) # index[0] = 'Código Ponto' index which gets the value of code point index


"""
Applying quality functions over Parameter values obtaining the respective quality values
"""

q1 = [custom.q_i_coliforms(value) for value in q1]

q2 = [custom.q_i_pH(value) for value in q2]

q3 = [custom.q_i_DBO(value) for value in q3]

q4 = [custom.q_i_nitrogen(value) for value in q4]

q5 = [custom.q_i_phosphorus(value) for value in q5]

q6 = [custom.q_i_temperature() for value in q6]

q7 = [custom.q_i_turbidity(value) for value in q7]

q8 = [custom.q_i_total_residue(value) for value in q8]

for i in range(len(q9_list)):
  dox, temp, code_point = q9_list[i]
  q9.append(custom.q_i_dissolved_oxigen_saturation_percentage(dox, temp, code_point))

#%%
# Dataframe with the quality values
quality = {'q1': q1,
           'q2': q2,
           'q3': q3,
           'q4': q4,
           'q5': q5,
           'q6': q6,
           'q7': q7,
           'q8': q8,
           'q9': q9}

df_quality = pd.DataFrame(quality)

df_quality

#%%
"""
Calculate WQI values from quality values
"""

w1, w2, w3, w4, w5, w6, w7, w8, w9 = 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08, 0.17 # Respective parameter weights in WQI equation

wqi = list()

for index, row in df_quality.iterrows():
  qw_i = np.prod([math.pow(row['q1'], w1), math.pow(row['q2'], w2), math.pow(row['q3'], w3), math.pow(row['q4'], w4), math.pow(row['q5'], w5), math.pow(row['q6'], w6), math.pow(row['q7'], w7), math.pow(row['q8'], w8), math.pow(row['q9'], w9)])
  wqi.append(qw_i)

WQI = {'WQI': wqi}

df_temp_wqi = pd.DataFrame(WQI)

df_final_wqi = pd.concat([df_quality, df_temp_wqi], ignore_index=False, axis=1)

df_final_wqi

#%%
"""
Final dataframe concatenating Raw Data with Quality data and WQI
"""

df_final = pd.concat([df_wqi.reset_index(level=[0, 1, 0]), df_final_wqi], ignore_index=False, axis=1)

df_final

#%%
# Cast object to date values in 'Data Coleta' column

df_final['Data Coleta'] = pd.to_datetime(df_final['Data Coleta'], format='%d/%m/%Y')

#%%
# Sort values by date ('Data Coleta')

df_final = df_final.sort_values('Data Coleta')

df_final.reset_index(drop=True, inplace=True)

df_final

df_final.info()

#%%
# Debug and validate df_final info
df_final.describe()

#%%
df_final.to_csv("../data/processed/tabela_completa.csv", sep=";")

#%%