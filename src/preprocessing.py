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