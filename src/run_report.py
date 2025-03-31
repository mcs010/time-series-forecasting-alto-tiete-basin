"""
Author: Matheus Cardoso da Silva
"""

#%%
import matplotlib.pyplot as plt
import pandas as pd

import run_experiment

#%%
# Plot series
# ==============================================================================
#set_dark_theme()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axs = plt.subplots(10, 1, figsize=(18, 26), sharex=True)

for i, s in enumerate(run_experiment.series_dict.values()):
    axs[i].plot(s, label=s.name, color=colors[i])
    axs[i].legend(loc='upper right', fontsize=14)
    axs[i].tick_params(axis='both', labelsize=8)
    axs[i].axvline(pd.to_datetime(run_experiment.end_train), color='white', linestyle='--', linewidth=1)  # End train

fig.suptitle('Series in `series_dict`', fontsize=15)
plt.savefig("../reports/figures/")
#plt.tight_layout()