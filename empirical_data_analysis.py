# %% Imports
import numpy as np
import pandas as pd


# %% Import data
data_dir = "/media/maikito/mad_mini/consequential_task/psychopy/h1_v2/data/saved/"
file_name = "mad_test1_consequential_2025-04-16_18h07.26.854.csv"
data_mad = pd.read_csv(f"{data_dir}{file_name}")
file_name = "riccardo_774594_consequential_2025-04-24_18h09.56.705.csv"
data_rm = pd.read_csv(f"{data_dir}{file_name}")

data_mad.groupby("g_block").count()
data_rm.groupby("g_block").count()

# identify useful columns
data_rm["mouse_2.time"]
data_rm["trials_within_episode.mouse_2.clicked_name"]
