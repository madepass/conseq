# %% Imports
import os

from mne.io import read_raw_edf
from scipy.io import loadmat

from utils.utils import import_config

# %% Import config
cfg = import_config()

# %% Import data

# Import a single subject EDF
subj_dir = cfg.dir.data + "/" + cfg.dir.subjects[-1]
fn = [f for f in os.listdir(subj_dir) if f.endswith(".EDF")][0]
data = read_raw_edf(subj_dir + "/" + fn, preload=True, verbose=False)
raw_data = data.get_data()
info = data.info
sf = info["sfreq"]
channels = data.ch_names
pulses = raw_data[257, :]
print(
    f"{raw_data.shape[1] / sf / 60:.2f} minutes of data from {len(channels)} channels at {sf} Hz"
)
# Import a single subject behavior
mat_file_name = "subjsHospClinic_221006.mat"
data_behavior = loadmat(f"{data_dir}/{mat_file_name}")
# allEvents & eventsTime, err_trial
allEvents = data_behavior["allEvents"][0][0]
# allEvents: shape = n_trials x n_events; events: [cross wire, left disappears, right disappears, go*, selection*]
eventsTime = data_behavior["eventsTime"][0][0]
err_trial = data_behavior["err_trial"][0][0]
rt = data_behavior["RT"][0][0]

data_behavior["err_trial"][1][0].shape
