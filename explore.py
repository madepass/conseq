# %% Magics
%load_ext autoreload
%autoreload 2

# %% Imports
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyedflib
from mne.io import read_raw_edf
from scipy.io import loadmat

from utils.utils import NeuralData, import_config

# %% Import config
cfg = import_config()

# %% Import/load neural data
if not cfg.switches.load_data:
    # Import a single subject EDF
    print("Importing data...")
    subj_dir = cfg.dir.data + "/" + cfg.dir.subjects[-2]
    fn = [f for f in os.listdir(subj_dir) if f.endswith(".EDF")][0]
    f = pyedflib.EdfReader(subj_dir + "/" + fn)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sf = f.samplefrequency(0)
    pulse = f.readSignal(257)
    signals_to_read = np.arange(1, n, 10)
    signals_to_read = np.append(signals_to_read, 257)
    sigs = np.zeros((len(signals_to_read), f.getNSamples()[0]))
    for i, s in enumerate(signals_to_read):
        print(f"Reading signal {i+1} of {len(signals_to_read)}...")
        sigs[i, :] = f.readSignal(s)
    f._close()
    labels = np.array(signal_labels)[signals_to_read]

    # %% Organize data into a dataclass
    data = NeuralData(sigs, sf, labels, subj_dir, fn)
    # save dataclass
    print("Saving data...")
    np.savez_compressed(f"{subj_dir}/data.npz", data=data)
else:  # %% Load dataclass
    print("Loading data...")
    data = np.load(f"{subj_dir}/data.npz", allow_pickle=True)["data"].item()
    print("Data loaded.")

# %% Import/load behavior data
# list files with .mat extension
mat_files = [f for f in os.listdir(subj_dir) if f.endswith(".mat")]
bhvs = []
for f in mat_files:
    bhv = loadmat(f"{subj_dir}/{f}")["bhv"]
    n_trials = len(bhv["BehavioralCodes"][0])
    # allEvents & eventsTime, err_trial
    # allEvents: shape = n_trials x n_events; events: [cross wire, left disappears, right disappears, go*, selection*]
    _ = bhv["BehavioralCodes"][0]
    event_codes = [
        _[i]["CodeNumbers"][0][0].flatten().tolist() for i in range(n_trials)
    ]
    event_times = [_[i]["CodeTimes"][0][0].flatten().tolist() for i in range(n_trials)]
    start_time = [_[0][0] for _ in bhv["AbsoluteTrialStartTime"][0]]
    error_trials = [_[0][0] for _ in bhv["TrialError"][0]]

    _ = bhv["UserVars"][0]
    stimuli = [_[i]["presented_stimuli"][0][0][0].tolist() for i in range(n_trials)]
    choice_position = [_[i]["choice_position"][0][0][0].tolist() for i in range(n_trials)]
    reward = [_[i]["feedback"][0][0][0][0] for i in range(n_trials)]
    # TODO: Mouse trajectories, etc.
    df = pd.DataFrame(
        {
            "event_codes": event_codes,
            "event_times": event_times,
            "start_time": start_time,
            "error_trials": error_trials,
            "stimuli": stimuli,
            "choice_position": choice_position,
            "reward": reward,
        }
    )
    bhvs.append(df)

# TODO: make behavior dataclass w/ postinit to calculate correct/incorrect, etc.
#%%
# -1 = left, 1 = right
choices = bhvs[2]["choice_position"].tolist()  # add this to dataclass
choices = [_[0] for _ in choices]
choices = [1 if _ > 0 else -1 for _ in choices]
plt.plot(choices)
# %%
pulse_channel = np.where(data.channels == "TRIG")[0][0]
pulses = data.data[pulse_channel, :]
# plot pulses
plt.plot(pulses)
