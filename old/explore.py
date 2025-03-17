# %% Imports
import os
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyedflib
import seaborn as sns
from mne.io import read_raw_edf
from scipy.io import loadmat

from utils.utils import NeuralData, bandpass_filter, import_config

# %% Import config
cfg = import_config()

# %% Import/load neural data
subj_dir = cfg.dir.data + "/" + cfg.dir.subjects[-2]
if not cfg.switches.load_data:
    # Import a single subject EDF
    print("Importing data...")
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

    # Organize data into a dataclass
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
    f_ = f.split("_")
    horizon = [_ for _ in f_ if "horizon" in _]
    # extract number from string
    horizon = horizon[0].split("horizon")[1]
    h = horizon[0]
    b = horizon[1]
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
    choice_position = [
        _[i]["choice_position"][0][0][0].tolist() for i in range(n_trials)
    ]
    reward = [_[i]["feedback"][0][0][0][0] for i in range(n_trials)]
    horizon = [h] * n_trials
    block = [b] * n_trials
    # TODO: Mouse trajectories, etc.
    df = pd.DataFrame(
        {
            "horizon": horizon,
            "block": block,
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
# %%
# -1 = left, 1 = right
choices = bhvs[2]["choice_position"].tolist()  # add this to dataclass
choices = [_[0] for _ in choices]
choices = [1 if _ > 0 else -1 for _ in choices]
chose_left = [1 if _ == -1 else 0 for _ in choices]
chose_right = [1 if _ == 1 else 0 for _ in choices]
left_bigger = [
    1 if _ == -1 and bhvs[2]["stimuli"][i][0] > bhvs[2]["stimuli"][i][1] else 0
    for i, _ in enumerate(choices)
]
# chose_left_and_left_big = [ for i, _ in enumerate(bhvs[2]["stimuli"].tolist())]
chose_left_and_left_big = [
    1 if _ == 1 and left_bigger[i] == 1 else 0 for i, _ in enumerate(chose_left)
]
chose_right_and_right_big = [
    1 if _ == 1 and left_bigger[i] == 0 else 0 for i, _ in enumerate(chose_right)
]
chose_big = chose_left_and_left_big + chose_right_and_right_big
correct_choice = []
# %%
pulse_channel = np.where(data.channels == "TRIG")[0][0]
pulses = data.data[pulse_channel, :]
# plot pulses
plt.figure(figsize=(20, 5))
plt.plot(pulses)
plt.xlim(0.57e7, 0.73e7)
plt.ylim(-1500, 10)
plt.title("Pulses H2_B1")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
# %%
plt.figure()
plt.plot(pulses)
offset = 0.10e7
plt.xlim(0.57e7 + offset, 0.60e7 + offset)
plt.ylim(-1500, 10)
plt.title("Pulses H2_B1")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
# %% Horizon 2 Block 1
np.unique(pulses)
block_start, block_end = 5700000, 7300000
p = pulses[block_start:block_end]
plt.figure(figsize=(20, 5))
plt.plot(p)
plt.title("Pulses H2_B1")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.savefig(f"{cfg.dir.save}/pulses.png")

start_time_neural = np.where(p < -200)[0][0]
start_time_neural += block_start
start_time_neural_s = start_time_neural / data.sf
signal2 = data.data[:, start_time_neural:]

_ = bhvs[0]["start_time"].values
# convert to seconds
start_times = _ / 1e3  # seconds
start_times_minutes = start_times / 60
# convert start times to samples
start_times_samples = np.round(start_times * data.sf).astype(int)

# forward difference
start_times_samples_diff = np.diff(start_times_samples)
start_times_samples = np.append(
    start_times_samples,
    np.round(start_times_samples[-1] + start_times_samples_diff.mean()).astype(int),
)

# bandpass filter signal2
signal2 = bandpass_filter(signal2, data.sf, 200, 500)

signal2_epoched = []
for previous, current in zip(start_times_samples, start_times_samples[1:]):
    _ = signal2[:, previous:current]
    signal2_epoched.append(_)

sa = []
for trial in signal2_epoched:
    _ = [np.median(abs(e)) for e in trial]
    sa.append(_)
sa = np.array(sa)
sa_mean = np.mean(sa, axis=0)  # mean across trials
sa_std = np.std(sa, axis=0)  # std across trials

plt.figure()
plt.plot(sa[:, :10])
plt.title("Spectral Amplitude (200-500Hz; H2_B1)")
plt.xlabel("Trials")
plt.ylabel("Spectral Amplitude")
plt.savefig(f"{cfg.dir.save}/sa.png")

nt = ["NT1", "NT2", "NT3"] * (sa.shape[0] // 3)
sa_df = pd.DataFrame(sa)
# add nt column
sa_df["nt"] = nt

for i in range(sa.shape[1]):
    plt.figure()
    sns.boxplot(x="nt", y=i, data=sa_df)
    plt.title(f"Channel {i}")
    plt.ylabel("Spectral Amplitude")
    plt.xlabel("Trial within episode")
    plt.savefig(f"{cfg.dir.save}/channel_{i}.png")

# %%
