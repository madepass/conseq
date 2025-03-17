""" mat2csv.py
mat to csv conversion script for the data collected at the University of Barcelona.
Ideal if you wish to load data into a Pandas dataframe
    e.g. df = pd.read_csv("/home/maikito/Documents/barcelona_data.csv")
Input: .mat file
Output: .csv file
Author: Michael DePass
"""
#%% Import packages
from scipy.io import loadmat
import pandas as pd
import numpy as np

#  ev_code=[10,30,50,80,85] 30/50 = disappearance of first then second stimuli

#%% User-defined parameters
# File path & name of the .mat file to load
mat_file_name = "/home/maikito/Documents/rl_consequential/project_output/original/newSubj_220506.mat"
# File path & name of the .csv file to save
csv_save_name = "/home/maikito/Documents/barcelona_data2.csv"


#%% Load .mat, organize in Pandas dataframe, export to CSV
def block2horizon(block):
    blocks = np.array([1, 2, 3, 4, 5, 6])  # hardcoded block2horizon mapping
    horizons = np.array([0, 1, 1, 2, 2, 2])
    return horizons[np.where(blocks == block)][0]


def blocks2trials(blocks):
    trial_nums = np.arange(1, 300)
    end_idx = [ind for ind, b in enumerate(blocks[:-1]) if blocks[ind + 1] != b]
    end_idx.append(len(blocks) - 1)
    trials = []
    for i, ind in enumerate(end_idx):
        if i == 0:
            trials.append(trial_nums[: ind + 1])
        elif 0:  # i == (len(end_idx)-1):
            trials.append(trial_nums[: (len(blocks) - ind - 1)])
        else:
            trials.append(trial_nums[: (ind - end_idx[i - 1])])
    return np.concatenate(trials)


print("Loading .mat file...")
data = loadmat(mat_file_name)
print("Writing to csv (this may take a while)...")
subject_dfs = []
for n_sub in range(data["decision"].shape[1]):
    big, right, performance, stimuli, orders = [], [], [], [], []
    MT, RT, diff, err, eyesP, eyesX, eyesY, mvOff, mvOn, peakV, tPeakV, eventsTime = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    eye_data_max_len = min([h.shape[0] for h in data["eyesP"][:, n_sub]])
    for h in range(data["decision"].shape[0]):
        big.append(data["decision"][h, n_sub])
        right.append(data["choice"][h, n_sub])
        performance.append(data["Performance"][h, n_sub])
        stimuli.append(data["Stimuli"][h, n_sub])
        orders.append(data["OrderTask"][h, n_sub])
        MT.append(data["MT"][h, n_sub])
        RT.append(data["RT"][h, n_sub])
        diff.append(data["TrialDiff"][h, n_sub])
        err.append(data["err_trial"][h, n_sub])
        eyesP.append(data["eyesP"][h, n_sub][:eye_data_max_len, :])
        eyesX.append(data["eyesX"][h, n_sub][:eye_data_max_len, :])
        eyesY.append(data["eyesY"][h, n_sub][:eye_data_max_len, :])
        mvOff.append(data["mvOff"][h, n_sub])
        mvOn.append(data["mvOn"][h, n_sub])
        peakV.append(data["peakVel"][h, n_sub])
        tPeakV.append(data["tPeakVel"][h, n_sub])
        eventsTime.append(data["eventsTime"][h, n_sub])
    big, right, orders = (
        np.concatenate(big),
        np.concatenate(right),
        np.concatenate(orders),
    )
    performance, stimuli = np.concatenate(performance), np.concatenate(stimuli)
    MT = np.concatenate(MT)
    RT = np.concatenate(RT)
    diff = np.concatenate(diff)
    err = np.concatenate(err)
    eyesP, eyesX, eyesY = (
        np.concatenate(eyesP, axis=1),
        np.concatenate(eyesX, axis=1),
        np.concatenate(eyesY, axis=1),
    )
    eye_transform = lambda l: [ts for ts in l.T]
    eyesP, eyesX, eyesY = (
        eye_transform(eyesP),
        eye_transform(eyesX),
        eye_transform(eyesY),
    )
    mvOff, mvOn = np.concatenate(mvOff), np.concatenate(mvOn)
    peakV, tPeakV = np.concatenate(peakV), np.concatenate(tPeakV)
    eventsTime = np.concatenate(
        eventsTime
    )  # ev_code=[10,30,50,80,85]; 30/50 = appearance left/right?
    H = np.concatenate([np.zeros(100), np.ones(200), np.ones(300) * 2])
    subject = np.zeros(len(big)) + n_sub + 1
    trials = np.concatenate(
        [
            np.arange(1, 101),
            np.arange(1, 101),
            np.arange(1, 101),
            np.arange(1, 106),
            np.arange(1, 106),
            np.arange(1, 91),
        ]
    )
    right = np.squeeze(right)
    left_appeared_first = np.array(
        [1 if events[1] < events[2] else 0 for events in eventsTime]
    )
    first = np.logical_and(left_appeared_first == 1, right == -1) + np.logical_and(
        left_appeared_first == 0, right == 1
    )
    cols = [
        "subject",
        "horizon",
        "order",
        "trial",
        "nte",
        "big",
        "right",
        "first",
        "performance",
        "rt",
        "mt",
        "diff",
        "err",
        "eyesP",
        "eyesX",
        "eyesY",
        "mvOn",
        "mvOff",
        "peakV",
        "tPeakV",
        "stimuli",
    ]
    df = pd.DataFrame(columns=cols)
    df["subject"] = subject
    df["horizon"] = H
    df["order"] = orders
    df["trial"] = trials
    df["nte"] = np.concatenate(
        [np.ones(100), np.array([1, 2] * 100), np.array([1, 2, 3] * 100)]
    )
    df["big"] = big
    df["right"] = right
    df["first"] = first
    df["performance"] = performance
    df["rt"] = RT
    df["mt"] = MT
    df["diff"] = diff
    df["err"] = err
    df["eyesP"], df["eyesX"], df["eyesY"] = eyesP, eyesX, eyesY
    df["mvOn"], df["mvOff"] = mvOn, mvOff
    df["peakV"], df["tPeakV"] = peakV, tPeakV
    df["stimuli"] = stimuli
    subject_dfs.append(df)
all_subjects = pd.concat(subject_dfs)
all_subjects.loc[all_subjects["big"].isna(), "right"] = np.nan
all_subjects.loc[all_subjects["big"].isna(), "first"] = np.nan
presented_stimuli = np.tile(stimuli, (len(subject_dfs), 1))
presented_stimuli = [list(stim) for stim in presented_stimuli]
all_subjects["presented_stimuli"] = presented_stimuli
all_subjects.to_csv(csv_save_name)
print("Finished writing to csv.")

subject_dfs[0][subject_dfs[0].horizon == 1].head()
# load data
data2 = pd.read_csv(csv_save_name)
