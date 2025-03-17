from scipy.io import loadmat
import pandas as pd
import numpy as np

from main import import_config
import seaborn as sns
import matplotlib.pyplot as plt
#  ev_code=[10,30,50,80,85] 30/50 = disappearance of first then second stimuli

file_dir = '/home/maikito/Documents/rl_consequential/project_output/newSubj_220506.mat'
data = loadmat(file_dir)
cfg = import_config()


def block2horizon(block):
    blocks = np.array([1, 2, 3, 4, 5, 6])  # hardcoded block2horizon mapping
    horizons = np.array([0, 1, 1, 2, 2, 2])
    return horizons[np.where(blocks == block)][0]


def blocks2trials(blocks):
    trial_nums = np.arange(1, 300)
    end_idx = [ind for ind, b in enumerate(blocks[:-1]) if blocks[ind+1] != b]
    end_idx.append(len(blocks) - 1)
    trials = []
    for i, ind in enumerate(end_idx):
        if i == 0:
            trials.append(trial_nums[:ind+1])
        elif 0: #i == (len(end_idx)-1):
            trials.append(trial_nums[:(len(blocks)-ind-1)])
        else:
            trials.append(trial_nums[:(ind-end_idx[i-1])])
    return np.concatenate(trials)


subject_dfs = []
for n_sub in range(data['decision'].shape[1]):
    big, right, performance, stimuli, orders = [], [], [], [], []
    MT, RT, diff, err, eyesP, eyesX, eyesY, mvOff, mvOn, peakV, tPeakV, eventsTime = [], [], [], [], [], [], [], [], [], [], [], []
    eye_data_max_len = min([h.shape[0] for h in data['eyesP'][:, n_sub]])
    print(eye_data_max_len)
    for h in range(data['decision'].shape[0]):
        big.append(data['decision'][h, n_sub])
        right.append(data['choice'][h, n_sub])
        performance.append(data['Performance'][h, n_sub])
        stimuli.append(data['Stimuli'][h, n_sub])
        orders.append(data['OrderTask'][h, n_sub])

        MT.append(data['MT'][h, n_sub])
        RT.append(data['RT'][h, n_sub])
        diff.append(data['TrialDiff'][h, n_sub])
        err.append(data['err_trial'][h, n_sub])
        # max_ind = min([d.shape[0] for d in [data['eyesP'][h, n_sub], data['eyesX'][h, n_sub], data['eyesY'][h, n_sub]]])
        eyesP.append(data['eyesP'][h, n_sub][:eye_data_max_len, :])
        eyesX.append(data['eyesX'][h, n_sub][:eye_data_max_len, :])
        eyesY.append(data['eyesY'][h, n_sub][:eye_data_max_len, :])
        mvOff.append(data['mvOff'][h, n_sub])
        mvOn.append(data['mvOn'][h, n_sub])
        peakV.append(data['peakVel'][h, n_sub])
        tPeakV.append(data['tPeakVel'][h, n_sub])
        eventsTime.append(data['eventsTime'][h, n_sub])

    big, right, orders = np.concatenate(big), np.concatenate(right), np.concatenate(orders)
    performance, stimuli = np.concatenate(performance), np.concatenate(stimuli)
    MT = np.concatenate(MT)
    RT = np.concatenate(RT)
    diff = np.concatenate(diff)
    err = np.concatenate(err)
    eyesP, eyesX, eyesY = np.concatenate(eyesP, axis=1), np.concatenate(eyesX, axis=1), np.concatenate(eyesY, axis=1)
    eye_transform = lambda l: [ts for ts in l.T]
    eyesP, eyesX, eyesY = eye_transform(eyesP), eye_transform(eyesX), eye_transform(eyesY)
    mvOff, mvOn = np.concatenate(mvOff), np.concatenate(mvOn)
    peakV, tPeakV = np.concatenate(peakV), np.concatenate(tPeakV)
    eventsTime = np.concatenate(eventsTime)  # ev_code=[10,30,50,80,85]; 30/50 = appearance left/right?

    # H = [block2horizon(b) for b in blocks]
    H = np.concatenate([np.zeros(100), np.ones(200), np.ones(300)*2])
    subject = np.zeros(len(big)) + n_sub + 1
    # trials = blocks2trials(blocks)
    trials = np.concatenate([np.arange(1, 101), np.arange(1, 101), np.arange(1, 101),
                             np.arange(1, 106), np.arange(1, 106), np.arange(1, 91)])

    right = np.squeeze(right)
    left_appeared_first = np.array([1 if events[1] < events[2] else 0 for events in eventsTime])
    first = np.logical_and(left_appeared_first==1, right==-1) + np.logical_and(left_appeared_first==0, right==1)

    cols = ['subject', 'horizon', 'order', 'trial', 'nte', 'big', 'right', 'first', 'performance', 'rt', 'mt', 'diff',
            'err', 'eyesP', 'eyesX', 'eyesY', 'mvOn', 'mvOff', 'peakV', 'tPeakV']

    df = pd.DataFrame(columns=cols)
    df['subject'] = subject
    df['horizon'] = H
    df['order'] = orders
    df['trial'] = trials
    df['nte'] = np.concatenate([np.ones(100), np.array([1, 2] * 100), np.array([1, 2, 3] * 100)])
    df['big'] = big
    df['right'] = right
    df['first'] = first
    df['performance'] = performance
    df['rt'] = RT
    df['mt'] = MT
    df['diff'] = diff
    df['err'] = err
    df['eyesP'], df['eyesX'], df['eyesY'] = eyesP, eyesX, eyesY
    df['mvOn'], df['mvOff'] = mvOn, mvOff
    df['peakV'], df['tPeakV'] = peakV, tPeakV

    subject_dfs.append(df)

all_subjects = pd.concat(subject_dfs)

all_subjects.loc[all_subjects['big'].isna(), 'right'] = np.nan
all_subjects.loc[all_subjects['big'].isna(), 'first'] = np.nan

all_subjects.loc[0,'big']

all_subjects.to_csv('/home/maikito/Documents/rl_consequential/project_output' + '/' + 'barcelona_data.csv')

df = all_subjects
#%% Learning Time
# Learning time defined as episode# after which participant executes correct policy for at least 9/10 subsequent trials
# Error trials & most difficult trials are excluded


# df = pd.read_csv('/home/maikito/Documents/rl_consequential/project_output' + '/' + 'barcelona_data.csv')
# def calculate_learning_time(df: pd.DataFrame) -> list:
#     """
#     Function to calculate learning times for all horizons and subjects
#     :param df: dataframe containing all subjects data.
#     :return: list of lists. Each sublist contains the H0, H1, and H2 learning times, quantified as the trial # after
#     which the participant executes teh correct policy for at least 9/10 subsequent trials (exluding error trials
#     and most difficulty trials).
#     """
#     subject_learning_times = []
#     for s in df['subject'].unique():
#         subject_learning_time = []
#         for h in df['horizon'].unique():
#             df_ = df.loc[(df['subject'] == s) & (df['horizon'] == h) & (df['err'] == 0) & (df['diff'] != 0.99) &
#                          df['performance'].notna()]
#             # perf = [e for e in np.array(df_[['performance']]) if np.isnan(e) == 0]  # extract trial# as well
#             perf = df_[['trial', 'performance']].reset_index(drop=True)
#             for i, p in perf.iterrows():
#                 print(p)
#                 if len(np.where(perf.loc[i:(i + 10), 'performance'] == 1)[0]) >= 9:
#                     print('a')
#                     learning_time = perf.loc[i, 'trial']
#                     break
#                 if i == (len(perf) - 11):
#                     print('b')
#                     learning_time = -1
#                     break
#             subject_learning_time.append(learning_time)
#         subject_learning_times.append(subject_learning_time)
#     return subject_learning_times


# learning_times = calculate_learning_time(df)


#%% Plotting


for h in df['horizon'].unique():
    fig, ax = plt.subplots(3, 6, figsize=(30, 15))
    fig.suptitle(f'Participant Decisions (H{str(h)})')
    for s, ax_ in zip(df['subject'].unique(), ax.flat):
        df_ = df.loc[(df['subject'] == s) & (df['horizon'] == h)]
        sns.scatterplot(x='trial', y='big', hue='nte', data=df_, ax=ax_, palette='bright')
    plt.tight_layout()
    save_name = f'participant_decisions_h{str(int(h))}.svg'
    plt.savefig(cfg.dirs.save + '/' + save_name, format='svg')
    plt.show()


#%%
