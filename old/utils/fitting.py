import numpy as np
import pandas as pd

from main import import_config
from utils import learning_time, calculate_learning_time, generate_random_sequence, \
    binary_cross_correlation
from utils import q
from hyperopt import hp, fmin, tpe, Trials


def correct_decisions(ts):
    correct = np.array([0, 1] * (len(ts) // 2))  # assumes H1
    return correct


def import_stimuli():
    block_1_stimuli = 'hor1_block1.txt'
    block_2_stimuli = 'hor1_block2.txt'
    col_names = ['trial', 'stim1', 'stim2', 'stim3', 'stim4']
    stim_1 = pd.read_csv(cfg.dirs.save + '/' + block_1_stimuli, header=None)
    stim_2 = pd.read_csv(cfg.dirs.save + '/' + block_2_stimuli, header=None)
    stim_1 = stim_1[0].str.split(' ', expand=True)
    stim_2 = stim_2[0].str.split(' ', expand=True)
    stim_1.set_axis(col_names, axis=1, inplace=True)
    stim_2.set_axis(col_names, axis=1, inplace=True)
    stimuli = pd.concat([stim_1, stim_2], ignore_index=True)
    stimuli = stimuli.astype(float)
    return stimuli


def objective(x):
    """Objective function to minimize"""
    good_idx = np.where(np.isnan(x['human_decisions']) == 0)[0]
    bcc = binary_cross_correlation(human_decisions[good_idx], agent_decisions[good_idx])
    learning_time_diff = x['human_learning_time'] - x['agent_learning_time']
    if np.isnan(x['human_learning_time']) and np.isnan(x['agent_learning_time']):
        normalized_learning_time_diff = 0
    elif np.isnan(x['human_learning_time']) != np.isnan(x['agent_learning_time']):  # XOR
        normalized_learning_time_diff = 1
    else:
        normalized_learning_time_diff = learning_time_diff / (len(human_decisions) - 11)  # values: [0, 1]

    normalized_bcc = abs(bcc[0]) if bcc[0] > 0 else 0  # values: [0, 1]
    normalized_bcc = (normalized_bcc - 1) * -1  # now 0 indicates perfect correlation, 1 indicates negative or none
    f = np.mean([normalized_learning_time_diff, normalized_bcc])
    return f


def import_data(file_path):
    data = pd.read_csv(file_path)
    return data


cfg = import_config()
df = import_data(cfg.dirs.save + '/barcelona_data.csv')
human_learning_times = calculate_learning_time(df)

human_decisions = df.loc[(df['subject'] == 1) & (df['horizon'] == 1), 'big'].to_numpy()
agent_decisions = generate_random_sequence(len(human_decisions))
human_learning_time = human_learning_times[0][1]
agent_learning_time = learning_time(correct_decisions(agent_decisions))
x_dict = {'human_decisions': human_decisions, 'agent_decisions': agent_decisions,
          'human_learning_time': human_learning_time, 'agent_learning_time': agent_learning_time}
f_x = objective(x_dict)
print(f_x)

stimuli = import_stimuli()
stimuli.to_csv(f'{cfg.dirs.save}/real_stimuli.txt')
# Proof of concept
a, g, e = 0.5, 1, 0.3
# result = q(cfg.dirs.real_stimuli, reward_type='stimuli', alpha=a, gamma=g, epsilon=e,
#            constant_epsilon=False, constant_alpha=False, optimistic=(True, 'unbiased', 1))

x2 = {'a': 0.5, 'g': 1, 'e': 0.3, 'human_decisions': human_decisions,
      'human_learning_time': human_learning_time, 'stimuli': stimuli}
result2 = q(x2['stimuli'], reward_type='stimuli', alpha=x2['a'], gamma=x2['g'], epsilon=x2['e'],
            constant_epsilon=False, constant_alpha=False, optimistic=(True, 'unbiased', 1))


def objective2(x):
    """Objective function to minimize"""
    result = q(x['stimuli'], reward_type='stimuli', alpha=x['a'], gamma=x['g'], epsilon=x['e'],
               constant_epsilon=False, constant_alpha=False, optimistic=(True, 'unbiased', 1))
    rl_decisions = result['action'].to_numpy()
    agent_learning_time = learning_time(correct_decisions(rl_decisions))
    good_idx = np.where(np.isnan(x['human_decisions']) == 0)[0]
    # print(result['action'])
    bcc = binary_cross_correlation(human_decisions[good_idx], rl_decisions[good_idx])
    learning_time_diff = x['human_learning_time'] - agent_learning_time
    if np.isnan(x['human_learning_time']) and np.isnan(agent_learning_time):
        normalized_learning_time_diff = 0
    elif np.isnan(x['human_learning_time']) != np.isnan(agent_learning_time):  # XOR
        normalized_learning_time_diff = 1
    else:
        normalized_learning_time_diff = learning_time_diff / (len(human_decisions) - 11)  # values: [0, 1]

    normalized_bcc = abs(bcc[0]) if bcc[0] > 0 else 0  # values: [0, 1]
    normalized_bcc = (normalized_bcc - 1) * -1  # now 0 indicates perfect correlation, 1 indicates negative or none
    f = np.mean([normalized_learning_time_diff, normalized_bcc])
    return f


def objective_simple_q(x):  # implement objective2 except repeat n times and take the mean
    """Objective function to minimize"""
    f = []
    for n in range(x['n_rep']):

        result = q(x['stimuli'], reward_type='stimuli', alpha=x['a'], gamma=x['g'], epsilon=x['e'],
                   constant_epsilon=False, constant_alpha=False, optimistic=(True, 'unbiased', 1))
        rl_decisions = result['action'].to_numpy()
        agent_learning_time = learning_time(correct_decisions(rl_decisions))
        good_idx = np.where(np.isnan(x['human_decisions']) == 0)[0]
        # print(result['action'])
        bcc = binary_cross_correlation(human_decisions[good_idx], rl_decisions[good_idx])
        learning_time_diff = x['human_learning_time'] - agent_learning_time
        if np.isnan(x['human_learning_time']) and np.isnan(agent_learning_time):
            normalized_learning_time_diff = 0
        elif np.isnan(x['human_learning_time']) != np.isnan(agent_learning_time):  # XOR
            normalized_learning_time_diff = 1
        else:
            normalized_learning_time_diff = learning_time_diff / (len(human_decisions) - 11)  # values: [0, 1]

        normalized_bcc = abs(bcc[0]) if bcc[0] > 0 else 0  # values: [0, 1]
        normalized_bcc = (normalized_bcc - 1) * -1  # now 0 indicates perfect correlation, 1 indicates negative or none
        f.append(np.mean([normalized_learning_time_diff, normalized_bcc]))
    return np.mean(f)


# Fit
space = hp.choice('x', [
    {
        'a': hp.uniform('a', 1E-6, 1),
        'g': 1,  # hp.uniform('g', 0, 1),
        'e': hp.uniform('e', 0, 1),
        'human_decisions': human_decisions,
        'human_learning_time': human_learning_time,
        'stimuli': stimuli
    }
])

trials = Trials()
best = fmin(objective2, space, algo=tpe.suggest, max_evals=100, trials=trials)
# best: {'a': 0.6070645246790454, 'e': 0.27606031430986566, 'x': 0}
