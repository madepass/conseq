import numpy as np
import pandas as pd
import random
import yaml
from ypstruct import struct
from utils import q
from ast import literal_eval


#  https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def import_config():
    """ Imports & parses config YAML file. First two levels are MATLAB-like structs"""
    with open('config.yaml', 'r') as ymlFile:  # put in function
        cfg = struct(yaml.safe_load(ymlFile))
    for k in cfg:
        if type(cfg[k]) == dict:
            cfg[k] = struct(cfg[k])
    return cfg


def generate_stimuli(horizon: int, n_episodes: str, save_dir: str):
    from utils import horizon_1
    data = np.zeros((n_episodes * 2, 5))
    stimuli = horizon_1(n_episodes)
    trials = np.arange(1, len(stimuli) + 1)
    data[:, 0] = trials
    data[:, 1:] = stimuli
    if horizon == 1:
        first = [random.sample([0, 1], 2) if i % 2 == 0 else random.sample([0, 1], 2) + random.sample([0, 1], 2) for
                 i, t in enumerate((trials))]
        right = [random.sample([0, 1], 2) if i % 2 == 0 else random.sample([0, 1], 2) + random.sample([0, 1], 2) for
                 i, t in enumerate((trials))]
    else:
        assert 'fuck you'
    df = pd.DataFrame(data=data, columns=['trial', 'stim1', 'stim2', 'stim3', 'stim4'])
    df['first'] = first
    df['right'] = right
    save_name = save_dir + '/' + 'stimuli.txt'
    df.to_csv(save_name)
    print('Stimuli generated.')


def linear_decay(current_step: int, total_steps: int, final_val, initial_val=0.9) -> float:
    r = np.max([(total_steps - current_step) / total_steps, 0])
    updated = (initial_val - final_val) * r + final_val
    return updated


def q_learning(stimuli_path: str, reward_type='stimuli', alpha=0.2, gamma=0.8, epsilon=0.2, constant_epsilon=True,
               constant_alpha=True, optimistic=(False, 'biased'), verbose=True):
    df = pd.read_csv(stimuli_path + '/' + 'stimuli.txt')
    df.head()

    state_space = np.array([0, 1])  # NT, 1 or 2 for Horizon 1
    action_space = np.array([0, 1])  # SMALL or BIG

    q_table = np.zeros([len(state_space), len(action_space)])
    if optimistic[0]:
        if optimistic[1] == 'biased':
            q_table[:, 1] += 5
        else:
            q_table += 5

    nH = 1

    # For plotting metrics
    actions = []
    presented_stimuli = []
    rewards = []
    greedy = []
    states = [0, 1] * (1 + len(df) // 2)
    q_values = np.zeros((q_table.shape[0], q_table.shape[1], len(df)))

    for t in range(0, df.shape[0]):
        state = states[t]  # df.iloc[t,:]
        epochs, penalties, reward, = 0, 0, 0

        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_space)  # Explore action space
            greedy.append(0)
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
            greedy.append(1)
        actions.append(action)

        # next_state, reward, done, info = env.step(action)
        next_state = states[t + 1]
        if any(np.isnan(df.iloc[t, :])):  # state 0, first trial within episode
            stimuli = df.iloc[t, 1:3]
            stimuli_ind = np.argmax(stimuli) if action else np.argmin(stimuli)
        else:  # second trial within episode
            stimuli = df.iloc[t, 3:5] if stimuli_ind else df.iloc[t, 1:3]
        presented_stimuli.append(stimuli)

        if reward_type == 'stimuli':
            reward = max(stimuli) if action else min(stimuli)
            rewards.append(reward)
        elif reward_type == 'gloria':
            if any(np.isnan(df.iloc[t, :])):  # NT1
                stimuli2 = df.iloc[t + 1, 3:5] if stimuli_ind else df.iloc[t + 1, 1:3]
                reward = np.mean(stimuli2) - np.mean(stimuli)
            else:  # NT2
                # reward = max(stimuli) if action else min(stimuli) # gloria
                reward = (max(stimuli) - min(stimuli)) if action else (min(stimuli) - max(stimuli))  # mad
                reward *= 2  # mad
            rewards.append(reward)
        else:
            raise ValueError('reward_type not recognized...')

        q_values[:, :, t] = q_table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        if any(np.isnan(df.iloc[t, :])):  # if NT1
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        else:  # NT2
            new_value = (1 - alpha) * old_value + alpha * reward
        q_table[state, action] = new_value

        # print(epsilon)
        if not constant_epsilon:
            # epsilon *= 0.99
            epsilon = linear_decay(t, 100, final_val=epsilon)
        if not constant_alpha:
            alpha = linear_decay(t, df.shape[0], final_val=alpha)

    if verbose: print("Training finished.\n")

    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli], greedy,
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'greedy', 'action', 'action*', 'reward'])

    q_values = np.round(q_values, decimals=3)
    q_values = np.transpose(q_values, ((2, 0, 1)))
    q_values = pd.DataFrame(data=[[x] for x in q_values])
    result['q'] = q_values
    result.to_csv(stimuli_path + '/' + 'q_learning_result.txt')
    return result


def plot(cfg):
    from utils import action_scatter_q
    action_scatter_q(cfg, 'q_learning_result.txt', 'actions_q.svg')
    # action_scatter_parallel(cfg, 'q_learning_result_parallel.txt', 'actions_q_parallel.svg')
    # action_scatter_sequential(cfg, 'q_learning_result_sequential.txt', 'actions_q_sequential.svg')
    # action_scatter_model(cfg, 'model_learning_result.txt', 'actions_model.svg')
    return 1


def main():
    print_hi('Mike')
    cfg = import_config()
    # generate_stimuli(1, cfg.gen_trials.n_episodes, cfg.dirs.save)
    # q_learning(cfg.dirs.save)  # deprecated, can delete this line and above function
    # q_original(cfg.dirs.save, reward_type='stimuli', optimistic=(True, 'biased', 5), constant_epsilon=False, epsilon=0.3, gamma=1, constant_alpha=False, alpha=0.5) # also deprecated
    q(cfg.dirs.stimuli, reward_type=cfg.q.reward_type, optimistic=literal_eval(cfg.q.optimistic),
      constant_epsilon=cfg.q.constant_epsilon, epsilon=cfg.q.epsilon,
      gamma=cfg.q.gamma, constant_alpha=cfg.q.constant_alpha, alpha=cfg.q.alpha)
    # q_parallel(cfg.dirs.save, reward_type=cfg.qp.reward_type, optimistic=literal_eval(cfg.qp.optimistic),
    #            constant_epsilon=cfg.qp.constant_epsilon, epsilon=cfg.qp.epsilon, gamma=cfg.qp.gamma,
    #            constant_alpha=cfg.qp.constant_alpha, alpha=cfg.qp.constant_alpha)
    # q_sequential(cfg.qs.hypotheses, cfg.qs.n, cfg.dirs.save, reward_type=cfg.qs.reward_type,
    #              optimistic=literal_eval(cfg.qs.optimistic),
    #              constant_epsilon=cfg.qs.constant_epsilon, epsilon=cfg.qs.epsilon, gamma=cfg.qs.gamma,
    #              constant_alpha=cfg.qs.constant_alpha, alpha=cfg.qs.alpha)
    # model(cfg, n_to_update=cfg.qm.n_to_update)
    plot(cfg)


if __name__ == '__main__':
    main()
