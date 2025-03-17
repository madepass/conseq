import pandas as pd
import numpy as np
import random
from ast import literal_eval
from pathlib import Path


#  TODO move helper functions in this file to learning_utils.py


def linear_decay(current_step: int, n_steps: int, initial_val=0.6, final_val=0) -> float:
    r = np.max([(n_steps - current_step) / n_steps, 0])
    decayed = (initial_val - final_val) * r + final_val
    return decayed


def q_original(stimuli_path: str, reward_type='gloria', alpha=0.2, gamma=0.8, epsilon=0.2, constant_epsilon=True,
               constant_alpha=True, optimistic=(False, 'biased', 5), verbose=True):
    df = pd.read_csv(stimuli_path + '/' + 'stimuli.txt')
    df.head()

    state_space = np.array([0, 1, 2])  # S1 = NT1, S2/S3 = NT2 low/high reward respectively
    action_space = np.array([0, 1])  # SMALL or BIG

    q_table = np.zeros([len(state_space), len(action_space)])
    if optimistic[0]:
        if optimistic[1] == 'biased':
            q_table[:, 1] += optimistic[2]
        else:
            q_table += optimistic[2]

    nH = 1

    # For plotting metrics
    actions = []
    presented_stimuli = []
    rewards = []
    greedy = []
    # states = [0, 1] * (1 + len(df) // 2)  # commented ever since added third state
    states = [0]
    epsilons = []
    alphas = []
    q_values = np.zeros((q_table.shape[0], q_table.shape[1], len(df)))

    initial_epsilon = epsilon
    initial_alpha = alpha
    for t in range(0, df.shape[0]):
        state = states[t]  # df.iloc[t,:]
        epochs, penalties, reward, = 0, 0, 0
        alphas.append(alpha)
        epsilons.append(epsilon)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_space)  # Explore action space
            greedy.append(0)
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
            greedy.append(1)
        actions.append(action)

        # next_state, reward, done, info = env.step(action)
        if state == 0:
            states.append(1) if action == 1 else states.append(2)
        else:
            states.append(0)

        if np.isnan(df.iloc[t, 4]):  # state 0, first trial within episode
            stimuli = df.iloc[t, 2:4]
            stimuli_ind = np.argmax(stimuli) if action else np.argmin(stimuli)
        else:  # second trial within episode
            stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
        presented_stimuli.append(stimuli)

        if reward_type == 'stimuli':
            reward = max(stimuli) if action else min(stimuli)
            rewards.append(reward)
        elif reward_type == 'gloria':
            if any(np.isnan(df.iloc[t, :])):  # NT1
                stimuli2 = df.iloc[t + 1, 4:6] if stimuli_ind else df.iloc[t + 1, 2:4]
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
        next_state = states[t + 1]
        next_max = np.max(q_table[next_state])
        if np.isnan(df.iloc[t, 4]):  # if NT1
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        else:  # NT2
            new_value = (1 - alpha) * old_value + alpha * reward
        q_table[state, action] = new_value

        # print(epsilon)
        if not constant_epsilon:
            # epsilon *= 0.99
            epsilon = linear_decay(t + 1, 100, initial_val=initial_epsilon, final_val=0)
        if not constant_alpha:
            alpha = linear_decay(t + 1, 100, initial_val=initial_alpha, final_val=0)

    if verbose: print("Training finished.\n")

    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli], greedy,
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'greedy', 'action', 'action*', 'reward'])

    q_values = np.round(q_values, decimals=3)
    q_values = np.transpose(q_values, ((2, 0, 1)))
    q_values = pd.DataFrame(data=[[x] for x in q_values])
    result['q'] = q_values
    result['epsilon'] = epsilons
    result['alpha'] = alphas
    result.to_csv(stimuli_path + '/' + 'q_learning_result.txt')
    return result


# TODO: biased optimistic initial values doesn't work as intended for q_sequential
def initialize_q_table(action_space, state_space, optimistic):
    q_table = np.zeros([len(state_space), len(action_space)])
    if optimistic[0]:
        if optimistic[1] == 'biased':
            q_table[:, 1] += optimistic[2]
        else:
            q_table += optimistic[2]
    return q_table


def q(stimuli_path, reward_type='stimuli', alpha=0.2, gamma=0.8, epsilon=0.2, constant_epsilon=True,
      constant_alpha=True, optimistic=(False, 'biased', 5), save_result=False, fixed_decisions=False,
      verbose=True):
    """
    Epsilon-greedy reinforcement learning agent. Restructured version of q_original.
    :param fixed_decisions: (False, action_vector)
    :param save_result:
    :param stimuli_path: filepath of stimuli txt file
    :param reward_type: type of reward signal {'gloria', 'stimuli'}
    :param alpha: learning rate [0, 1]
    :param gamma: discount factor [0, 1]
    :param epsilon: exploration rate [0, 1]
    :param constant_epsilon: if False, epsilon will decay, starting at the value of epsilon until 0
    :param constant_alpha: if False, alpha will decay, starting at the value of alpha until 0
    :param optimistic: if True, initializes q-table with non-zero values. if 'biased', will bias agent towards big.
    :param verbose: if True, will print stuff
    :return: results dataframe. Also prints results to txt file
    """
    # print('hello world')
    # print(stimuli_path)
    # print(type(stimuli_path))
    if isinstance(stimuli_path, str):
        df = pd.read_csv(stimuli_path)
    else:
        df = stimuli_path.copy()  # only this uncommented for hyopt...
    #  initialize q table
    state_space = np.array([0, 1, 2])  # S1 = NT1, S2/S3 = NT2 low/high reward respectively
    action_space = np.array([0, 1])  # SMALL or BIG
    q_table = initialize_q_table(action_space, state_space, optimistic)
    #  For plotting metrics
    actions = []
    presented_stimuli = []
    rewards = []
    greedy = []
    states = [0]
    epsilons = []
    alphas = []
    if fixed_decisions:
        fixed_actions = []
    q_values = np.zeros((q_table.shape[0], q_table.shape[1], len(df)))
    initial_epsilon = epsilon
    initial_alpha = alpha
    #  Learn
    for t in range(0, df.shape[0]):
        state = states[t]  # df.iloc[t,:]
        # epochs, penalties, reward, = 0, 0, 0
        alphas.append(alpha)
        epsilons.append(epsilon)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_space)  # Explore action space
            greedy.append(0)
        else:
            action = np.random.choice(np.where(q_table[state] == q_table[state].max())[
                                          0])  # Exploit learned values: argmax w/ random tie-breaks
            greedy.append(1)
        if fixed_decisions:
            if t < len(fixed_decisions):
                action = int(fixed_decisions[t])
                print(action)
                fixed_actions.append(1)
            else:
                fixed_actions.append(0)
        actions.append(action)
        if np.isnan(df.iloc[t, 4]):  # first trial in episode
            stimuli = df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
            stimuli_ind = np.argmax(stimuli) if chose_big else np.argmin(stimuli)  # action --> chose_big
        else:  # second trial in episode
            stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
        presented_stimuli.append(stimuli)

        states.append(get_next_state(1, state, chose_big))

        reward = reward_function(1, 'stimuli', t, df, stimuli, stimuli_ind, chose_big)
        rewards.append(reward)

        q_values[:, :, t] = q_table
        old_value = q_table[state, action]
        next_state = states[t + 1]
        next_max = np.max(q_table[next_state])
        if np.isnan(df.iloc[t, 4]):  # if NT1
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        else:  # NT2
            new_value = (1 - alpha) * old_value + alpha * reward
        q_table[state, action] = new_value
        #  Decay exploitation & learning rates
        if not constant_epsilon:
            epsilon = linear_decay(t + 1, 100, initial_val=initial_epsilon, final_val=0)
        if not constant_alpha:
            alpha = linear_decay(t + 1, 100, initial_val=initial_alpha, final_val=0)

    if verbose:
        print("Training finished.\n")

    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli], greedy,
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'greedy', 'action', 'action*', 'reward'])

    q_values = np.round(q_values, decimals=3)
    q_values = np.transpose(q_values, ((2, 0, 1)))
    q_values = pd.DataFrame(data=[[x] for x in q_values])
    result['q'] = q_values
    result['epsilon'] = epsilons
    result['alpha'] = alphas
    result['chose_big'] = actions
    if fixed_decisions:
        result['fixed_action'] = fixed_actions
    if save_result:
        path = Path(stimuli_path)
        result.to_csv('q_learning_result.txt')
    return result


def q_parallel(stimuli_path: str, reward_type='gloria', alpha=0.2, gamma=0.8, epsilon=0.2, constant_epsilon=True,
               constant_alpha=True, optimistic=(False, 'biased', 5), verbose=True):
    df = pd.read_csv(stimuli_path)
    df.head()

    state_space = np.array([0, 1, 2])  # S1 = NT1, S2/S3 = NT2 low/high reward respectively
    action_space = np.array([0, 1, 2, 3, 4, 5])  # small, big, left, right, first, second # parallel

    q_table = np.zeros([len(state_space), len(action_space)])
    if optimistic[0]:
        if optimistic[1] == 'biased':
            q_table[:, 1] += optimistic[2]  # makes initial bias towards picking big option in every trial
        else:
            q_table += optimistic[2]

    nH = 1

    # For plotting metrics
    actions = []
    presented_stimuli = []
    rewards = []
    greedy = []
    # states = [0, 1] * (1 + len(df) // 2)  # commented ever since added third state
    states = [0]
    epsilons = []
    alphas = []
    chose_bigs = []
    q_values = np.zeros((q_table.shape[0], q_table.shape[1], len(df)))

    initial_epsilon = epsilon
    initial_alpha = alpha
    for t in range(0, df.shape[0]):
        state = states[t]  # df.iloc[t,:]
        # epochs, penalties, reward, = 0, 0, 0
        alphas.append(alpha)
        epsilons.append(epsilon)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_space)  # Explore action space
            greedy.append(0)
        else:
            action = np.random.choice(np.where(q_table[state] == q_table[state].max())[
                                          0])  # Exploit learned values # argmax w/ random tie-breaking
            greedy.append(1)
        actions.append(action)

        if np.isnan(df.iloc[t, 4]):  # first trial in episode
            stimuli = df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
            stimuli_ind = np.argmax(stimuli) if chose_big else np.argmin(stimuli)  # action --> chose_big
        else:  # second trial in episode
            stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
        chose_bigs.append(chose_big)
        presented_stimuli.append(stimuli)
        # if np.isnan(df.iloc[t, 4]):  # state 0, first trial within episode
        #     # stimuli = df.iloc[t, 2:4]
        #     stimuli_ind = np.argmax(stimuli) if chose_big else np.argmin(stimuli)  # action --> chose_big
        # else:  # second trial within episode
        #     stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
        # presented_stimuli.append(stimuli)

        # next_state, reward, done, info = env.step(action)
        # if action == 1:
        #     chose_big = 1
        # elif action == 2 and df.loc[t, 'right']

        states.append(get_next_state(1, state, chose_big))

        reward = reward_function(1, 'stimuli', t, df, stimuli, stimuli_ind, chose_big)
        rewards.append(reward)

        q_values[:, :, t] = q_table
        old_value = q_table[state, action]
        next_state = states[t + 1]
        next_max = np.max(q_table[next_state])
        if np.isnan(df.iloc[t, 4]):  # if NT1
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        else:  # NT2
            new_value = (1 - alpha) * old_value + alpha * reward
        q_table[state, action] = new_value

        # print(epsilon)
        if not constant_epsilon:
            epsilon = linear_decay(t + 1, 100, initial_val=initial_epsilon, final_val=0)
        if not constant_alpha:
            alpha = linear_decay(t + 1, 100, initial_val=initial_alpha, final_val=0)

    if verbose: print("Training finished.\n")

    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli], greedy,
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'greedy', 'action', 'action*', 'reward'])

    q_values = np.round(q_values, decimals=3)
    q_values = np.transpose(q_values, ((2, 0, 1)))
    q_values = pd.DataFrame(data=[[x] for x in q_values])
    result['q'] = q_values
    result['epsilon'] = epsilons
    result['alpha'] = alphas
    result['chose_big'] = chose_bigs
    result.to_csv(stimuli_path + '/' + 'q_learning_result_parallel.txt')
    return result


def q_sequential(strategies: list, n_to_evaluate: int, stimuli_path: str, reward_type='gloria', alpha=0.2,
                 gamma=0.8, epsilon=0.2,
                 constant_epsilon=True, constant_alpha=True, optimistic=(False, 'biased', 5), verbose=True):
    df = pd.read_csv(stimuli_path)

    state_space = np.array([0, 1, 2])  # S1 = NT1, S2/S3 = NT2 low/high reward respectively
    action_space = np.array([0, 1, 2, 3, 4, 5])  # small, big, left, right, first, second # parallel
    action_spaces = {'sb': np.array([0, 1]), 'lr': np.array([2, 3]), 'fs': np.array([4, 5])}
    hypo_dict = {'lr': [], 'sb': [], 'fs': []}  # hypothesis dict
    q_table_sb = initialize_q_table([0, 1], state_space, optimistic)  # small/big
    q_table_lr = initialize_q_table([2, 3], state_space, optimistic)  # left/right
    q_table_fs = initialize_q_table([4, 5], state_space, optimistic)  # first/second
    q_tables = {'lr': q_table_sb, 'sb': q_table_sb, 'fs': q_table_fs}
    # For plotting metrics
    actions = []
    presented_stimuli = []
    rewards = []
    greedy = []
    # states = [0, 1] * (1 + len(df) // 2)  # commented ever since added third state
    states = [0]
    epsilons = []
    alphas = []
    chose_bigs = []
    q_values_sb = np.zeros((q_table_sb.shape[0], q_table_sb.shape[1], len(df)))
    q_values_lr = np.zeros((q_table_lr.shape[0], q_table_lr.shape[1], len(df)))
    q_values_fs = np.zeros((q_table_fs.shape[0], q_table_fs.shape[1], len(df)))
    q_values_all_strategies = {'sb': q_values_sb, 'lr': q_values_lr, 'fs': q_values_fs}
    initial_epsilon = epsilon
    initial_alpha = alpha
    current_strategies = []
    si = 0  # "strategy index" of the strategy being employed
    all_hypotheses_tested_flag = 0
    for t in range(0, df.shape[0]):
        state = states[t]  # df.iloc[t,:]
        # epochs, penalties, reward, = 0, 0, 0
        alphas.append(alpha)
        epsilons.append(epsilon)

        print(si)
        current_strategy = strategies[si]
        current_strategies.append(current_strategy)

        if random.uniform(0, 1) < epsilon:
            action = random.choice(action_spaces[current_strategy])  # Explore action space
            greedy.append(0)
        else:
            action = np.random.choice(
                np.where(q_tables[current_strategy][state] == q_tables[current_strategy][state].max())[
                    0])  # Exploit learned values # argmax w/ random tie-breaking
            # Above line returns index...
            action = action_spaces[current_strategy][action]
            greedy.append(1)
        actions.append(action)
        if action in [0, 2, 4]:
            action_ind = 0
        else:
            action_ind = 1

        if np.isnan(df.iloc[t, 4]):  # first trial in episode
            stimuli = df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
            stimuli_ind = np.argmax(stimuli) if chose_big else np.argmin(stimuli)  # action --> chose_big
        else:  # second trial in episode
            stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
            chose_big = big_chosen(1, stimuli, action, df, t)
        chose_bigs.append(chose_big)
        presented_stimuli.append(stimuli)

        states.append(get_next_state(1, state, chose_big))

        reward = reward_function(1, 'stimuli', t, df, stimuli, stimuli_ind, chose_big)
        rewards.append(reward)

        q_values_all_strategies[current_strategy][:, :, t] = q_tables[current_strategy]
        old_value = q_tables[current_strategy][state, action_ind]
        next_state = states[t + 1]
        next_max = np.max(q_tables[current_strategy][next_state])
        if np.isnan(df.iloc[t, 4]):  # if NT1
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        else:  # NT2
            new_value = (1 - alpha) * old_value + alpha * reward
        q_tables[current_strategy][state, action_ind] = new_value

        # print(epsilon)
        if not constant_epsilon:
            epsilon = linear_decay(t + 1, 100, initial_val=initial_epsilon, final_val=0)
        if not constant_alpha:
            alpha = linear_decay(t + 1, 100, initial_val=initial_alpha, final_val=0)

        #  TODO: put below into "select_strategy" function
        if ((
                    t + 1) % n_to_evaluate == 0) and all_hypotheses_tested_flag == 0:  # and si < len(strategies):  # increment strategy index after n trials
            si += 1
        if si == len(strategies):
            all_hypotheses_tested_flag = 1
            strategy_means = np.mean(np.array(rewards).reshape(-1, n_to_evaluate), axis=1)
            best_strategy_ind = np.argmax(strategy_means)
            si = best_strategy_ind

    if verbose: print("Training finished.\n")

    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli], greedy,
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'greedy', 'action', 'action*', 'reward'])

    # q_values = np.round(q_values, decimals=3)
    # q_values = np.transpose(q_values, ((2, 0, 1)))
    # q_values = pd.DataFrame(data=[[x] for x in q_values])
    # result['q'] = q_values
    result['epsilon'] = epsilons
    result['alpha'] = alphas
    result['chose_big'] = chose_bigs
    result['hypothesis'] = current_strategies
    result.to_csv(stimuli_path + '/' + 'q_learning_result_sequential.txt')
    return result


def big_chosen(horizon: int, stimuli: pd.Series, action: int, stimuli_df: pd.DataFrame, t: int):
    chose_big = 0
    match action:
        case 1:
            chose_big = 1
        case 2:  # chose left
            ind_chosen = np.where(np.array(literal_eval(stimuli_df.loc[t, 'right'])) == 0)[0][0]
            chose_big = 1 if stimuli[ind_chosen] == max(stimuli) else 0
        case 3:  # chose right
            ind_chosen = np.where(np.array(literal_eval(stimuli_df.loc[t, 'right'])) == 1)[0][0]
            chose_big = 1 if stimuli[ind_chosen] == max(stimuli) else 0
        case 4:  # chose first
            ind_chosen = np.where(np.array(literal_eval(stimuli_df.loc[t, 'first'])) == 1)[0][0]
            chose_big = 1 if stimuli[ind_chosen] == max(stimuli) else 0
        case 5:  # chose second
            ind_chosen = np.where(np.array(literal_eval(stimuli_df.loc[t, 'first'])) == 0)[0][0]
            chose_big = 1 if stimuli[ind_chosen] == max(stimuli) else 0
    return chose_big


def reward_function(horizon: int, reward_type: str, t: int, df: pd.DataFrame, stimuli: pd.Series, stimuli_ind: int,
                    chose_big: int):
    if reward_type == 'stimuli':
        reward = max(stimuli) if chose_big else min(stimuli)
        # rewards.append(reward)
    elif reward_type == 'gloria':
        if any(np.isnan(df.iloc[t, :])):  # NT1
            stimuli2 = df.iloc[t + 1, 4:6] if stimuli_ind else df.iloc[t + 1, 2:4]
            reward = np.mean(stimuli2) - np.mean(stimuli)
        else:  # NT2
            # reward = max(stimuli) if action else min(stimuli) # gloria
            reward = (max(stimuli) - min(stimuli)) if chose_big else (min(stimuli) - max(stimuli))  # mad
            reward *= 2  # mad
        # rewards.append(reward)
    else:
        raise ValueError('reward_type not recognized...')
    return reward


def get_next_state(horizon: int, current_state: int, chose_big: int):
    if current_state == 0:
        next_state = 1 if chose_big == 1 else 2
    else:
        next_state = 0
    return next_state


def flatten(input_list):
    return [x for xs in input_list for x in xs]


def find_h1_policy(reward_table, transition_probs, hypotheses):
    expected_reward = []
    for h in hypotheses:
        r1 = reward_table[0, h[0]]
        #  r2 = P(S2|S1,action) * Reward|S2,action + P(S3|S1,action) * Reward|S3,action
        r2 = transition_probs[0, h[0]] * reward_table[1, h[1]] + transition_probs[1, h[0]] * reward_table[2, h[1]]
        r_total = r1 + r2  # expected reward per episode
        expected_reward.append(r_total)
    return expected_reward


def repeat_list_elements(a_list: list, n_repeats: int):
    return [item for item in a_list for i in range(n_repeats)]


def model(cfg, n_to_update=10):
    df = pd.read_csv(cfg.dirs.stimuli)

    state_space = np.array([0, 1, 2])  # S1 = NT1, S2/S3 = NT2 low/high reward respectively
    action_space = np.array([0, 1])  # SMALL or BIG
    hypothesis_space = [[0, 1], [1, 1]]  # small, big & big, big
    r_table = np.zeros([len(state_space), len(action_space)])
    r_counts = np.zeros([len(state_space), len(action_space)])
    s_transition_probs = np.zeros((2, 2))  # S2/S3 x action; hardcoded for H1

    # actions = []
    # actions = hypothesis_space * n_to_update
    actions = repeat_list_elements(hypothesis_space, n_to_update)
    actions = [x for xs in actions for x in xs]

    presented_stimuli = []
    rewards = []
    states = [0]
    for t in range(0, df.shape[0]):
        nt = t % 2  # 0 = nt1, 1 = nt2
        if t < n_to_update * len(flatten(hypothesis_space)):  # % n_to_update != 0:
            state = states[-1]

            action = actions[t]

            if state == 0:
                states.append(1) if action == 1 else states.append(2)
            else:
                states.append(0)

            if np.isnan(df.iloc[t, 4]):  # state 0, first trial within episode
                stimuli = df.iloc[t, 2:4]
                stimuli_ind = np.argmax(stimuli) if action else np.argmin(stimuli)
            else:  # second trial within episode
                stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
            presented_stimuli.append(stimuli)

            reward = max(stimuli) if action else min(stimuli)
            rewards.append(reward)

            r_table[state, action] += reward
            r_counts[state, action] += 1
            if nt == 0:
                s_transition_probs[states[-1] - 1, action] += 1

            continue

        else:
            if t == n_to_update * len(flatten(hypothesis_space)):
                # compute model
                r_table = r_table / (r_counts + 1e-10)  # 1e-10 prevent divide by zero
                s_transition_probs /= s_transition_probs.sum(axis=0, keepdims=True)  # column normalize
                expected_reward_per_policy = find_h1_policy(r_table, s_transition_probs, hypothesis_space)
                optimal_policy = hypothesis_space[
                    np.where(expected_reward_per_policy == np.amax(expected_reward_per_policy))[0][0]]

            # implement model
            state = states[-1]
            action = optimal_policy[0] if state == 0 else optimal_policy[1]
            actions.append(action)
            if state == 0:
                states.append(1) if action == 1 else states.append(2)
            else:
                states.append(0)
            if np.isnan(df.iloc[t, 4]):  # state 0, first trial within episode
                stimuli = df.iloc[t, 2:4]
                stimuli_ind = np.argmax(stimuli) if action else np.argmin(stimuli)
            else:  # second trial within episode
                stimuli = df.iloc[t, 4:6] if stimuli_ind else df.iloc[t, 2:4]
            presented_stimuli.append(stimuli)

            reward = max(stimuli) if action else min(stimuli)
            rewards.append(reward)

    print("Training finished.\n")

    # actions += optimal_policy * ((len(df) - len(actions)) // 2)
    states = states[:-1]
    correct_actions = [0, 1] * (len(df) // 2)
    result = pd.DataFrame(data=np.array(
        [np.arange(len(actions)) + 1, np.array(states[:len(df)]) + 1, [list(i) for i in presented_stimuli],
         actions, correct_actions, rewards]).T, columns=['T', 'NT', 'stimuli', 'action', 'action*', 'reward'])
    result.to_csv(cfg.dirs.save + '/' + 'model_learning_result.txt')
    return result


def q2learned(q_table: np.array) -> bool:
    learned = False
    if q_table[0, 0] > q_table[0, 1]:
        if q_table[2, 0] <  q_table[2, 1]:
            learned = True
    return learned

from utils import horizon_1
from utils import import_stimuli
from types import FunctionType
from hyperopt import fmin, tpe, Trials


class Agent:
    def __init__(self):
        self.n_samples = None
        self.loss_function_args = None
        self.best_params = None
        self.trials = None
        self.loss_function = None
        self.space = None
        self.algorithm = None
        self.stimuli = []
        self.learning_results = []
        self.fitting_results = []

    def __repr__(self):
        return 'Instance of Agent class'

    def __str__(self):
        return 'member of Agent class'

    def set_stimuli(self, stimuli_type='simulated'):
        if stimuli_type == 'empirical':
            stimuli = import_stimuli()
        elif stimuli_type == 'simulated':
            stimuli = horizon_1(2000)
        else:
            raise ValueError('Stimuli option must be set to either empirical or simulated.')
        self.stimuli = stimuli

    def set_learning_algorithm(self):
        pass

    def set_fitting_params(self, learning_algo: FunctionType, param_space: dict, objective: FunctionType):
        self.algorithm = learning_algo
        self.space = param_space
        self.loss_function = objective
        #  TODO: check if algorithm & space match, and print mismatches in error message

    def fit(self, n_samples: int):
        trials = Trials()
        best = fmin(self.loss_function, self.space, algo=tpe.suggest, max_evals=n_samples, trials=trials)
        self.n_samples = n_samples
        self.trials = trials
        self.best_params = best

    def learn(self, learning_algo: FunctionType, params: dict):
        learning_algo(params)