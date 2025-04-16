"""
Figure out which values of gain (G) to use for Consequential Task v2
Also, which values of difficulty (d).
"""
# %%
import random
import numpy as np
import pandas as pd

# %% Functions
def generate_stimuli(state, m, d, g):
    if state == 0:
        stimuli = [m - d / 2, m + d / 2]
    elif state == 1:
        stimuli = [m - g - d / 2, m - g + d / 2]
    else:  # state == 2
        stimuli = [m + g - d / 2, m + g + d / 2]
    random.shuffle(stimuli)  # randomize left or right position on screen
    return stimuli


def generate_cumulative_rewards(gs, difficulty):
    data = []
    for g in gs:
        d_data = []
        for d in difficulty:
            stimuli_trial_1 = generate_stimuli(0, m, d, g)
            stimuli_trial_2_bb = generate_stimuli(1, m, d, g)
            stimuli_trial_2_sb = generate_stimuli(2, m, d, g)
            small_big_cum_reward = min(stimuli_trial_1) + max(stimuli_trial_2_sb)
            big_big_cum_reward = max(stimuli_trial_1) + max(stimuli_trial_2_bb)
            cum_rewards = (round(small_big_cum_reward, 2), round(big_big_cum_reward, 2))
            d_data.append(cum_rewards)
        data.append(d_data)
    df = pd.DataFrame(
        data, columns=list(map(str, difficulty)), index=list(map(str, gs))
    )
    print("(cum reward small-big, cum reward big-big)")
    print(df, "\n")
    difference_df = df.applymap(lambda tup: tup[0] - tup[1])
    print("(cum reward small-big) - (sum reward big-big)")
    print(difference_df)


def min_and_max_stimuli(ms, difficulty, g):
    max_d = max(difficulty)
    min_stim = min(ms) - max(gs) - max_d / 2
    max_stim = max(ms) + max(gs) + max_d / 2
    print(f"({min_stim}, {max_stim})")


def get_ms(g, difficulty, n):
    ms = np.linspace(abs(g) + max(difficulty) / 2, 1 - abs(g) - max(difficulty) / 2, 5)
    return ms


# %% Cumulative rewards for current difficulty values
gs = [-0.3, 0, 0.1, 0.2, 0.3]
difficulty = [0.01, 0.05, 0.1, 0.15, 0.2]
# make sure this equation for ms is equal to psychopy implementation
# g should be abs(g), otherwise incorrect for negative g
m = 0.5
generate_cumulative_rewards(gs, difficulty)
# condition A: G < 0
# Proposal: G = -0.3, equal but opposite to largest G
# Case: small-big

# Scratch paper
# cum_reward_sb = (m - d / 2) + (m + g + d / 2)  # small big
# cum_reward_bb = (m + d / 2) + (m - g + d / 2)  # big big
# print(cum_reward_sb, cum_reward_bb)
# cum_reward_sb = 2m + g
# cum_reward_bb = 2m - g + d
# cum_reward_sb - sum_reward_bb = 2g - d

# %% Cumulative rewards for updated difficulty values
# gs = [-0.3, 0, 0.05, 0.1, 0.3] # 0.05 will probably be too small to detect?
gs = [-0.3, 0, 0.1, 0.3]
difficulty = [0.05, 0.2, 0.35]

print("g: ", gs)
print("difficulty: ", difficulty)
print("\n")
generate_cumulative_rewards(gs, difficulty)
# scratch paper
# 2g - d = 0
# g = d / 2
# let g = 0.025, d = 0.05

# %% Checking upper and lower bounds
ms = get_ms(gs[0], difficulty, 5)
min_and_max_stimuli(ms, difficulty, gs[0])
