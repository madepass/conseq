# %% Imports
import math
import numpy as np
import pandas as pd
import random

# %% Debugging tools
from pdb import set_trace as st
from pdb import pm
import wat
import snoop
from IPython import embed as ipe

# %% Functions
def horizon_1_og(g=0.3, n_episodes=100, difficulty=[0.01, 0.05, 0.1, 0.15, 0.2]):
    """
    Original horizon_1 function which generates stimuli for all n_episodes
    """
    # Difficulty Level (how distinguishable are the stimuli)
    couples = np.array([0, 1])

    # Horizon 1
    nrep = int(n_episodes / len(difficulty))

    hor = 1
    nrep2 = (hor + 1) * nrep
    nTrials2 = int(nrep2 * len(difficulty))
    trialListValues = np.empty((nTrials2, 2 ** (hor + 1)))
    trialListValues[:] = np.nan

    # gain/loss for each trial
    for n, diff in zip(np.arange(0, len(difficulty)) + 1, difficulty):
        # mean value is limited
        mu_lim = (diff / 2) + g

        # equalizing stimuli as a function of difficulty but varying the mean
        mu0 = (
            np.random.randint(1, round((1 - 2 * mu_lim) * 100), (nrep, 1))
            + math.floor(mu_lim * 100)
        ) / 100

        sl1 = (-1) ** np.random.randint(
            0, 2, (nrep, 1)
        )  # randomize small-large stimuli
        sl2 = (-1) ** np.random.randint(
            0, 2, (nrep, 1)
        )  # randomize small-large stimuli

        # first trial
        t = 1
        trialListValues[
            np.arange(
                (hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1)
            )
            - 1,
            couples[0] : couples[1] + 1,
        ] = mu0 + sl1 * diff * [-1 / 2, 1 / 2]
        # second trial
        t = 2
        # two cases - the previous choice favours the small stimulus
        trialListValues[
            np.arange(
                (hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1)
            )
            - 1,
            couples[0] : couples[1] + 1,
        ] = (
            mu0 + sl1 * g + sl2 * diff * [-1 / 2, 1 / 2]
        )
        # two cases - the previous choice favours the large stimulus
        trialListValues[
            np.arange(
                (hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1)
            )
            - 1,
            2:4,
        ] = (
            mu0 - sl1 * g + sl2 * diff * [-1 / 2, 1 / 2]
        )

    return np.round(trialListValues, decimals=3)


def multi_g(
    gs=[0.1, 0.3],
    episodes_per_block=100,
    episodes_per_g=20,
    difficulty=[0.01, 0.05, 0.1, 0.15, 0.2],
):
    """
    Notes
    =====
    something
    """
    all_stims = []
    episode_numbers, epgs = [], []
    for g in gs:
        epg = episodes_per_g + np.random.randint(-2, high=3)
        stims = horizon_1_og(g=g, n_episodes=epg, difficulty=difficulty)
        # append
        all_stims.append(stims)
        episode_numbers.append(np.arange(epg))
        epgs.append(epg)
    all_stims = np.concatenate(all_stims)
    episode_numbers = np.concatenate(episode_numbers)
    trial_numbers = [0, 1] * (len(all_stims) // 2)
    gains = [g for (g, epg) in zip(gs, epgs) for _ in range(epg)]
    gains_col = [gain for gain in gains for _ in range(2)]
    st()
    data = np.concatenate((episode_numbers, trial_numbers, all_stims, gains_col))
    stim_df = pd.DataFrame(data=data)
    return stim_df


class Horizon1:
    """
    Generate Horizon 1 environment.

    stochastic: either False, or float describing the probability of the "expected" outcome
    Should be between [0.5, 1]

    states
    ======
    0: first trial of an episode
    1: second trial, low reward
    2: second trial, high reward

    actions
    =======
    0: choose small
    1: choose big
    """

    def __init__(
        self,
        gs=[0.1, 0.3],
        episodes_per_g=20,
        difficulty=[0.01, 0.05, 0.1, 0.15, 0.2],
        stochastic=False,
        randomize_last_episode=False,
        verbose=True,
    ):
        self.gs = gs
        self.current_g_index = 0
        self.g = gs[0]
        self.epg = episodes_per_g
        self.episode_number = 1
        self.trial_number = 1
        self.difficulty = difficulty
        self.d = random.choice(self.difficulty)
        self.ms = np.linspace(
            abs(self.g) + max(self.difficulty) / 2,
            1 - abs(self.g) - max(self.difficulty) / 2,
            5,
        )
        self.m = np.round(np.random.choice(self.ms), decimals=2)
        self.all_stims = []
        self.accumulated_reward = 0
        self.stochastic = stochastic
        self.rle = randomize_last_episode
        self.actions = [0, 1]
        self.state = [
            0,
            self.generate_stimuli(0, self.m, self.d),
        ]  # ["state", [stimuli]]
        self.neg = self.number_of_episodes_to_generate()
        self.done = False
        self.verbose = verbose
        assert (self.stochastic is False) or (
            0.5 <= self.stochastic < 1.0
        ), "stochastic should be False (deterministic state transitions), or a float between 0.5 and 1 indicating the probability of the expected transition."

    def reset(self):
        """
        unfinished
        """
        self.state = [0, self.generate_stimuli(0)]
        self.episode_number = 1
        self.trial_number = 1
        self.current_g_index = 0
        return self.state

    def number_of_episodes_to_generate(self):
        if self.rle:  # randomize last episode
            neg = [self.epg + np.random.randint(-2, high=3) for g in self.gs]
        else:
            neg = [self.epg] * len(self.gs)
        return neg  # list containing number of episodes to generate per g

    def generate_stimuli(self, state, m, d):
        if state == 0:
            stimuli = [m - d / 2, m + d / 2]
        elif state == 1:
            stimuli = [m - self.g - d / 2, m - self.g + d / 2]
        else:  # state == 2
            stimuli = [m + self.g - d / 2, m + self.g + d / 2]
        random.shuffle(stimuli)  # randomize left or right position on screen
        return stimuli

    def state_transition(self, action):
        unexpected_transition = 0
        if self.stochastic:
            unexpected_transition = 1 if random.uniform(0, 1) > self.stochastic else 0
        if (self.state[0] == 0) and (action == 0):
            new_state = 1 if unexpected_transition else 2  # high mean reward state
        elif (self.state[0] == 0) and (action == 1):
            new_state = 2 if unexpected_transition else 1  # low mean reward state
        else:  # state != 1 (terminal state, go back to state 0)
            new_state = 0
        new_state = [new_state, self.generate_stimuli(new_state, self.m, self.d)]
        self.state = new_state
        return new_state

    def _update(self):
        if self.episode_number == self.neg[self.current_g_index]:
            if self.current_g_index == len(self.gs) - 1:
                if self.state[0] != 0:
                    self.done = True
        if self.episode_number == self.neg[self.current_g_index]:
            if self.state[0] != 0:
                self.current_g_index += 1
                if self.current_g_index < len(self.gs):
                    self.g = self.gs[self.current_g_index]
                self.episode_number = 1
                self.trial_number += 1
                return 1
        self.g = self.gs[self.current_g_index]
        self.trial_number += 1
        if self.state[0] != 0:
            self.episode_number += 1

    def step(self, action):
        assert (
            self.done == False
        ), "Environment is done. Call reset() method to use same instance again."
        if self.trial_number == 1:
            # new_state = self.state_transition(action)
            reward = max(self.state[1]) if action else min(self.state[1])
            if self.verbose:
                print(self.g, self.episode_number, self.trial_number)
            env_params = [
                self.g,
                self.episode_number,
                self.trial_number,
                self.m,
                self.d,
                self.state[0],
                self.state[1],
                action,
            ]
            # self._update()
            # comment above and add line below
            new_state = self.state_transition(action)
            self.trial_number += 1
            return new_state, reward, self.done, env_params
        if self.state[0] == 0:  # set m and d for the episode
            self.m = np.round(np.random.choice(self.ms), decimals=2)
            self.d = random.choice(self.difficulty)
        env_params = [
            self.g,
            self.episode_number,
            self.trial_number,
            self.m,
            self.d,
            self.state[0],
            self.state[1],
            action,
        ]
        new_state = self.state_transition(action)
        reward = max(new_state[1]) if action else min(new_state[1])
        if self.verbose:
            print(self.g, self.episode_number, self.trial_number)
        self._update()
        return new_state, reward, self.done, env_params


# %% Main
def main():
    pass


if __name__ == "__main__":
    main()

# %% Scratch paper
# Section to keep most current work before refactoring
# env = Horizon1(gs=[0.1, 0.3], episodes_per_g=20)
env = Horizon1(
    gs=[0.3, 0.15],
    episodes_per_g=20,
    stochastic=False,
    randomize_last_episode=False,
    verbose=True,
)
states, rewards, dones, params = [], [], [], []
done = False
while not done:
    state, reward, done, env_params = env.step(0)
    states.append(state)
    rewards.append(reward)
    dones.append(done)
    params.append(env_params)
results_df = pd.DataFrame(
    data=params,
    columns=["g", "episode", "trial", "m", "d", "state", "stimuli", "action"],
)
results_df.to_csv("/home/maikito/mad/conseq/output/results_df.csv")
