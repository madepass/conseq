# %% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from typing import List, Tuple, Union, Dict
import gym
from gym import spaces
from environments import Horizon1
import math
from ast import literal_eval

# %% Functions
def translate_state(env_state):
    """
    Input: state from environment
    Output: agent interpretation of environmental state

    env states
    ==========
    0: trial 1
    1: trial 2 - low reward
    2: trial 2 - high reward

    agent states
    ============
    0: trial 1 - low diff (ie difference between stimuli)
    1: trial 1 - medium diff
    2: trial 1 - high diff
    3: trial 2
    """
    stimuli = env_state["stimuli"]
    diff = round(max(stimuli) - min(stimuli), 3)
    if env_state["state_id"] == 0:  # first trial in episode
        if np.isclose(diff, 0.05):
            agent_state = 0
        elif np.isclose(diff, 0.2):
            agent_state = 1
        elif np.isclose(diff, 0.35):
            agent_state = 2
        else:
            raise ("Unknown difficulty value!")
    else:  # second trial
        agent_state = 3
    return agent_state


class Horizon1VC(gym.Env):
    """
    Gym-compliant custom environment for a 2-trial decision task.

    Observation:
        Dict with:
            'state_id': Discrete(3) - task state (0, 1, or 2)
            'stimuli': Box(2,) - stimulus pair values in [0, 1]

    Actions:
        0 = choose small
        1 = choose big

    Reward:
        Value of chosen stimulus.

    Each episode contains exactly 2 trials.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gs: List[float] = [-0.3, 0, 0.1, 0.3],
        episodes_per_g: int = 30,
        difficulty: List[float] = [0.05, 0.2, 0.35],
        stochastic: Union[bool, float] = False,
        randomize_last_episode: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        assert stochastic is False or (
            0.5 <= stochastic < 1.0
        ), "stochastic must be False or float between 0.5 and 1."

        self.gs = random.sample(gs, len(gs))  # shuffle
        self.epg = episodes_per_g
        self.difficulty = difficulty
        self.stochastic = stochastic
        self.rle = randomize_last_episode
        self.verbose = verbose

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "state_id": spaces.Discrete(3),
                "stimuli": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

        self.reset()

    def reset(self, *, seed=None, options=None) -> Dict[str, Union[int, np.ndarray]]:
        if seed is not None:
            super().reset(seed=seed)

        self.current_g_index = 0
        self.g = self.gs[0]
        self.episode_number = 1
        self.trial_number = 1
        self.neg = self._generate_episode_counts()
        self.done = False
        self.ms = self._generate_mean_values(self.g)
        self.m = round(np.random.choice(self.ms), 2)
        self.d = random.choice(self.difficulty)
        self.state = [0, self._generate_stimuli(0, self.m, self.d)]

        return self._get_obs()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Union[int, np.ndarray]], float, bool, bool, Dict]:
        assert not self.done, "Episode is done. Call reset()."

        reward = max(self.state[1]) if action else min(self.state[1])
        info = self._collect_info(action)

        if self.trial_number == 1:
            self.state = self._state_transition(action)
            self.trial_number += 1
            if self.verbose:
                print(
                    f"g={self.g:.2f}, ep={self.episode_number}, trial={self.trial_number}"
                )
            return self._get_obs(), reward, self.done, False, info

        # Trial 2
        self._update_counters()
        self.state = self._state_transition(action)
        if self.verbose:
            print(
                f"g={self.g:.2f}, ep={self.episode_number}, trial={self.trial_number}"
            )

        return self._get_obs(), reward, self.done, False, info

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        return {
            "state_id": self.state[0],
            "stimuli": np.array(self.state[1], dtype=np.float32),
        }

    def _generate_episode_counts(self) -> List[int]:
        return [
            self.epg + np.random.randint(-2, 3) if self.rle else self.epg
            for _ in self.gs
        ]

    def _generate_mean_values(self, g: float) -> np.ndarray:
        margin = max(self.difficulty) / 2
        return np.linspace(abs(g) + margin, 1 - abs(g) - margin, 5).round(3)

    def _generate_stimuli(self, state: int, m: float, d: float) -> List[float]:
        if state == 0:
            stimuli = [m - d / 2, m + d / 2]
        elif state == 1:
            stimuli = [m - self.g - d / 2, m - self.g + d / 2]
        else:
            stimuli = [m + self.g - d / 2, m + self.g + d / 2]
        random.shuffle(stimuli)
        stimuli = [round(stimulus, 3) for stimulus in stimuli]
        return stimuli

    def _state_transition(self, action: int) -> List[Union[int, List[float]]]:
        state_id = self.state[0]
        unexpected = self.stochastic and random.random() > self.stochastic

        if state_id == 0:
            if action == 0:
                next_state = 1 if unexpected else 2
            else:
                next_state = 2 if unexpected else 1
        else:
            next_state = 0

        return [next_state, self._generate_stimuli(next_state, self.m, self.d)]

    def _update_counters(self):
        if self.episode_number == self.neg[self.current_g_index]:
            if self.current_g_index == len(self.gs) - 1:
                self.done = True
            else:
                self.current_g_index += 1
                self.g = self.gs[self.current_g_index]
                self.ms = self._generate_mean_values(self.g)
                self.episode_number = 1
                self.trial_number = 1
                self.m = round(np.random.choice(self.ms), 3)
                self.d = random.choice(self.difficulty)
        else:
            self.episode_number += 1
            self.trial_number = 1
            self.m = round(np.random.choice(self.ms), 2)
            self.d = random.choice(self.difficulty)

    def _collect_info(self, action: int) -> Dict:
        return {
            "g": self.g,
            "episode": self.episode_number,
            "trial": self.trial_number,
            "m": self.m,
            "d": self.d,
            "state_id": self.state[0],
            "stimuli": self.state[1],
            "action": action,
        }


class QLearningAgent:
    def __init__(
        self,
        actions,
        learning_rate=0.10,
        discount_factor=1,
        temperature=1.0,
        decay_rate=0.99,
        min_temperature=0.1,
        q_table_options="dict",
        alpha_decay=0.99,
    ):
        self.actions = actions
        self.alpha = learning_rate
        self.alpha_decay_rate = alpha_decay
        self.gamma = discount_factor
        self.temperature = temperature  # τ
        self.decay_rate = decay_rate  # temperature decay
        self.min_temperature = min_temperature
        self.q_table_options = q_table_options
        self.q_table = self._initialize_q_table()
        self.state = None

    def _initialize_q_table(self):
        if self.q_table_options == "dict":
            q_table = {
                0: np.zeros(len(self.actions)),  # trial 1 - low delta
                1: np.zeros(len(self.actions)),  # trial 1 - medium delta
                2: np.zeros(len(self.actions)),  # trial 1 - high delta
                3: np.zeros(len(self.actions)),  # trial 2
            }
        elif q_table == "defaultdict":
            self.q_table = defaultdict(
                lambda: np.zeros(len(self.actions))
            )  # Q[state_id][action]
        else:
            raise ("Unsupported q_table type")
        return q_table

    def _translate_state(self, env_state):
        """
        Input: state from environment
        Output: agent interpretation of environmental state

        env states
        ==========
        0: trial 1
        1: trial 2 - low reward
        2: trial 2 - high reward

        agent states
        ============
        0: trial 1 - low diff (ie difference between stimuli)
        1: trial 1 - medium diff
        2: trial 1 - high diff
        3: trial 2
        """
        stimuli = env_state["stimuli"]
        diff = round(max(stimuli) - min(stimuli), 3)
        if env_state["state_id"] == 0:  # first trial in episode
            if np.isclose(diff, 0.05):
                agent_state = 0
            elif np.isclose(diff, 0.2):
                agent_state = 1
            elif np.isclose(diff, 0.35):
                agent_state = 2
            else:
                raise ("Unknown difficulty value!")
        else:  # second trial
            agent_state = 3
        return agent_state

    def get_info(self) -> Dict:
        return {
            "alpha": self.alpha,
            "temperature": self.temperature,
            "q_table": [list(i) for i in agent.q_table.values()],
        }

    def _softmax(self, q_values):
        tau = self.temperature
        if tau == 0:
            return np.eye(len(q_values))[np.argmax(q_values)]
        preferences = q_values / tau
        max_pref = np.max(preferences)  # For numerical stability
        exp_preferences = np.exp(preferences - max_pref)
        return exp_preferences / np.sum(exp_preferences)

    def choose_action(self, state):
        state_id = self._translate_state(state)
        if state_id in [0, 1, 2]:  # trial 1
            q_values = self.q_table[state_id]
            probs = self._softmax(q_values)
            return np.random.choice(len(q_values), p=probs)
        else:  # trial 2 - terminal state
            return 1  # big, only logical choice

    def update(self, state_id, action, reward, next_state_id):
        if state_id == 3:  # terminal state
            best_next = 0
            if self.temperature > self.min_temperature:
                self.temperature *= self.decay_rate
            self.alpha *= self.alpha_decay_rate
        else:
            best_next = np.max(self.q_table[next_state_id])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_id][action]
        self.q_table[state_id][action] += self.alpha * td_error

    def policy(self, state_id):
        return np.argmax(self.q_table[state_id])


def correct_choice_column(env_df):
    # returns list of 0/1 (i.e., incorrect/correct)
    correct_choice = []
    for index, row in env_df.iterrows():
        if row["trial"] == 2:
            correct = row["action"]  # 1/big correct for terminal state
        elif row["d"] > (2 * row["g"]):
            correct = row["action"]  # correct always big
        elif row["d"] == (2 * row["g"]):
            correct = 1  # both strats equal
        elif row["d"] < (2 * row["g"]) and row["trial"] == 1:
            correct = 1 if row["action"] == 0 else 0
        else:
            print("theres a case i havent thought of, or an error")
        correct_choice.append(correct)
    return correct_choice


def import_agent_info_df(file_path):
    agent_info_df = pd.read_csv(file_path)
    # visualize q-values
    q_values = agent_info_df["q_table"].apply(literal_eval)
    agent_info_df["q_table"] = q_values
    return agent_info_df


# %% Training loop
# TODO: training loop function, can import to other scripts
env = Horizon1VC(gs=[0.3], episodes_per_g=500)
agent = QLearningAgent([0, 1], learning_rate=0.1, temperature=3)

num_episodes = sum(env._generate_episode_counts())  # Use total episode count

env_info_list = []
agent_info_list = []
env_state = env.reset()
while not env.done:
    action = agent.choose_action(env_state)
    next_state_env, reward, done, _, info = env.step(action)
    env_info_list.append(info)
    agent_info_list.append(agent.get_info())

    state_id = agent._translate_state(env_state)
    next_state_id = agent._translate_state(next_state_env)
    agent.update(state_id, action, reward, next_state_id)

    env_state = next_state_env

env_info_df = pd.DataFrame(env_info_list)
env_info_df.to_csv("./output/env_info.csv")
agent_info_df = pd.DataFrame(agent_info_list)
correct_column = correct_choice_column(env_info_df)
agent_info_df["correct"] = correct_column
agent_info_df.to_csv("./output/agent_info.csv")

# %% Visualize results
# load data
env_info_df = pd.read_csv("./output/env_info.csv")
agent_info_df = import_agent_info_df("./output/agent_info.csv")

# visualize q_values
agent_info_df["q_table"]
q1, q2, q3, q4, q5, q6 = (
    [],
    [],
    [],
    [],
    [],
    [],
)
for index, row in agent_info_df.iterrows():
    q1.append(row["q_table"][0][0])
    q2.append(row["q_table"][0][1])
    q3.append(row["q_table"][1][0])
    q4.append(row["q_table"][1][1])
    q5.append(row["q_table"][2][0])
    q6.append(row["q_table"][2][1])
qs = [q1, q2, q3, q4, q5, q6]

# %%
fig = plt.figure(figsize=(10, 5))
for i, _ in enumerate(qs):
    plt.plot(_, label=f"q{i+1}")
ax = plt.gca()
ax.spines[["right", "top"]].set_visible(False)
plt.legend()
plt.show()
