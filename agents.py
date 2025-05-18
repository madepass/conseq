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

# %% Functions


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
        self.state = self._state_transition(action)
        self._update_counters()
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
    def __init__(self, actions, learning_rate=0.15, discount_factor=0.99, epsilon=0.1):
        self.actions = actions
        self.q_table = defaultdict(
            lambda: np.zeros(len(actions))
        )  # Q[state_id][action]
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state_id):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state_id])

    def update(self, state_id, action, reward, next_state_id):
        best_next = np.max(self.q_table[next_state_id])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_id][action]
        self.q_table[state_id][action] += self.alpha * td_error

    def policy(self, state_id):
        return np.argmax(self.q_table[state_id])


# %% Training loop
env = Horizon1VC()
agent = QLearningAgent(
    [0, 1],
)

num_episodes = sum(env._generate_episode_counts())  # Use total episode count

state = env.reset()
info_list = []
total_reward = 0

while not env.done:
    state_id = state["state_id"]
    action = agent.choose_action(state_id)
    next_state, reward, done, _, info = env.step(action)
    info_list.append(info)

    next_state_id = next_state["state_id"]
    agent.update(state_id, action, reward, next_state_id)

    total_reward += reward

    # if info["trial"] == 2:  # End of episode
    #     episode_rewards.append(total_reward)
    #     total_reward = 0

    state = next_state
info_df = pd.DataFrame(info_list)
info_df.to_csv("./output/info.csv")
# print("Training complete!")
# print(f"Average reward: {np.mean(episode_rewards):.3f}")

# %% Main

agent = QLearningAgent([0, 1])  # 0: choose small, 1: choose big
env = Horizon1VC()
obs = env.reset()
terminated = False
info_list = []
while not terminated:
    obs, reward, terminated, truncated, info = env.step(0)
    info_list.append(info)

info_df = pd.DataFrame(data=info_list)
info_df.to_csv("./output/info.csv")
