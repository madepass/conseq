# %% Imports
import math
import numpy as np

# %% Debugging tools
from pdb import set_trace as st
from pdb import pm
import wat
import snoop

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
    g=[0.1, 0.3],
    n_episodes=100,
    episodes_per_g=20,
    difficulty=[0.01, 0.05, 0.1, 0.15, 0.2],
):
    epg = episodes_per_g + np.random.randint(-2, high=3)
    all_stims = []
    for gain in g:
        stims = horizon_1_og(g=gain, n_episodes=epg)
        all_stims.append(stims)
    return all_stims


# %% Main
def main():
    pass


if __name__ == "__main__":
    main()

# %% Scratch paper
# Section to keep most current work before refactoring
env = horizon_1_og()
