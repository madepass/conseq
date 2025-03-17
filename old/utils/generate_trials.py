####################################################
# Created by Ignasi Cos, ignasi.cos@ub.edu
# Date: 30/09/2020
#
# Modified by Gloria Cecchini, gloria.cecchini@ub.edu
# Date: 19/04/2021
#
# Translated by Michael DePass michaeladepass@gmail.com
# Date: 28/06/2021
#
#
# H0: mean +/- difficulty/2
# H1: mean +/- difficulty/2
#     mean +/- gain +/- difficulty/2
# H2: mean +/- difficulty/2
#     mean +/- gain1 +/- difficulty/2
#     mean +/- gain1 +/- gain2 +/- difficulty/2
#
# ONLY H1
#####################################################
import numpy as np
import math

# Difficulty Level (how distinguishable are the stimuli)
difficulty = [0.01, 0.05, 0.1, 0.15, 0.2]
couples = np.array([0, 1])

# Horizon 1
nEpisode = 50
nrep = int(nEpisode / len(difficulty))

hor = 1
nrep2 = (hor + 1) * nrep
nTrials2 = int(nrep2 * len(difficulty))
trialListValues = np.empty((nTrials2, 2 ** (hor + 1)))
trialListValues[:] = np.nan


# gain/loss for each trial
gl1 = 0.3

for n, diff in zip(np.arange(0, len(difficulty)) + 1, difficulty):
    # mean value is limited
    mu_lim = diff / 2 + gl1

    # equalizing stimuli as a function of difficulty but varying the mean
    mu0 = (
        np.random.randint(1, round((1 - 2 * mu_lim) * 100), (nrep, 1))
        + math.floor(mu_lim * 100)
    ) / 100

    sl1 = (-1) ** np.random.randint(0, 2, (nrep, 1))  # randomize small-large stimuli
    sl2 = (-1) ** np.random.randint(0, 2, (nrep, 1))  # randomize small-large stimuli

    # first trial
    t = 1
    trialListValues[
        np.arange((hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1))
        - 1,
        couples[0] : couples[1] + 1,
    ] = mu0 + sl1 * diff * [-1 / 2, 1 / 2]
    # second trial
    t = 2
    # two cases - the previous choice favours the small stimulus
    trialListValues[
        np.arange((hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1))
        - 1,
        couples[0] : couples[1] + 1,
    ] = (
        mu0 + sl1 * gl1 + sl2 * diff * [-1 / 2, 1 / 2]
    )
    # two cases - the previous choice favours the large stimulus
    trialListValues[
        np.arange((hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1))
        - 1,
        2:4,
    ] = (
        mu0 - sl1 * gl1 + sl2 * diff * [-1 / 2, 1 / 2]
    )


def horizon_1(n_episodes):
    import numpy as np
    import math

    # Difficulty Level (how distinguishable are the stimuli)
    difficulty = [0.01, 0.05, 0.1, 0.15, 0.2]
    couples = np.array([0, 1])

    # Horizon 1
    nEpisode = n_episodes
    nrep = int(nEpisode / len(difficulty))

    hor = 1
    nrep2 = (hor + 1) * nrep
    nTrials2 = int(nrep2 * len(difficulty))
    trialListValues = np.empty((nTrials2, 2 ** (hor + 1)))
    trialListValues[:] = np.nan

    # gain/loss for each trial
    gl1 = 0.3

    for n, diff in zip(np.arange(0, len(difficulty)) + 1, difficulty):
        # mean value is limited
        mu_lim = (diff / 2) + gl1

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
            mu0 + sl1 * gl1 + sl2 * diff * [-1 / 2, 1 / 2]
        )
        # two cases - the previous choice favours the large stimulus
        trialListValues[
            np.arange(
                (hor + 1) * (n - 1) * nrep + t, (hor + 1) * n * nrep + 1, (hor + 1)
            )
            - 1,
            2:4,
        ] = (
            mu0 - sl1 * gl1 + sl2 * diff * [-1 / 2, 1 / 2]
        )

    return np.round(trialListValues, decimals=3)
