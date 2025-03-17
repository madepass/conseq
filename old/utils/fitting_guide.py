# Good old pandas and numpy
import pandas as pd
import numpy as np

# Unfortunately I'm still using matplotlib for graphs
import matplotlib.pyplot as plt
import seaborn as sns

# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Introduction%20to%20Bayesian%20Optimization%20with%20Hyperopt.ipynb
# https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f


def objective(x):
    """Objective function to minimize"""

    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Return the value of the polynomial
    return f(x) * 0.05


# Space over which to evluate the function is -5 to 6
x = np.linspace(-5, 6, 10000)
y = objective(x)

miny = min(y)
minx = x[np.argmin(y)]

# Visualize the function
plt.figure(figsize=(8, 6))
plt.style.use("fivethirtyeight")
plt.title("Objective Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.vlines(minx, min(y) - 50, max(y), linestyles="--", colors="r")
plt.plot(x, y)

# Print out the minimum of the function and value
print("Minimum of %0.4f occurs at %0.4f" % (miny, minx))

# Domain

from hyperopt import hp

# Create the domain space
space = hp.uniform("x", -5, 6)

from hyperopt.pyll.stochastic import sample

samples = []

# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(space))

# Histogram of the values
plt.hist(samples, bins=20, edgecolor="black")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("Domain Space")

# Hyperparameter Optimization Algorithm
from hyperopt import rand, tpe

# Create the algorithms
tpe_algo = tpe.suggest
rand_algo = rand.suggest

# History

from hyperopt import Trials

# Create two trials objects
tpe_trials = Trials()
rand_trials = Trials()

# Run the optimization

from hyperopt import fmin

# Run 2000 evals with the tpe algorithm
tpe_best = fmin(
    fn=objective,
    space=space,
    algo=tpe_algo,
    trials=tpe_trials,
    max_evals=2000,
    rstate=np.random.RandomState(50),
)

print(tpe_best)

# Run 2000 evals with the random algorithm
rand_best = fmin(
    fn=objective,
    space=space,
    algo=rand_algo,
    trials=rand_trials,
    max_evals=2000,
    rstate=np.random.RandomState(50),
)

# Print out information about losses
print(
    "Minimum loss attained with TPE:    {:.4f}".format(
        tpe_trials.best_trial["result"]["loss"]
    )
)
print(
    "Minimum loss attained with random: {:.4f}".format(
        rand_trials.best_trial["result"]["loss"]
    )
)
print("Actual minimum of f(x):            {:.4f}".format(miny))

# Print out information about number of trials
print(
    "\nNumber of trials needed to attain minimum with TPE:    {}".format(
        tpe_trials.best_trial["misc"]["idxs"]["x"][0]
    )
)
print(
    "Number of trials needed to attain minimum with random: {}".format(
        rand_trials.best_trial["misc"]["idxs"]["x"][0]
    )
)

# Print out information about value of x
print("\nBest value of x from TPE:    {:.4f}".format(tpe_best["x"]))
print("Best value of x from random: {:.4f}".format(rand_best["x"]))
print("Actual best value of x:      {:.4f}".format(minx))


# Results

tpe_results = pd.DataFrame(
    {
        "loss": [x["loss"] for x in tpe_trials.results],
        "iteration": tpe_trials.idxs_vals[0]["x"],
        "x": tpe_trials.idxs_vals[1]["x"],
    }
)

tpe_results.head()

tpe_results["rolling_average_x"] = (
    tpe_results["x"].rolling(50).mean().fillna(method="bfill")
)
tpe_results["rolling_average_loss"] = (
    tpe_results["loss"].rolling(50).mean().fillna(method="bfill")
)
tpe_results.head()

plt.figure(figsize=(10, 8))
plt.plot(tpe_results["iteration"], tpe_results["x"], "bo", alpha=0.5)
plt.xlabel("Iteration", size=22)
plt.ylabel("x value", size=22)
plt.title("TPE Sequence of Values", size=24)
plt.hlines(minx, 0, 2000, linestyles="--", colors="r")

plt.figure(figsize=(8, 6))
plt.hist(tpe_results["x"], bins=50, edgecolor="k")
plt.title("Histogram of TPE Values")
plt.xlabel("Value of x")
plt.ylabel("Count")

# Sort with best loss first
tpe_results = tpe_results.sort_values("loss", ascending=True).reset_index()

plt.plot(tpe_results["iteration"], tpe_results["loss"], "bo", alpha=0.3)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("TPE Sequence of Losses")

print(
    "Best Loss of {:.4f} occured at iteration {}".format(
        tpe_results["loss"][0], tpe_results["iteration"][0]
    )
)

# Random Results

rand_results = pd.DataFrame(
    {
        "loss": [x["loss"] for x in rand_trials.results],
        "iteration": rand_trials.idxs_vals[0]["x"],
        "x": rand_trials.idxs_vals[1]["x"],
    }
)

rand_results.head()

plt.figure(figsize=(10, 8))
plt.plot(rand_results["iteration"], rand_results["x"], "bo", alpha=0.5)
plt.xlabel("Iteration", size=22)
plt.ylabel("x value", size=22)
plt.title("Random Sequence of Values", size=24)
plt.hlines(minx, 0, 2000, linestyles="--", colors="r")

# Sort with best loss first
rand_results = rand_results.sort_values("loss", ascending=True).reset_index()

plt.figure(figsize=(8, 6))
plt.hist(rand_results["x"], bins=50, edgecolor="k")
plt.title("Histogram of Random Values")
plt.xlabel("Value of x")
plt.ylabel("Count")

# Print information
print(
    "Best Loss of {:.4f} occured at iteration {}".format(
        rand_results["loss"][0], rand_results["iteration"][0]
    )
)
