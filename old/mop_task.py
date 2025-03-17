# https://stats.stackexchange.com/questions/343914/expected-value-of-maximum-of-samples-from-normal-distribution
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
mu = 0.5
sigma = 0.25
max_samples = 16
n_reps = 10000

# %%
def draw_samples(n_samples, n_reps):
    samples_mat = []
    for i in range(n_reps):
        samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        samples_mat.append(samples)
    return np.stack(samples_mat)

a = draw_samples(2, 4)

samples_all = []
for i in range(1, max_samples + 1):
    samples_all.append(draw_samples(i, n_reps))

expected_max = []
error = []
for i in samples_all:
    expected_max.append(np.mean(np.max(i, axis=1)))
    error.append(np.std(np.max(i, axis=1)))
#%%
fig = plt.figure(figsize=(10, 5))
plt.errorbar(range(1, max_samples+1), expected_max, yerr=error, marker='o')
plt.xlabel('Number of samples')
plt.ylabel('Expected maximum')
plt.title('Expected maximum of samples from normal distribution')
