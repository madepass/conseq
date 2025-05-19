"""
Power calculations via simulations

- generate random data from distribution with expected properties
- conduct planned statistical analysis from simulated data
- store results of the statistical test of interest (eg p-value)
- repeat previous steps many times (eg 1000 times)
- compute power by averaging the number of times p-value <= 0.05
"""
# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.power as smp

# %% Functions & classes


# %% pre-packaged approach (statsmodels)
nobs = smp.tt_ind_solve_power(effect_size=0.2, nobs1=None, alpha=0.05, power=0.8)
