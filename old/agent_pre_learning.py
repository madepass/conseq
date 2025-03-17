from ast import literal_eval

import pandas as pd
import numpy as np
from main import import_config
from utils import Agent, q, q2learned
from hyperopt import hp
from utils import calculate_learning_time
from utils import objective_simple_q

cfg = import_config()
data = pd.read_csv(cfg.dirs.save + '/barcelona_data.csv')
learning_times = calculate_learning_time(data)

subject_id = 3
subject_1_data = data[(data['subject'] == subject_id) & (data['horizon'] == 1)]
subject_1_decisions = subject_1_data['big'].to_numpy()
subject_1_h1_learning_time = learning_times[subject_id][1]

# %% Simple SB Learning w/ Agent class
agent = Agent()
agent.set_stimuli(stimuli_type='empirical')
space = hp.choice('x', [
    {
        'a': hp.uniform('a', 1E-6, 1),
        'g': 1,  # hp.uniform('g', 0, 1),
        'e': hp.uniform('e', 0, 1),
        'human_decisions': subject_1_decisions,
        'human_learning_time': subject_1_h1_learning_time,
        'stimuli': agent.stimuli,
        'n_rep': 10
    }
])
agent.set_fitting_params(q, space, objective_simple_q)
agent.fit(100)
print(agent.best_params)
# %% Fixed Actions until LT w/ Agent class
agent2 = Agent()
agent2.set_stimuli(stimuli_type='empirical')
# agent.learn(q )
alphas, gammas, learned = [], [], []
for t in range(1000):
    alpha = np.random.uniform(0, 1)
    gamma = np.random.uniform(0, 1)
    result = q(cfg.dirs.stimuli, reward_type=cfg.q.reward_type, optimistic=literal_eval(cfg.q.optimistic),
               constant_epsilon=cfg.q.constant_epsilon, epsilon=cfg.q.epsilon,
               gamma=gamma, constant_alpha=cfg.q.constant_alpha, alpha=alpha,
               fixed_decisions=list(subject_1_decisions[:subject_1_h1_learning_time]), save_result=True)
    alphas.append(alpha)
    gammas.append(gamma)
    learned.append(q2learned(result.loc[subject_1_h1_learning_time, 'q']))
    print(t)
import matplotlib.pyplot as plt
from seaborn import scatterplot
cols = ['alphas', 'gammas', 'learned']
result_df = pd.DataFrame(data=np.stack((np.array(alphas), np.array(gammas), np.array(learned)), axis=1),
                         columns=cols)
scatterplot(data=result_df, x='alphas', y='gammas', hue='learned')
plt.show()

#%% Scratch paper
result = q(cfg.dirs.stimuli, reward_type=cfg.q.reward_type, optimistic=literal_eval(cfg.q.optimistic),
           constant_epsilon=cfg.q.constant_epsilon, epsilon=0,
           gamma=0, constant_alpha=cfg.q.constant_alpha, alpha=0.9,
           fixed_decisions=list(subject_1_decisions[:subject_1_h1_learning_time]), save_result=True)
q_table = result.loc[subject_1_h1_learning_time, 'q']
q2learned(q_table)




