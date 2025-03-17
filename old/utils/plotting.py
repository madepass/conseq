import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#  TODO: Add learning time to plots
#  TODO: Make action_scatter_model function

def calculate_performance(window_size: int, results_df: pd.DataFrame) -> list:
    correct = results_df['chose_big'] == results_df['action*']
    performance = []
    for i in np.arange(0, len(correct), window_size):
        performance.append(sum(correct[i:(i+window_size)])/window_size)
    return performance


def action_scatter_q(cfg, file_name, save_name):
    df = pd.read_csv(cfg.dirs.save + '/' + file_name)

    fig, ax = plt.subplots(5, 1, figsize=(10, 8))

    fig.suptitle('RL Agent Actions')

    set_xlim = False

    ax[0].title.set_text('Actions')
    ax[0].plot(np.arange(len(df[df['NT'] == 1]['action']))+1, df[df['NT'] == 1]['action'].to_numpy(), 'ro', label='S1', ms=2)
    ax[0].set_ylabel('Action', fontsize=18)
    ax[0].legend()
    if set_xlim: ax[0].set_xlim((0, 100))

    ax[1].plot(np.arange(len(df[df['NT'] == 2]['action']))+1, df[df['NT'] == 2]['action'].to_numpy(), 'go', label='S2', ms=2)
    ax[1].set_ylabel('Action', fontsize=18)
    ax[1].legend()
    if set_xlim: ax[1].set_xlim((0, 100))

    ax[2].plot(np.arange(len(df[df['NT'] == 3]['action']))+1, df[df['NT'] == 3]['action'].to_numpy(), 'bo', label='S3', ms=2)
    ax[2].set_ylabel('Action', fontsize=18)
    ax[2].set_xlabel('Episode', fontsize=12)
    ax[2].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    performance = calculate_performance(10, df)
    ax[3].title.set_text('Performance')
    ax[3].plot(np.arange(len(performance)) + 1, performance, 'ko', label='performance', ms=2)
    ax[3].set_ylabel('Perf.', fontsize=18)
    ax[3].set_xlabel(f'Time (groups of 10 episodes)', fontsize=12)
    ax[3].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    ax[4].set_title('Hyperparameters')
    ax[4].set_axis_off()
    cell_text = [[str(cfg.q.epsilon), str(cfg.q.constant_epsilon), str(cfg.q.alpha), str(cfg.q.constant_alpha),
                  str(cfg.q.optimistic)]]
    col_labels = ['epsilon', 'constant_epsilon', 'alpha', 'constant_alpha', 'optimistic']
    table = ax[4].table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    table.scale(1, 3)

    plt.tight_layout()
    plt.savefig(cfg.dirs.save + '/' + save_name, format='svg')
    plt.show()
    return 1


def action_scatter_parallel(cfg, file_name, save_name):
    df = pd.read_csv(cfg.dirs.save + '/' + file_name)

    fig, ax = plt.subplots(5, 1, figsize=(10, 8))

    fig.suptitle('RL Agent Actions')

    set_xlim = False

    ax[0].title.set_text('Actions')
    ax[0].plot(np.arange(len(df[df['NT'] == 1]['action']))+1, df[df['NT'] == 1]['action'].to_numpy(), 'ro', label='S1', ms=2)
    ax[0].set_ylabel('Action', fontsize=18)
    ax[0].legend()
    if set_xlim: ax[0].set_xlim((0, 100))

    ax[1].plot(np.arange(len(df[df['NT'] == 2]['action']))+1, df[df['NT'] == 2]['action'].to_numpy(), 'go', label='S2', ms=2)
    ax[1].set_ylabel('Action', fontsize=18)
    ax[1].legend()
    if set_xlim: ax[1].set_xlim((0, 100))

    ax[2].plot(np.arange(len(df[df['NT'] == 3]['action']))+1, df[df['NT'] == 3]['action'].to_numpy(), 'bo', label='S3', ms=2)
    ax[2].set_ylabel('Action', fontsize=18)
    ax[2].set_xlabel('Episode', fontsize=12)
    ax[2].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    performance = calculate_performance(10, df)
    ax[3].title.set_text('Performance')
    ax[3].plot(np.arange(len(performance)) + 1, performance, 'ko', label='performance', ms=2)
    ax[3].set_ylabel('Perf.', fontsize=18)
    ax[3].set_xlabel(f'Time (groups of 10 episodes)', fontsize=12)
    ax[3].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    ax[4].set_title('Hyperparameters')
    ax[4].set_axis_off()
    cell_text = [[str(cfg.qp.epsilon), str(cfg.qp.constant_epsilon), str(cfg.qp.alpha), str(cfg.qp.constant_alpha),
                  str(cfg.qp.optimistic)]]
    col_labels = ['epsilon', 'constant_epsilon', 'alpha', 'constant_alpha', 'optimistic']
    table = ax[4].table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    table.scale(1, 3)

    plt.tight_layout()
    plt.savefig(cfg.dirs.save + '/' + save_name, format='svg')
    plt.show()
    return 1


def action_scatter_sequential(cfg, file_name, save_name):
    df = pd.read_csv(cfg.dirs.save + '/' + file_name)

    fig, ax = plt.subplots(5, 1, figsize=(10, 8))

    fig.suptitle('RL Agent Actions')

    set_xlim = False

    ax[0].title.set_text('Actions')
    ax[0].plot(np.arange(len(df[df['NT'] == 1]['action']))+1, df[df['NT'] == 1]['action'].to_numpy(), 'ro', label='S1', ms=2)
    ax[0].set_ylabel('Action', fontsize=18)
    ax[0].legend()
    if set_xlim: ax[0].set_xlim((0, 100))

    ax[1].plot(np.arange(len(df[df['NT'] == 2]['action']))+1, df[df['NT'] == 2]['action'].to_numpy(), 'go', label='S2', ms=2)
    ax[1].set_ylabel('Action', fontsize=18)
    ax[1].legend()
    if set_xlim: ax[1].set_xlim((0, 100))

    ax[2].plot(np.arange(len(df[df['NT'] == 3]['action']))+1, df[df['NT'] == 3]['action'].to_numpy(), 'bo', label='S3', ms=2)
    ax[2].set_ylabel('Action', fontsize=18)
    ax[2].set_xlabel('Episode', fontsize=12)
    ax[2].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    performance = calculate_performance(10, df)
    ax[3].title.set_text('Performance')
    ax[3].plot(np.arange(len(performance)) + 1, performance, 'ko', label='performance', ms=2)
    ax[3].set_ylabel('Perf.', fontsize=18)
    ax[3].set_xlabel(f'Time (groups of 10 episodes)', fontsize=12)
    ax[3].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    ax[4].set_title('Hyperparameters')
    ax[4].set_axis_off()
    cell_text = [[str(cfg.qp.epsilon), str(cfg.qp.constant_epsilon), str(cfg.qp.alpha), str(cfg.qp.constant_alpha),
                  str(cfg.qp.optimistic)]]
    col_labels = ['epsilon', 'constant_epsilon', 'alpha', 'constant_alpha', 'optimistic']
    table = ax[4].table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    table.scale(1, 3)

    plt.tight_layout()
    plt.savefig(cfg.dirs.save + '/' + save_name, format='svg')
    plt.show()
    return 1


def action_scatter_model(cfg, file_name, save_name):
    df = pd.read_csv(cfg.dirs.save + '/' + file_name)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    fig.suptitle('RL Agent Actions')

    set_xlim = True

    ax[0].title.set_text('Actions')
    ax[0].plot(np.arange(len(df[df['NT'] == 1]['action']))+1, df[df['NT'] == 1]['action'].to_numpy(), 'ro', label='S1', ms=2)
    ax[0].set_ylabel('Action', fontsize=18)
    ax[0].legend()
    if set_xlim: ax[0].set_xlim((0, 100))

    ax[1].plot(np.arange(len(df[df['NT'] == 2]['action']))+1, df[df['NT'] == 2]['action'].to_numpy(), 'go', label='S2', ms=2)
    ax[1].set_ylabel('Action', fontsize=18)
    ax[1].legend()
    if set_xlim: ax[1].set_xlim((0, 100))

    ax[2].plot(np.arange(len(df[df['NT'] == 3]['action']))+1, df[df['NT'] == 3]['action'].to_numpy(), 'bo', label='S3', ms=2)
    ax[2].set_ylabel('Action', fontsize=18)
    ax[2].set_xlabel('Episode', fontsize=12)
    ax[2].legend()
    if set_xlim: ax[2].set_xlim((0, 100))

    # performance = calculate_performance(10, df)
    # ax[3].title.set_text('Performance')
    # ax[3].plot(np.arange(len(performance)) + 1, performance, 'ko', label='performance', ms=2)
    # ax[3].set_ylabel('Perf.', fontsize=18)
    # ax[3].set_xlabel(f'Time (groups of 10 episodes)', fontsize=12)
    # ax[3].legend()
    # if set_xlim: ax[2].set_xlim((0, 100))
    #
    # ax[4].set_title('Hyperparameters')
    # ax[4].set_axis_off()
    # cell_text = [[str(cfg.q.epsilon), str(cfg.q.constant_epsilon), str(cfg.q.alpha), str(cfg.q.constant_alpha),
    #               str(cfg.q.optimistic)]]
    # col_labels = ['epsilon', 'constant_epsilon', 'alpha', 'constant_alpha', 'optimistic']
    # table = ax[4].table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    # table.scale(1, 3)

    plt.tight_layout()
    plt.savefig(cfg.dirs.save + '/' + save_name, format='svg')
    plt.show()
    return 1




# stim_df = pd.read_csv(cfg.dirs.save + '/stimuli.txt')
# q_df = pd.read_csv(cfg.dirs.save + '/q_learning_result.txt')
# q_df = pd.read_csv(cfg.dirs.save + '/q_learning_result_sequential.txt')













