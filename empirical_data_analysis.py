# %% Imports
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% Functions


def letter_annotation(ax, xoffset, yoffset, letter):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=12, weight="bold")


def convert_stringified_lists(df):
    for col in df.columns:
        # Check only if column is of object (string) type
        if df[col].dtype == "object":
            try:
                # Try evaluating the first non-null value
                sample = df[col].dropna().iloc[0]
                evaluated = ast.literal_eval(sample)
                if isinstance(evaluated, list):
                    df[col] = df[col].apply(
                        lambda x: ast.literal_eval(x) if pd.notna(x) else x
                    )
            except (ValueError, SyntaxError):
                # Not a valid Python literal or not a list — skip
                continue
    return df


def import_psychopy(file_path: str) -> pd.DataFrame:
    """
    Import .csv export from psychopy experiment
    Returns pandas dataframe
    """
    remove_idx = []  # rows to remove
    df = pd.read_csv(file_path)
    # instructions duration
    instructions_start = df["image.started"][0]
    instructions_end = df["mouse_instructions.time"][0]
    instructions_end = instructions_end.replace("[", "]")
    instructions_end = float(instructions_end.split("]")[1])
    instructions_duration = instructions_end - instructions_start
    instructions_idx = list(df.loc[pd.isna(df["g_block"]), :].index)
    remove_idx += instructions_idx
    # drop environment initialization rows
    aux = df["initialize_g_block_variables.started"].isna()
    new_env_idx = list(np.where(aux == False)[0] - 1)
    remove_idx += new_env_idx[1:]
    aux = list(np.where(df["thisN"] == 3)[0])
    experiment_end_ind = aux
    remove_idx += experiment_end_ind
    df = df.drop(df.index[remove_idx], axis=0)
    df.reset_index(inplace=True)
    # what columns do we want to keep?
    trial_number = list(np.arange(df.shape[0]) + 1)
    df["trial"] = trial_number
    keep_columns = [
        "g_block",
        "episodes.thisN",
        "trial",
        "mouse_2.x",
        "mouse_2.y",
        "mouse_2.leftButton",
        "mouse_2.time",
        "mouse_2.clicked_name",
        "d",
        "m",
        "left_stimulus_height",
        "right_stimulus_height",
        "left_chosen",
        "chose_bigger",
        "left_stimulus_1.started",  # left stimulus appears
        "left_stimulus_1.stopped",  # left stimulus disappears
        "right_stimulus_1.started",  # right stimulus appears
        "right_stimulus_1.stopped",  # right stimulus disappears
        "left_stimuli_outline_2.started",  # GO signal
        "trial.stopped",
    ]
    df2 = df[keep_columns]
    df2 = convert_stringified_lists(df2)
    # better column names
    df2.rename(
        columns={
            "g_block": "g",
            "episodes.thisN": "episode",
            "mouse_2.x": "mouse_x",
            "mouse_2.y": "mouse_y",
            "mouse_2.leftButton": "mouse_click",
            "mouse_2.time": "mouse_time",
            "left_chosen": "chose_left",
            "chose_bigger": "chose_big",
            "left_stimuli_outline_2.started": "go",
            "trial.stopped": "mouse_click_time",
        },
        inplace=True,
    )
    df2 = df2.astype({"episode": int})
    df2.episode = df2.episode + 1
    return df2


def import_pavlovia(file_path: str) -> pd.DataFrame:
    """
    Import .csv export from online PsychoPy experiment hosted on Pavlovia.
    Returns pandas dataframe
    """
    breakpoint()
    remove_idx = []  # rows to remove
    df = pd.read_csv(file_path)
    # instructions duration
    instructions_start = df["instructions.started"][0]
    instructions_end = df["mouse_instructions.time"][0]
    instructions_end = instructions_end.replace("[", "]")
    instructions_end = float(instructions_end.split("]")[1])
    instructions_duration = instructions_end - instructions_start
    instructions_idx = list(df.loc[pd.isna(df["g_block"]), :].index)
    remove_idx += instructions_idx
    # drop environment initialization rows
    aux = df["initialize_g_block_variables.started"].isna()
    new_env_idx = list(np.where(aux == False)[0] - 1)
    remove_idx += new_env_idx[1:]
    aux = list(np.where(df["thisN"] == 3)[0])
    experiment_end_ind = aux
    remove_idx += experiment_end_ind
    df = df.drop(df.index[remove_idx], axis=0)
    df.reset_index(inplace=True)
    # what columns do we want to keep?
    trial_number = list(np.arange(df.shape[0]) + 1)
    df["trial"] = trial_number
    keep_columns = [
        "g_block",
        "episodes.thisN",
        "trial",
        "mouse_2.x",
        "mouse_2.y",
        "mouse_2.leftButton",
        "mouse_2.time",
        "mouse_2.clicked_name",
        "d",
        "m",
        "left_stimulus_height",
        "right_stimulus_height",
        "left_chosen",
        "chose_bigger",
        "left_stimulus_1.started",  # left stimulus appears
        "left_stimulus_1.stopped",  # left stimulus disappears
        "right_stimulus_1.started",  # right stimulus appears
        "right_stimulus_1.stopped",  # right stimulus disappears
        "left_stimuli_outline_2.started",  # GO signal
        "trial.stopped",
    ]
    df2 = df[keep_columns]
    df2 = convert_stringified_lists(df2)
    # better column names
    df2.rename(
        columns={
            "g_block": "g",
            "episodes.thisN": "episode",
            "mouse_2.x": "mouse_x",
            "mouse_2.y": "mouse_y",
            "mouse_2.leftButton": "mouse_click",
            "mouse_2.time": "mouse_time",
            "left_chosen": "chose_left",
            "chose_bigger": "chose_big",
            "left_stimuli_outline_2.started": "go",
            "trial.stopped": "mouse_click_time",
        },
        inplace=True,
    )
    df2 = df2.astype({"episode": int})
    df2.episode = df2.episode + 1
    return df2


def stim_height_categories(series):
    new_col = []
    for i in series:
        if i < 0.4:
            new_col.append("low")
        elif 0.4 < i < 0.6:
            new_col.append("medium")
        else:
            new_col.append("high")
    return new_col


# %% Import data (local)
data_dir = "/media/maikito/mad_mini/consequential_task/psychopy/h1_v2/data/saved/"
file_name = "mad_test2_consequential_2025-05-08_15h37.47.862.csv"
data_mad = pd.read_csv(f"{data_dir}{file_name}")

fig_dir = "/media/maikito/mad_mini/consequential_task/figs/"

file_path = f"{data_dir}{file_name}"
data = import_psychopy(file_path)

# %% Exploratory analysis

# check mouse trajectory sampling
aux = []
for i in data.mouse_time:
    aux.append(np.diff(i))
aux = [j for i in aux for j in i]
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.hist(aux)
ax.text(
    0,
    5200,
    f"mean inter-sample duration: {round(np.mean(aux), 4)}\ncorresponding sampling rate: {round(1/np.mean(aux), 2)}",
)
save_name = fig_dir + "mouse_sample_duration.svg"
fig.savefig(save_name)
del aux
# conclusion: frame rate is not super consistent, but on average it's fairly accurate

# visualize decisions
fig = plt.figure(figsize=(10, 3))
plt.scatter(np.arange(len(data.trial)), data.chose_big, s=2, color="k")
ax = plt.gca()
ax.vlines(x=[60, 120, 180], ymin=0, ymax=1, color="k", linestyles="dashed")
ax.set_xlim(left=0, right=240)
xticks = [60, 120, 180, 240]
ax.set_xticks(xticks, list(map(str, xticks)))
yticks = [0, 1]
ax.set_yticks(yticks, ["small", "big"])
ax.set_ylabel("choice")
ax.set_xlabel("trial")
g_x = list(np.arange(30, 240, 60))
for i, g in enumerate(data.g.unique()):
    plt.text(g_x[i], 0.5, f"g = {g}", horizontalalignment="center")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "decisions.svg"
fig.savefig(save_name)


# reaction time vs. d, g, mean(R), pre/post-learning
data["rt"] = data.mouse_click_time - data.go
fig = plt.figure(figsize=(3, 2))
plt.hist(data.rt, color="k", fill=False)
ax = plt.gca()
ax.set_ylabel("count (trials)")
ax.set_xlabel("RT (s)")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "rt_hist.svg"
fig.savefig(save_name)


fig = plt.figure(figsize=(5, 5))
bp = sns.barplot(data=data, x="g", y="rt", color="k", fill=False)
ax = plt.gca()
ax.set_ylabel("RT (s)")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "rt_vs_g.svg"
fig.savefig(save_name)


fig = plt.figure(figsize=(5, 5))
bp = sns.barplot(data=data, x="d", y="rt", color="k", fill=False)
ax = plt.gca()
ax.set_ylabel("RT (s)")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "rt_vs_d.svg"
fig.savefig(save_name)


mean_stim_height = (data.left_stimulus_height + data.right_stimulus_height) / 2
fig = plt.figure(figsize=(5, 5))
plt.hist(mean_stim_height, color="k", fill=False)
ax = plt.gca()
ax.set_ylabel("count (trials)")
ax.set_xlabel(r"$\bar{R}$")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "r_hist.svg"
fig.savefig(save_name)


data["stim_mean"] = mean_stim_height
data["stim_mean_cat"] = stim_height_categories(mean_stim_height)
data["stim_mean_cat"] = pd.Categorical(
    data["stim_mean_cat"], categories=["low", "medium", "high"], ordered=True
)
fig = plt.figure(figsize=(5, 5))
bp = sns.barplot(data=data, x="stim_mean_cat", y="rt", color="k", fill=False)
ax = plt.gca()
ax.set_ylabel("RT (s)")
ax.set_xlabel(r"$\bar{R}$")
ax.spines[["right", "top"]].set_visible(False)
plt.tight_layout()
save_name = fig_dir + "rt_vs_r.svg"
fig.savefig(save_name)

# %% Big daddy figure
fig = plt.figure(figsize=(10, 5))
# create subfigures for first and second row
(row1fig, row2fig) = fig.subfigures(2, 1, height_ratios=[1, 1])
# split bottom row subfigure in two subfigures
(fig_row2left, fig_row2right) = row2fig.subfigures(
    1, 2, wspace=0.000, width_ratios=(1, 2)
)

# row 1 plots
row1_axs = row1fig.subplots(1, 1)
ax = row1_axs
ax.scatter(np.arange(len(data.trial)), data.chose_big, s=2, color="k")
ax.vlines(x=[60, 120, 180], ymin=0, ymax=1, color="k", linestyles="dashed")
ax.set_xlim(left=0, right=240)
xticks = [60, 120, 180, 240]
ax.set_xticks(xticks, list(map(str, xticks)))
yticks = [0, 1]
ax.set_yticks(yticks, ["small", "big"])
ax.set_ylabel("choice")
ax.set_xlabel("trial")
g_x = list(np.arange(30, 240, 60))
for i, g in enumerate(data.g.unique()):
    plt.text(g_x[i], 0.5, f"g = {g}", horizontalalignment="center")
ax.spines[["right", "top"]].set_visible(False)
letter_annotation(ax, -0.1, 1, "A")

row1fig.subplots_adjust(bottom=0.2)

# row 2 plots

axs = fig_row2left.subplots(2, 1)

ax = axs[0]
ax.hist(data.rt, color="k", fill=False)
ax.set_ylabel("count (trials)")
ax.set_xlabel("RT (s)")
ax.spines[["right", "top"]].set_visible(False)
letter_annotation(ax, -0.5, 1, "B")

ax = axs[1]
ax.hist(mean_stim_height, color="k", fill=False)
ax.set_ylabel("count (trials)")
ax.set_xlabel(r"$\bar{R}$")
ax.spines[["right", "top"]].set_visible(False)

fig_row2left.subplots_adjust(left=0.30, right=0.8, bottom=0.2, top=1, hspace=0.5)

axs = fig_row2right.subplots(1, 3, sharey=True)

ax = axs[0]
bp = sns.barplot(data=data, x="d", y="rt", color="k", fill=False, ax=ax)
ax.set_ylabel("RT (s)")
ax.spines[["right", "top"]].set_visible(False)
letter_annotation(ax, -0.4, 1, "C")


ax = axs[1]
bp = sns.barplot(data=data, x="g", y="rt", color="k", fill=False, ax=ax)
ax.set(ylabel="")
ax.spines[["right", "top"]].set_visible(False)

ax = axs[2]
bp = sns.barplot(data=data, x="stim_mean_cat", y="rt", color="k", fill=False, ax=ax)
ax.set_xlabel(r"$\bar{R}$")
ax.set(ylabel="")
ax.spines[["right", "top"]].set_visible(False)

fig_row2right.subplots_adjust(left=0, right=0.9, bottom=0.2, top=1)

save_name = fig_dir + "big_daddy.svg"
fig.savefig(save_name)

# %% Import data (pavlovia)
fig_dir = "/media/maikito/mad_mini/consequential_task/figs/"
pavlovia_dir = "/media/maikito/mad_mini/consequential_task/psychopy/gitlab_repos/consequential_task_A/data/"
file_name = "gloria2_consequential_2025-05-20_15h58.18.512.csv"
file_path = f"{pavlovia_dir}{file_name}"
data_gloria = pd.read_csv(file_path)

fig_dir = "/media/maikito/mad_mini/consequential_task/figs/"

data = import_pavlovia(file_path)
