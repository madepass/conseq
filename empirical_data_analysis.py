# %% Imports
import numpy as np
import pandas as pd


# %% Import data
data_dir = "/media/maikito/mad_mini/consequential_task/psychopy/h1_v2/data/saved/"
file_name = "mad_test2_consequential_2025-05-08_15h37.47.862.csv"
data_mad = pd.read_csv(f"{data_dir}{file_name}")

data_mad.groupby("g_block").count()

# identify useful columns
data_rm["mouse_2.time"]
data_rm["trials_within_episode.mouse_2.x"]
data_rm["trials_within_episode.mouse_2.y"]
data_rm["trials_within_episode.mouse_2.clicked_name"]

data_rm["trials_within_episode.thisN"]  # is this trial within episode?
data_rm["right_stimuli_outline_1.stopped"]

# %%
file_path = f"{data_dir}{file_name}"


def import_psychopy(file_path: str) -> pd.DataFrame:
    """
    Import psychopy .csv
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
    new_env_idx = list(np.where(aux == False)[0])
    remove_idx += new_env_idx[1:]
    aux = list(np.where(df["thisN"] == 3)[0])
    experiment_end_ind = aux
    remove_idx += experiment_end_ind
    df = df.drop(df.index[remove_idx], axis=0)
    df.reset_index(inplace=True)
    print(df)
    # what columns do we want to keep?
    breakpoint()
    trial_number = list(np.arange(df.shape[0]) + 1)
    df["trial"] = trial_number
    keep_columns = [
        "trial",
        "g_block",
        "thisN",
        "mouse_2.x",
        "mouse_2.y",
        "mouse_2.leftButton",
        "mouse_2.time",
        "mouse_2.clicked_name",
        "episodes.thisN",
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
    ]
    # mouse trajectories
    df["mouse_2.x"]
    df["mouse_2.y"]
    # choice data

    return df


data = import_psychopy(file_path)


# %% Import psydata
file_name = "riccardo_consequential_2025-04-24_18h09.56.705.psydat"
psydata = fromFile(f"{data_dir}{file_name}")
save_name = f"{data_dir}psydat_widetext.csv"
psydata.saveAsWideText(save_name, delim=",")
