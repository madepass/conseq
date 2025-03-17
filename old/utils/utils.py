from dataclasses import dataclass
from typing import List

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from ypstruct import struct


def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Compute the filter coefficients for a Butterworth bandpass filter.
    """
    # Compute the Nyquist frequency
    nyq = 0.5 * fs
    # Compute the low and high frequencies
    low = lowcut / nyq
    high = highcut / nyq
    # Compute the filter coefficients
    b, a = butter(order, [low, high], btype="band")
    # Return the filter coefficients
    return b, a


def bandpass_filter(lfp, fs, lowcut, highcut):
    """
    Apply a bandpass filter to the LFP signal.
    """
    # Compute the filter coefficients
    b, a = butter_bandpass(lowcut, highcut, fs)
    # Apply the filter
    lfp_filtered = filtfilt(b, a, lfp, axis=1)
    # Return the filtered LFP signal
    return lfp_filtered


def import_config():
    """Imports & parses config YAML file. First two levels are MATLAB-like structs"""
    # os.system("pwd")
    with open("./config.yaml", "r") as ymlFile:  # put in function
        cfg = struct(yaml.safe_load(ymlFile))
    for k in cfg:
        if type(cfg[k]) == dict:
            cfg[k] = struct(cfg[k])
    print("Imported config.yaml")
    return cfg

# neural data dataclass
@dataclass
class NeuralData():
    """Class for neural data"""
    data: np.ndarray
    sf: int
    channels: list
    subj_dir: str
    fn: str

    def __post_init__(self):
        self.n_channels = len(self.channels)
        self.n_samples = self.data.shape[1]
        self.n_seconds = self.n_samples / self.sf
        self.n_minutes = self.n_seconds / 60
        self.n_hours = self.n_minutes / 60
        self.n_days = self.n_hours / 24

# behavior data dataclass
@dataclass
class BehaviorData():
    decisions: np.ndarray
    rt: np.ndarray
    correct: np.ndarray

@dataclass
class Dataset():
    data = List[NeuralData]
