from dataclasses import dataclass
from typing import List

import numpy as np
import yaml
from ypstruct import struct


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
