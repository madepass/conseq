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
