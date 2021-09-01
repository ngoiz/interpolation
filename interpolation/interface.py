"""Interface utilities

with SHARPy:
    - run normal cases
    - run interpolation

with YAML:
    - read yaml to array
    - write array to yaml

"""
import yaml
import numpy as np


# YAML utilities
def load_yaml(yaml_file_name):
    """
    Loads yaml file to list of dictionaries

    Args:
        yaml_file_name (str): Path to YAML file containing input of parameters.

    Returns:
        list: List of dictionaries
    """
    with open(yaml_file_name, 'r') as yaml_file:
        out_dict = yaml.load(yaml_file, Loader=yaml.Loader)
        yaml_file.close()
    return out_dict


def yaml_to_array(yaml_file):
    """
    Read a yaml file containing a list of parameter-value pairs to an array

    Args:
        yaml_file:

    Returns:
        np.array
    """
    data_list = load_yaml(yaml_file)

    n_points = len(data_list)
    n_params = len(data_list[0])
    array = np.zeros((n_points, n_params))
    for ith, entry in enumerate(data_list):
        array[ith, :] = [entry[param_name] for param_name in entry.keys()]

    return array

