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
import importlib.util
import configobj
import os
import sharpy.sharpy_main


# SHARPy interfaces
def run_sharpy(case_name, parameters, simulation_settings):
    """
    The user needs to provide a path to a module that runs the desired SHARPy case
    this module needs to have a generate_pazy function

    Args:
        case_name (str): Name of the case to be run
        parameters (dict): Dictionary containing parameter names and values
        simulation_settings (dict): all other SHARPy simulation settings required by the
            ``generate_pazy`` function

    Raises:
        AttributeError: If the specified module does not contain a ``generate_pazy`` function

    Returns:

    """
    py_file_path = '/home/ng213/2TB/pazy_code/pazy-sharpy/07_Interpolation/generate_single_speed.py'
    spec = importlib.util.spec_from_file_location('run_sharpy', py_file_path)
    sharpy_case_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharpy_case_module)

    kwargs = {**parameters, **simulation_settings}

    try:
        sharpy_case_module.generate_pazy(case_name,
                                         **kwargs)
    except AttributeError:
        raise AttributeError('Ensure specified file contains the proper function name and signature')


def run_sharpy_interpolation(case_name, cases_folder, output_folder, source_cases_folder, input_file,
                             input_library=None, simulation_settings=None):

    if not os.path.isdir(cases_folder):
        os.makedirs(cases_folder)

    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': cases_folder,
                          'flow': ['ParametricModelInterpolation'],
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': output_folder,
                          'log_file': case_name + '.log'}

    settings['ParametricModelInterpolation'] = {
        'reference_case': simulation_settings.get('reference_case', 3),
        'interpolation_system': 'aeroelastic',
        'input_file': input_file,
        'cleanup_previous_cases': 'on',
        # 'projection_method': 'strongMAC',
        # 'independent_interpolation': 'on',
        'interpolation_settings': {
            # 'aerodynamic': {
            # 'projection_method': 'weakMAC',
            # 'interpolation_space': 'direct',
            # },
            'aeroelastic': {
                'projection_method': simulation_settings.get('projection_method', 'weakMAC'),
                'interpolation_space': simulation_settings.get('interpolation_space', 'direct'),
            },
            # 'structural': {
            # 'projection_method': 'weakMAC',
            # 'interpolation_space': 'direct',
            # },
        },
        'interpolation_scheme': 'linear',
        # 'interpolation_degree': 1,
        'postprocessors': ['AsymptoticStability', 'FrequencyResponse'],
        'postprocessors_settings': {'AsymptoticStability': {'print_info': 'on',
                                                            'export_eigenvalues': 'on',
                                                            },
                                    'FrequencyResponse': {'print_info': 'on',
                                                          'frequency_bounds': [1e-2,
                                                                               1e3],
                                                          'num_freqs': 100,
                                                          'frequency_spacing': 'log',
                                                          'compute_hinf': 'off',
                                                          'frequency_unit': 'w'
                                                          },
                                    'SaveStateSpace':
                                        {'target_system': ['aeroelastic'],
                                         'print_info': 'on',
                                         },
                                    }}

    if input_library is not None:
        settings['ParametricModelInterpolation']['library_filepath'] = input_library
    else:
        settings['ParametricModelInterpolation']['cases_folder'] = source_cases_folder

    config = configobj.ConfigObj()
    file_name = cases_folder + case_name + '.sharpy'
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

    sharpy.sharpy_main.main(['', cases_folder + '/' + case_name + '.sharpy'])


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


def array_to_yaml(parameters, data_grid, output_file):
    """
    Generates a YAML input file to be used by SHARPy's PMOR module.

    Each entry in the yaml file contains the parameter name and its values

    Args:
        parameters (list(str)): List of parameter names as strings
        data_grid (np.array): Array of dimensions (n_points, n_params) containing the interpolation points
        output_file (str): Location where to save the YAML file
    """

    data_list = []
    for i_point in range(data_grid.shape[0]):
        data_list.append({param_name: float(data_grid[i_point, param_index])
                          for param_index, param_name in enumerate(parameters)})

    dict_list_to_yaml(data_list, output_file)


def dict_list_to_yaml(parameters, output_file):
    """
    Write a list of dictionaries to yaml file

    Args:
        parameters (list(dict)): List of dictionaries with key (param_name): value pairs
        output_file (str): output yaml file

    """
    with open(output_file, 'w') as f:  # creates or appends to file - might be a good flag option for the user to set
        f.write(yaml.safe_dump(parameters))
