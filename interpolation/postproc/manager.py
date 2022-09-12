import pickle
import numpy as np
import os
import interpolation.interface as interface
import h5py
import glob
import sharpy.utils.h5utils as h5utils
import pandas as pd


def pickle_to_array(case_name, path=None):
    if path is None:
        filepath = f'/home/ng213/2TB/pazy_code/pazy-sharpy/07_Interpolation/{case_name}/training_library.pkl', 'rb'
    else:
        filepath = path

    with open(filepath, 'rb') as f:
        lib = pickle.load(f)

    training_points = np.zeros((len(lib[0]), 2))
    for ith, entry in enumerate(lib[0]):
        training_points[ith] = np.array([entry['parameters']['alpha'], entry['parameters']['u_inf']])

    return training_points


def set_as_string_to_set(string):
    string = string.strip('{}')  # remove trailing curly braces
    members = []
    string_entries = string.split('),')  # split at end of tuple
    for entry in string_entries:
        list_entry = entry.strip('() ')  # remove brackets and white spaces
        members.append(tuple([float(i) for i in list_entry.split(',')]))  # convert to float and add to list

    return set(members)


def cost_string_to_array(cost_string):
    return np.array([float(i) for i in cost_string.strip('[]').split()])


def get_cost(case_info_input, set_id):
    cost_string = case_info_input.loc[set_id]['cost']
    return cost_string_to_array(cost_string)


def get_points(case_info_input, set_id, which_set='training'):
    if which_set == 'training':
        try:
            set_points = set_as_string_to_set(case_info_input.loc[set_id]['training_points'])
        except KeyError:
            print('Set id (training):', set_id)
#             print(case_info_input[case_info_input['set_id'] == set_id]['training_points'])
    elif which_set == 'testing':
        set_points = set_as_string_to_set(case_info_input.loc[set_id]['points_in_hull'])
    else:
        raise NameError
    return np.array([p for p in set_points])


def load_case_info(root_directory, case_name):
    """
    Returns:
        dict: containing case_info[set_id]['data' or 'info']
    """

    directory = root_directory + case_name

    case_info = {}
    n_iterations = len(glob.glob(root_directory + case_name + '/sampling_iteration*'))
    print(f'Found {n_iterations} iterations to process')
    for iteration_id in range(n_iterations):
        set_data_file = directory + '/sampling_iteration{:02g}/set_data.h5'.format(iteration_id)
        if os.path.exists(set_data_file):
            case_info[iteration_id] = {}
            with h5py.File(set_data_file, 'r') as f:
                case_info[iteration_id]['data'] = h5utils.load_h5_in_dict(f)
        else:
            if iteration_id == n_iterations - 1:
                print('Unable to find set data - iteration likely running')
                n_iterations -= 1
                break
            else:
                raise OSError('Error did not happen in last iteration..., iter_id{:g}'.format(iteration_id))

        training_points_yaml = directory + f'/sampling_iteration{iteration_id:02g}/fixed_interpolation_datapoints.yaml'
        training_points = interface.yaml_to_array(training_points_yaml)
        xt, yt = training_points[:, 0], training_points[:, 1]
        known_yaml = directory + f'/sampling_iteration{iteration_id:02g}/known_datapoints.yaml'
        known_points = interface.yaml_to_array(known_yaml)
        xk, yk = known_points[:, 0], known_points[:, 1]
        case_info[iteration_id]['points'] = {'training': (xt, yt),
                                             'known': (xk, yk)}

        case_info[iteration_id]['info'] = pd.read_pickle(
            directory + '/sampling_iteration{:02g}'.format(iteration_id) + '/sampling_report.pkl')
        case_info[iteration_id]['info'] = case_info[iteration_id]['info'].set_index('set_id')
    case_info['n_iterations'] = n_iterations
    return case_info