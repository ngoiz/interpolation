import numpy as np
import sharpytools.linear.stability as stability


def produce_root_locus(dataset, param, value, **kwargs):
    list_of_constant_cases, params = retrieve_constant_parameter_cases(param, value, dataset.aeroelastic)

    eigs = []
    param_array = []

    for case in list_of_constant_cases:
        try:
            param_array.append(np.ones((case.eigs.shape[0], len(case.parameter_value))) * case.parameter_value)
            eigs.append(case.eigs)
        except TypeError:
            param_array.append(np.ones_like(case.eigs[:, 0]) * case.parameter_value)
        except AttributeError:
            continue

    param_array = np.concatenate(param_array)
    eigs = np.concatenate(eigs)

    try:
        u_inf_idx = dataset.parameter_name.index('u_inf')
    except AttributeError:
        u_inf_idx = dataset.param_name.index('u_inf')

    vels = param_array[:, u_inf_idx]

    conditions = stability.filter_velocity_eigenvalues(vels, eigs, **kwargs)
    return vels[conditions], eigs[conditions]


def produce_vgf(dataset, param, value, **kwargs):

    list_of_constant_cases, params = retrieve_constant_parameter_cases(param, value, dataset.aeroelastic)

    eigs = []
    param_array = []

    for case in list_of_constant_cases:
        try:
            param_array.append(np.ones((case.eigs.shape[0], len(case.parameter_value))) * case.parameter_value)
            eigs.append(case.eigs)
        except TypeError:
            param_array.append(np.ones_like(case.eigs[:, 0]) * case.parameter_value)
        except AttributeError:
            continue

    param_array = np.concatenate(param_array)
    eigs = np.concatenate(eigs)

    try:
        u_inf_idx = dataset.parameter_name.index('u_inf')
    except AttributeError:
        u_inf_idx = dataset.param_name.index('u_inf')

    try:
        kwargs['wdmax']
    except KeyError:
        kwargs['wdmax'] = 40 * 2 * np.pi
    vel, damp, fn = stability.modes(param_array[:, u_inf_idx],
                                    eigs, use_hz=True,
                                    **kwargs)

    return vel, damp, fn


def compute_flutter(dataset, param, value, instability_damping=0., vel_vmin=0., **kwargs):

    v, damp, _ = produce_vgf(dataset, param, value, **kwargs)

    if kwargs.get('alternative_flutter_function', False):
        return stability.find_flutter_speed(v, damp, instability_damping, vel_vmin)
    else:
        return stability.find_flutter_speed2(v, damp, instability_damping, vel_vmin)


def retrieve_constant_parameter_cases(param, value, dataset):
    """

    Args:
        param: name
        value: constant value
        dataset: Actual/Interpolated.aeroelastic

    Returns:

    """

    list_of_cases = []
    param_value = []

    for case in dataset:
        if case.case_info[param] == value:
            list_of_cases.append(case)
            param_value.append(case.case_info['u_inf'])  # for now....

    # sort them out
    order = np.argsort(np.array(param_value))
    list_of_cases = [list_of_cases[i_order] for i_order in order]
    param_value = np.array([param_value[i_order] for i_order in order])

    return list_of_cases, param_value
