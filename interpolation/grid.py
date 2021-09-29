import numpy as np
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import os
import logging
import interpolation.interface as interface
import glob


class Grid:

    def __init__(self, settings):
        self.library = None  # pmor.library
        self.settings = settings
        self.path_to_library = None  # str: path to pickle containg pmor library
        self.source_case_directory = None  # str: directory from where to source other cases

        self.logger = logging.getLogger(__name__)

    def add_point(self, point_parameters, siminfo, source_cases_path):
        """

        Args:
            point_parameters (list or np.array): Parametric value of new point
            siminfo (interpolation.__main__.SimInfo): Simulation Info class to generate case names
            source_cases_path (str): path in which to find cases

        Returns:

        """
        # to-do; add point to grid, if already exists do not add
        pass


class Database(Grid):

    def __init__(self, settings):
        super().__init__(settings)

        self.save_pickle = True

    def load(self, siminfo=None, *source_cases_path):
        """
        Creates or loads a pmor library pkl from the settings
            library_path and library_name

        Args:
            siminfo (interpolation.__main__.SimInfo (optional)): SimInfo object to run sharpy at missing data points
            source_cases_path (str or list(str) (optional)): Locations where to look for cases

        """

        library_pickle_folder = self.settings['library_path']
        library_pickle_name = self.settings['library_name']
        if library_pickle_name.split('.')[-1] == 'pkl':
            library_pickle_name = ''.join(library_pickle_name.split('.')[:-1])
        library_settings = {'folder': library_pickle_folder,
                            'library_name': library_pickle_name}

        self.library = pmorlibrary.ROMLibrary()

        self.path_to_library = library_settings['folder'] + '/' + library_settings['library_name'] + '.pkl'
        self.library.parameters = [p_info['name'] for p_info in siminfo.parameters.values()]
        if os.path.exists(self.path_to_library):
            self.library.load_library(library_settings['folder'] + '/' + library_settings['library_name'] + '.pkl')
            self.library.folder = library_settings['folder']
            self.library.library_name = library_settings['library_name']
        else:
            self.library.create(library_settings)
            if self.save_pickle:
                self.library.save_library()

        # add extra points if needed - from points setting
        try:
            points_input = self.settings['points']
        except KeyError:
            pass
        else:
            if type(points_input) is str:
                points = interface.yaml_to_array(points_input)
            elif type(points_input) is list:
                points = points_input
            else:
                raise TypeError(f'`points` settings is neither path to yaml nor list. It is type {type(points_input)}')

            if siminfo is not None:
                for point in points:
                    self.add_point(point, siminfo, *source_cases_path)
            else:
                self.logger.warning('Provide a SimInfo object to run sharpy at missing points')

    def add_point(self, point_parameters, siminfo, *source_cases_path):

        point_info_list = sort_point_input(point_parameters)

        for point_info in point_info_list:
            self.add_individual_point(point_info, siminfo, *source_cases_path)

    def add_individual_point(self, point_parameters, siminfo, *source_cases_path):
        """

        Args:
            point_parameters (list or np.array or dict): List or array containing the parameters of the desired point
            siminfo (interpolation.__main__.SimulationInfo):
            source_cases_path (str): Path(s) in which to find cases, separated by commas

        Returns:

        """
        self.logger.info(f'Adding new point to database. Point is {point_parameters}')

        point_parameters = siminfo.parameter_sigfig(point_parameters)

        target_case_name = siminfo.case_name_generator(point_parameters)

        if type(source_cases_path) is not list:
            source_cases_path = list(source_cases_path)

        try:
            folder = siminfo.settings['simulation_settings']['output_folder']
        except KeyError:
            pass
        else:
            if folder not in source_cases_path:
                source_cases_path.append(folder)

        try:
            source_cases_path.append(self.settings['source_cases_path'])
        except KeyError:
            pass

        source_cases_path = flatten(source_cases_path)

        case_exists = False
        for path in source_cases_path:
            if sharpy_case_exists(path, case_name=target_case_name):
                case_exists = True
                src_path = path
                break
        if not case_exists:
            siminfo.run_sharpy(point_parameters)
            src_path = siminfo.settings['simulation_settings']['output_folder']

        self.library.load_case(src_path + '/' + target_case_name)
        if self.save_pickle:
            self.library.save_library()

    def __call__(self, *args):
        """
        Produces an array of points

        Args:
            *args: Optional first argument is string for yaml output folder

        Returns:
            np.array: ``n_points x n_parameter`` array of points
        """
        members_array = self.library.entries_array()
        if len(args) == 1 and type(args[0]) is str:
            self.library.save_as_yaml(args[0])
        return members_array


def sharpy_case_exists(path, case_name):
    """
    Checks whether a SHARPy case exists (return True) or not (return False)

    Args:
        path (str): Path in which to look for case
        case_name (str): Case name

    Returns:
        bool: True if exists, else False
    """
    if not os.path.isdir(path + '/' + case_name):
        logging.info(f'Case {case_name} not found at {path}')
        return False
    else:
        logging.info(f'Case {case_name} has been found at {path}')
        return True


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def sort_point_input(point_parameters):
    # sort out all possible input types...
    if type(point_parameters) is list:
        try:
            point_parameters[0][0]
        except TypeError:
            # single point only; occurs if the entry is a float or int and cannot be accessed
            point_info_list = [point_parameters]
        except IndexError:
            # single point only; occurs if the entry is a float or int and cannot be accessed
            point_info_list = [point_parameters]
        except KeyError:
            # the first entry is then a dictionary, so its a list(dict) which is ok
            point_info_list = point_parameters
        else:
            # input is list of lists and that is ok
            point_info_list = point_parameters
    elif type(point_parameters) is dict:
        point_info_list = [point_parameters]
    elif type(point_parameters) is np.ndarray:
        shape = point_parameters.shape
        if len(shape) > 1:
            # for the case the input is a 2D array
            point_info_list = [point_parameters[i_entry, :] for i_entry in range(shape[0])]
        else:
            point_info_list = [point_parameters]
    else:
        raise TypeError(f'Adding point. Cannot figure out what to do with input of type {type(point_parameters)}')

    return point_info_list

