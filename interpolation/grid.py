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
        """

        Args:
            point_parameters (list or np.array or dict): List or array containing the parameters of the desired point
            siminfo (interpolation.__main__.SimInfo):
            source_cases_path (str): Path(s) in which to find cases, separated by commas

        Returns:

        """
        self.logger.info(f'Adding new point to database. Point is {point_parameters}')

        target_case_name = siminfo.case_name_generator(point_parameters)

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
