import os
import interpolation.grid as grid
import logging
import interpolation.interface as interface


class Suite:
    """
    Interpolation Suite
    """
    def __init__(self):

        self.logger = logging.getLogger(__name__)

        self.simulation_info = None  # interpolation.__main__.SimInfo

        self.parameters = dict()  # dict: idx: param_info (dict)
        self.parameter_index = dict()  # dict: param_name: idx

        self.training_data = None  # grid.Grid with training points
        self.testing_data = None   # grid.Grid with testing points

        self.cost_function = None

    def create(self, siminfo):
        """
        Creates an interpolation suite.

        1. Create folder structure

        Args:
            siminfo (interpolation.__main__.SimInfo)
            input_data (dict): With keys:
                parameters: list of parameter information dictionary (name, format_scaling, index)

        Returns:

        """
        self.simulation_info = siminfo

        self._create_directories()

        for ith, parameter_info in enumerate(self.simulation_info.settings['parameters']):
            self.parameters[ith] = parameter_info
            self.parameter_index[parameter_info['name']] = parameter_info.get('index', ith)

        self.training_data = self._create_grid(self.simulation_info.settings['training_data'])
        self.testing_data = self._create_grid(self.simulation_info.settings['testing_data'])

    def interpolate(self):
        interpolation_case_folder = self.simulation_info.path + '/interpolation_cases/'
        interpolation_output_folder = self.simulation_info.path + '/interpolation_output/'
        for direc in [interpolation_case_folder, interpolation_output_folder]:
            if not os.path.isdir(direc):
                os.makedirs(direc)

        case_name = 'interpolation_' + self.simulation_info.name

        interpolation_points_file = interpolation_case_folder + '/' + case_name + '.yaml'
        self.logger.info(f'Starting interpolation - interpolation points input {interpolation_points_file}')

        # write yaml file
        self.testing_data(interpolation_points_file)

        self.run_interpolation(case_name=case_name,
                               interpolation_case_folder=interpolation_case_folder,
                               interpolation_output_folder=interpolation_output_folder,
                               interpolation_points_file=interpolation_points_file)

    def run_interpolation(self, case_name, interpolation_case_folder, interpolation_output_folder,
                          interpolation_points_file):
        interface.run_sharpy_interpolation(
            case_name,
            interpolation_case_folder,
            interpolation_output_folder,
            source_cases_folder=None,
            input_file=interpolation_points_file,
            input_library=self.training_data.path_to_library,
            simulation_settings=self.simulation_info.settings['interpolation_settings']
        )

    def _create_directories(self):
        directory_names = ['training_cases', 'training_output', 'testing_output', 'testing_cases']
        for direc in directory_names:
            if not os.path.isdir(self.simulation_info.path + '/' + direc):
                os.makedirs(self.simulation_info.path + '/' + direc)

    def _create_grid(self, grid_settings, save_pickle=True):
        """

        Args:
            grid_settings:
            save_pickle (bool): create a grid that saves a pickle or not

        Returns:

        """
        if grid_settings['type'].lower() == 'database':
            database = grid.Database(settings=grid_settings)
            database.save_pickle = save_pickle

            try:
                sim_output_folder = self.simulation_info.settings['simulation_settings']['output_folder']
            except KeyError:
                sim_output_folder = None

            database.load(self.simulation_info,
                          grid_settings.get('source_cases_path', None),
                          sim_output_folder)
            return database

    def interpolate_at_point(self, point_info, case_name, cases_folder, output_folder):
        """Interpolate at single point to be used with optimisation"""

        # create temporary testing grid that will not be saved and will only contain this one point
        temp_testing_data = self._create_grid(self.simulation_info.settings['testing_data'], save_pickle=False)
        temp_testing_data.add_point(point_info, self.simulation_info,
                                    self.simulation_info.settings['testing_data']['source_cases_path'],
                                    self.simulation_info.settings['simulation_settings']['output_folder'])

        for direc in [cases_folder, output_folder]:
            if not os.path.isdir(direc):
                os.makedirs(direc)

        input_file_name = cases_folder + '/input.yaml'
        temp_testing_data(input_file_name)

        self.run_interpolation(case_name, cases_folder, output_folder, input_file_name)

        interpolation_info = {'testing_library': temp_testing_data,
                              'interpolation_output_directory': output_folder}

        return interpolation_info
