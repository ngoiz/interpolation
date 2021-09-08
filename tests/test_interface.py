import unittest
import interpolation.interface as interface
import os
import shutil
import numpy as np
import interpolation.__main__ as interpolation_main
import configobj


class TestInterface(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    test_directories = []  # list of directories in which tests are created/output

    def test_sharpy_interface(self):
        test_directory = self.route_test_dir + '/test_sharpy_interface/'
        self.test_directories.append(test_directory)

        parameters = {'alpha': 2,
                      'u_inf': 10}

        simulation_settings = {
            'cases_subfolder': test_directory + '/cases/',
            'output_folder': test_directory + '/output/',
            'gravity_on': True,
            'skin_on': True,
            'M': 4,
            'N': 1,
            'Ms': 5,
            'inout_coordinates': 'modes',
            'flow': ['BeamLoader', 'BeamPlot'],
                               }
        interface.run_sharpy('test_case', parameters, simulation_settings)

        # check that directories exist
        directories = [simulation_settings['cases_subfolder'],
                       simulation_settings['output_folder']]
        for directory in directories:
            with self.subTest(f'Checking {directory}'):
                assert os.path.isdir(directory), f'Directory {directory} could not be found'

    def test_interpolation_sharpy(self):
        case_name = 'test_interpolation'
        parameters = ['alpha', 'u_inf']
        test_directory = self.route_test_dir + '/test_interpolation_sharpy/'
        self.test_directories.append(test_directory)
        if not os.path.isdir(test_directory):
            os.makedirs(test_directory)

        interpolation_cases_folder = test_directory + '/cases/'
        interpolation_output_folder = test_directory + '/output/'
        source_cases_folder = test_directory + '/source_cases_folder/'
        if not os.path.isdir(source_cases_folder):
            os.makedirs(source_cases_folder)    # need to create this one to trick sharpy
                                                # in believing there are source cases there

        simulation_settings = {'reference_case': 0,
                               'projection_method': 'weakMAC',
                               'interpolation_space': 'direct'}

        interpolation_points = np.array([[5, 10],
                                         [5, 30]])

        input_points_yaml_file = test_directory + '/interpolation_input.yaml'
        with self.subTest('Creating input yaml file'):
            interface.array_to_yaml(parameters, interpolation_points, input_points_yaml_file)

            assert os.path.exists(input_points_yaml_file), \
                f'Unable to find newly created yaml at {input_points_yaml_file}'

        # empty library requesting an input - not yet sure how to handle
        with self.subTest('SHARPy PMOR'):
            # this test will fail within SHARPy because there are no source cases
            # interface.run_sharpy_interpolation(case_name,
            #                                    cases_folder=interpolation_cases_folder,
            #                                    output_folder=interpolation_output_folder,
            #                                    source_cases_folder=None,
            #                                    input_file=input_points_yaml_file,
            #                                    simulation_settings=simulation_settings)
            pass

    def test_siminfo_run_sharpy(self):
        test_directory = self.route_test_dir + '/test_siminfo_sharpy_interface/'
        self.test_directories.append(test_directory)

        parameters = {'alpha': 2,
                      'u_inf': 10}

        temp_cases = self.route_test_dir + '/src/cases/'  # change this to real directory!
        temp_output = self.route_test_dir + '/src/output/'
        sharpy_simulation_settings = {
            'cases_subfolder': temp_cases,
            'output_folder': temp_output,
            'gravity_on': True,
            'skin_on': True,
            'M': 4,
            'N': 1,
            'Ms': 5,
            'inout_coordinates': 'modes',
            'flow': ['BeamLoader', 'SaveParametricCase'],
        }

        input_settings = {'simulation_name': 'test_simulation_info',
                          'directory': test_directory,
                          'parameters': [
                              {'name': 'alpha',
                               'format_scaling': 100},
                              {'name': 'u_inf',
                               'format_scaling': 10}],
                          'case_name_format': 'pazy_uinf{:04g}_alpha{:04g}',
                          'case_name_parameters': ['u_inf', 'alpha'], # in case order does not match parameter order
                          'simulation_settings': sharpy_simulation_settings}

        sim_info = interpolation_main.SimulationInfo(input_settings['simulation_name'],
                                                     input_settings['directory'],
                                                     input_settings)
        with self.subTest('Correct test name generation'):
            param_indices = sim_info.parameter_index
            param_info = sim_info.parameters

            case_name = sim_info.case_name_generator(parameters)
            target_case_name = input_settings['case_name_format'].format(*[
                parameters['u_inf'] * param_info[param_indices['u_inf']]['format_scaling'],
                parameters['alpha'] * param_info[param_indices['alpha']]['format_scaling'],
                                                                           ]
            )

            assert case_name == target_case_name, f'Case names not generated properly:\n\t' \
                                                  f'{case_name} does not match target: {target_case_name}'

        sim_info.run_sharpy(parameters)
        pmor_out = configobj.ConfigObj(
            sharpy_simulation_settings['output_folder'] + '/' + case_name + '/' + case_name + '.pmor.sharpy')

        for param_name, param_value in pmor_out['parameters'].items():
            with self.subTest('Incorrect SHARPy parameter setting test', param_name=param_name):
                assert float(parameters[param_name]) - float(param_value) < 1e-6, \
                    'Case has not properly run ' \
                    f'{param_name} is {param_value} rather ' \
                    f'than {parameters[param_name]}'

    @classmethod
    def tearDownClass(cls):

        for direc in cls.test_directories:
            shutil.rmtree(direc)
