import unittest
import interpolation.__main__ as interpolation_main
import os
import shutil


class TestBasic(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    output_directory = route_test_dir + '/output/'

    def test_directory_creation(self):
        input_settings = {'simulation_name': 'test_directory_creation',
                          'directory': self.output_directory,
                          'parameters': [
                              {'name': 'parameter1',
                               'format_scaling': 10},
                              {'name': 'parameter2',
                               'format_scaling': 1}],
                          'case_name_format': 'case_p1{:04g}_p2{:04g}'}

        sim_directory = os.path.abspath(input_settings['directory'] + '/' + input_settings['simulation_name'])

        input_settings['training_data'] = \
            {'type': 'database',
             'library_path': sim_directory,
             'library_name': 'training_library'}

        input_settings['testing_data'] = \
            {'type': 'database',
             'library_path': sim_directory,
             'library_name': 'testing_library'}

        interpolation_main.main(input_settings)

        with self.subTest('Directory creation'):
            assert os.path.isdir(sim_directory)

        with self.subTest('Log file exists'):
            expected_log_directory = sim_directory + '/' + input_settings['simulation_name'] + '.log'
            assert os.path.exists(expected_log_directory), 'Unable to find log at {:s}'.format(expected_log_directory)

        subdirectory_names = ['training_cases', 'training_output', 'testing_output', 'testing_cases']
        for directory_name in subdirectory_names:
            with self.subTest(directory_name):
                expected_directory = sim_directory + '/' + directory_name
                assert os.path.isdir(expected_directory), f'Expected {directory} not found'

        training_settings = input_settings['training_data']
        with self.subTest('Check library exists'):
            expected_library_directory = training_settings['library_path'] + '/' + training_settings['library_name'] + '.pkl'
            assert os.path.exists(expected_library_directory), \
                f'Library has not been created at {expected_library_directory}'

    def test_sim_info(self):
        input_settings = {'simulation_name': 'test_simulation_info',
                          'directory': self.output_directory,
                          'parameters': [
                              {'name': 'parameter1',
                               'format_scaling': 10},
                              {'name': 'parameter2',
                               'format_scaling': 1}],
                          'case_name_format': 'case_p1{:04g}_p2{:04g}'}

        sim_info = interpolation_main.SimulationInfo(input_settings['simulation_name'],
                                                     input_settings['directory'],
                                                     input_settings)

        p1_value, p2_value = 5, 10
        with self.subTest(msg='Case generation', p1_value=p1_value, p2_value=p2_value):
            case_name = sim_info.case_name_generator([p1_value, p2_value])

            true_case_name = input_settings['case_name_format'].format(p1_value * 10,
                                                                       p2_value * 1)

            assert case_name == true_case_name, f'Case name not generated correctly:\n\t' \
                                                f'{case_name} should read {true_case_name}'

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_directory)
