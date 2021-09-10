import unittest
import os
import shutil
import interpolation.__main__ as interpolation_main
import interpolation.suite as suite


class TestInterpolationSuite(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    test_directories = []

    def test_interpolation_suite(self):
        test_directory = self.route_test_dir + '/test_interpolation_suite/'
        self.test_directories.append(test_directory)

        training_grid_settings = {'type': 'database',
                                  'library_path': test_directory,
                                  'library_name': 'training_library',
                                  'source_cases_path': self.route_test_dir + '/src/output/',
                                  'points': [[2, 10],
                                             [2, 80],
                                             [5, 10],
                                             [5, 80]]}

        testing_grid_settings = {'type': 'database',
                                 'library_path': test_directory,
                                 'library_name': 'testing_library',
                                 'source_cases_path': self.route_test_dir + '/src/output/',
                                 'points': [[3, 20],
                                            [4, 50]]}

        sharpy_simulation_settings = {
            'cases_subfolder': test_directory + '/cases/',
            'output_folder': test_directory + '/output/',
            'gravity_on': True,
            'skin_on': True,
            'M': 4,
            'N': 1,
            'Ms': 5,
            'inout_coordinates': 'modes',
            # 'flow': ['BeamLoader', 'SaveParametricCase'],
        }

        interpolation_simulation_settings = {'reference_case': 0,
                                             'projection_method': 'weakMAC',
                                             'interpolation_space': 'direct'}

        input_settings = {'simulation_name': 'test_interpolation_suite',
                          'directory': test_directory,
                          'parameters': [
                              {'name': 'alpha',
                               'format_scaling': 100},
                              {'name': 'u_inf',
                               'format_scaling': 10}],
                          'case_name_format': 'pazy_uinf{:04g}_alpha{:04g}',
                          'case_name_parameters': ['u_inf', 'alpha'], # in case order does not match parameter order
                          'simulation_settings': sharpy_simulation_settings,
                          'interpolation_settings': interpolation_simulation_settings,
                          'testing_data': testing_grid_settings,
                          'training_data': training_grid_settings,
                          }

        interpolation_main.main(settings=input_settings)

        with self.subTest('Log file exists'):
            expected_log_directory = input_settings['directory'] + '/' + \
                    input_settings['simulation_name'] + '/' \
                    + input_settings['simulation_name'] + '.log'
            assert os.path.exists(expected_log_directory), 'Unable to find log at {:s}'.format(expected_log_directory)

    def test_suite_methods(self):
        test_directory = self.route_test_dir + '/test_suite_methods/'
        self.test_directories.append(test_directory)

        training_grid_settings = {'type': 'database',
                                  'library_path': test_directory,
                                  'library_name': 'training_library',
                                  'source_cases_path': self.route_test_dir + '/src/output/',
                                  'points': [[2, 10],
                                             [2, 80],
                                             [3, 10],
                                             [3, 80],
                                             [5, 10],
                                             [5, 80]]}

        testing_grid_settings = {'type': 'database',
                                 'library_path': test_directory,
                                 'library_name': 'testing_library',
                                 'source_cases_path': self.route_test_dir + '/src/output/'}

        sharpy_simulation_settings = {
            'cases_subfolder': test_directory + '/cases/',
            'output_folder': test_directory + '/output/',
            'gravity_on': True,
            'skin_on': True,
            'M': 4,
            'N': 1,
            'Ms': 5,
            'inout_coordinates': 'modes',
            # 'flow': ['BeamLoader', 'SaveParametricCase'],
        }

        interpolation_simulation_settings = {'reference_case': 0,
                                             'projection_method': 'weakMAC',
                                             'interpolation_space': 'direct'}

        input_settings = {'simulation_name': 'test_interpolation_suite',
                          'directory': test_directory,
                          'parameters': [
                              {'name': 'alpha',
                               'format_scaling': 100},
                              {'name': 'u_inf',
                               'format_scaling': 10}],
                          'case_name_format': 'pazy_uinf{:04g}_alpha{:04g}',
                          'case_name_parameters': ['u_inf', 'alpha'],  # in case order does not match parameter order
                          'simulation_settings': sharpy_simulation_settings,
                          'interpolation_settings': interpolation_simulation_settings,
                          'testing_data': testing_grid_settings,
                          'training_data': training_grid_settings,
                          }

        siminfo = interpolation_main.SimulationInfo(directory=input_settings['directory'],
                                                    name=input_settings['simulation_name'],
                                                    settings=input_settings)

        interpolation_suite = suite.Suite()
        interpolation_suite.create(siminfo)
        interpolation_suite.initialise_cost_function('EigenvalueComparison')

        point_info = {'alpha': 3, 'u_inf': 20}

        interpolation_suite.evaluate(point_info)

    @classmethod
    def tearDownClass(cls):
        for direc in cls.test_directories:
            shutil.rmtree(direc)


if __name__ == '__main__':
    unittest.main()
