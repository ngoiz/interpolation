import unittest
import interpolation.optimiser as optimiser
import shutil
import os
import interpolation.__main__ as interpolation_main
import interpolation.suite as suite
import interpolation.evaluation as evaluation


class TestOptimiser(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    test_directories = []

    @unittest.skip('Long test to run optimisation')
    def test_optimiser(self):
        test_directory = self.route_test_dir + '/test_optimiser/'
        self.test_directories.append(test_directory)

        training_grid_settings = {'type': 'database',
                                  'library_path': test_directory,
                                  'library_name': 'training_library',
                                  'source_cases_path': self.route_test_dir + '/src/output/',
                                  'points': [[2, 10],
                                             [2, 80],
                                             # [3, 10],
                                             # [3, 80],
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

        optimisation_settings = {}

        input_settings = {'simulation_name': 'test_interpolation_suite',
                          'directory': test_directory,
                          'parameters': [
                              {'name': 'alpha',
                               'format_scaling': 100,
                               'sigfig': 1,
                               'optimisation_domain': [2, 5]},
                              {'name': 'u_inf',
                               'format_scaling': 10,
                               'sigfig': 0,
                               'optimisation_domain': [10, 80]}],
                          'case_name_format': 'pazy_uinf{:04g}_alpha{:04g}',
                          'case_name_parameters': ['u_inf', 'alpha'],  # in case order does not match parameter order
                          'simulation_settings': sharpy_simulation_settings,
                          'interpolation_settings': interpolation_simulation_settings,
                          'testing_data': testing_grid_settings,
                          'training_data': training_grid_settings,
                          'optimiser_settings': optimisation_settings,
                          }

        siminfo = interpolation_main.SimulationInfo(directory=input_settings['directory'],
                                                    name=input_settings['simulation_name'],
                                                    settings=input_settings)

        opti = optimiser.Optimiser(siminfo)
        opti.initialise_interpolation_suite()
        opti.optimise()

    def test_cost_function(self):
        test_directory = self.route_test_dir + '/test_cost_function/'

        self.test_directories.append(test_directory)

        training_grid_settings = {'type': 'database',
                                  'library_path': test_directory,
                                  'library_name': 'training_library',
                                  'source_cases_path': self.route_test_dir + '/src/output/',
                                  'points': [[2, 10],
                                             [2, 80],
                                             # [3, 10],
                                             # [3, 80],
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

        optimisation_settings = {}

        input_settings = {'simulation_name': 'test_interpolation_suite',
                          'directory': test_directory,
                          'parameters': [
                              {'name': 'alpha',
                               'format_scaling': 100,
                               'sigfig': 1,
                               'optimisation_domain': [2, 5]},
                              {'name': 'u_inf',
                               'format_scaling': 10,
                               'sigfig': 0,
                               'optimisation_domain': [10, 80]}],
                          'case_name_format': 'pazy_uinf{:04g}_alpha{:04g}',
                          'case_name_parameters': ['u_inf', 'alpha'],  # in case order does not match parameter order
                          'simulation_settings': sharpy_simulation_settings,
                          'interpolation_settings': interpolation_simulation_settings,
                          'testing_data': testing_grid_settings,
                          'training_data': training_grid_settings,
                          'optimiser_settings': optimisation_settings,
                          'cost_function_name': 'EigenvalueComparison',
                          'cost_function_settings': {'damping_penalty_name': 'GaussianPenalty',
                                                     'damping_penalty_settings': {'multiplying_factor': 1}}
                          }

        siminfo = interpolation_main.SimulationInfo(directory=input_settings['directory'],
                                                    name=input_settings['simulation_name'],
                                                    settings=input_settings)

        interpolation_suite = suite.Suite()
        interpolation_suite.create(siminfo)
        interpolation_suite.initialise_cost_function('EigenvalueComparison')

        point_info = {'alpha': 2, 'u_inf': 70}
        # interpolation_suite.interpolate_at_point({'alpha': 3, 'u_inf': 20},
        #                                          case_name='trial_interpolation',
        #                                          cases_folder=siminfo.directory + '/interpolation_cases/',
        #                                          output_folder=siminfo.directory + '/interpolation_output/')

        cost = interpolation_suite.evaluate(point_info)

        with self.subTest('Test the Evaluation module'):
            print('Testing Evaluation')
            print(f'test output dir {sharpy_simulation_settings["output_folder"]}')
            comparison = evaluation.Evaluation(
                interpolated_directory=siminfo.path + '/interpolation_output/' + '/evaluation/',
                testing_output_directory=sharpy_simulation_settings['output_folder'] + '/')
            comparison.initialise_cost_function('EigenvalueComparison', {'damping_penalty_name': 'GaussianPenalty'})

            comparison.cost_report(siminfo.path)

    @classmethod
    def tearDownClass(cls):
        for direc in cls.test_directories:
            shutil.rmtree(direc)


if __name__ == '__main__':
    unittest.main()
