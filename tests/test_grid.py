import unittest
import os
import interpolation.__main__ as interpolation_main
import interpolation.grid as grid
import shutil
import interpolation.interface as interface


class TestGrid(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    test_directories = []

    def test_grid_loading(self):
        test_directory = self.route_test_dir + '/test_grid/'
        self.test_directories.append(test_directory)
        source_cases_path = self.route_test_dir + '/src/output'

        # SHARPy settings to run sharpy quickly when no case is present
        temp_cases = test_directory + '/cases/'
        temp_output = test_directory + '/output/'
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

        grid_settings = {'library_path': test_directory,
                         'library_name': 'test_library',
                         }

        input_settings = {'simulation_name': 'test_grid',
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

        gr = grid.Database(grid_settings)

        with self.subTest('Create library'):
            gr.load()  # the library does not exist so it has to be created
            assert os.path.exists(grid_settings['library_path'] + '/' +
                                  grid_settings['library_name'] + '.pkl'), 'Pickle file was not generated correctly'

        with self.subTest('Loading library'):
            """Loads the recently created library into another Database instance"""
            gr2 = grid.Database(grid_settings)
            gr2.load()

        with self.subTest('Add existing point via list'):
            point = [2, 10]
            gr2.add_point(point, sim_info, source_cases_path)
            members = gr2.library.entries()

            case_info = {'alpha': point[0], 'u_inf': point[1]}
            if case_info not in members:
                raise IndexError(f'Case was not properly added to list. Case {case_info} not in {members}')

        with self.subTest('Add non existent point via dict'):
            case_info = {'alpha': 3, 'u_inf': 20}
            gr2.add_point(case_info, sim_info, source_cases_path, sharpy_simulation_settings['output_folder'])
            members = gr2.library.entries()
            if case_info not in members:
                raise IndexError(f'Case was not properly added to list. Case {case_info} not in {members}')

        with self.subTest('Loading library with desired points from yaml'):
            points_input = [{'alpha': 5, 'u_inf': 80},
                            {'alpha': 3, 'u_inf': 10}]

            yaml_file = test_directory + '/input_database_points.yaml'

            interface.dict_list_to_yaml(points_input, yaml_file)
            grid_settings['points'] = yaml_file

            gr3 = grid.Database(grid_settings)
            gr3.load(sim_info, source_cases_path, sharpy_simulation_settings['output_folder'])
            members = gr3.library.entries()
            for case_info in points_input:
                if case_info not in members:
                    raise IndexError(f'Case was not properly added to list. Case {case_info} not in {members}')

    @classmethod
    def tearDownClass(cls):
        for direc in cls.test_directories:
            shutil.rmtree(direc)


if __name__ == '__main__':
    unittest.main()
