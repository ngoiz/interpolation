import os
import logging
import sharpytools.batch.sets as sets
import sharpytools.batch.interpolation as i_sets
import interpolation.costfunctions as costfunctions
import numpy as np


class Evaluation:

    def __init__(self, interpolated_directory, testing_output_directory, testing_library=None, **kwargs):

        self.logger = logging.getLogger(__name__)
        self.verbose = kwargs.get('verbose', False)
        self.cost_function = None

        self.testing_data = self.load_testing_data(testing_output_directory, testing_library)
        self.interpolated_data = self.load_interpolated_data(interpolated_directory)

    def load_testing_data(self, test_directory, testing_library=None):

        testing_data = sets.Actual(test_directory + '/*')
        testing_data.systems = ['aeroelastic']

        if testing_library is not None:
            rom_library = testing_library.library
        else:
            rom_library = None

        testing_data.load_bulk_cases('bode', 'eigs', eigs_legacy=False, rom_library=rom_library, verbose=self.verbose)

        return testing_data

    def load_interpolated_data(self, interpolated_directory):

        interpolated_data = i_sets.Interpolated(interpolated_directory, '/*')
        interpolated_data.systems = ['aeroelastic']
        interpolated_data.load_bulk_cases('bode', 'eigs', verbose=self.verbose)

        return interpolated_data

    def find_cases(self, param_info):

        t_case = self.testing_data.aeroelastic.find_param(param_info)
        i_case = self.interpolated_data.aeroelastic.find_param(t_case.case_info)
        if i_case is None:
            self.logger.warning(f'No case found in interpolation database! Case was {t_case.case_info}')
            return None

        return t_case, i_case

    def initialise_cost_function(self, cost_function_name=None, cost_function_settings=None):

        self.cost_function = costfunctions.get_cost_function(cost_function_name)
        self.cost_function.initialise(settings=cost_function_settings)

    def cost_report(self, output_directory=None, column_names=None):
        output = []
        if column_names is None:
            column_names = list(self.testing_data.aeroelastic(0).case_info.keys())
        else:
            for name in column_names:
                if name not in self.testing_data.aeroelastic(0).case_info.keys():
                    raise KeyError(f'input column_names {name} not found in case parameters')
        params = []
        for n_case, t_case in enumerate(self.testing_data.aeroelastic):
            i_case = self.interpolated_data.aeroelastic.find_param(t_case.case_info)
            if i_case is None:
                continue
            output.append(self.cost_function(t_case, i_case))
            params.append(np.array([t_case.case_info[name] for name in column_names]))

        out = np.column_stack((params, output))
        if output_directory is not None:
            np.savetxt(output_directory + '/cost_report.txt', out,
                       header=str.join('\t', column_names + ['cost']))

        return out
