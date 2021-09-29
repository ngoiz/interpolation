import argparse
import logging
import interpolation.suite as suite
import interpolation.interface as interface
import sharpy.io.logger_utils as logger_utils
import os
import warnings
import sys
import numpy as np
import interpolation.optimiser as optimiser
import interpolation.adaptivelhs as adaptivelhs
import yaml


def parse_inputs():
    parser = argparse.ArgumentParser(description=
                                     """Interpolation and optimisation of SHARPy reduced order models
                                     """)

    parser.add_argument('input_file', help='input file in YAML format')
    parser.add_argument("-r", "--refresh", action="store_true",
                        help="Refresh inputs (remove training_data.txt and testing_data.txt)")
    parser.add_argument('-o', '--optimize', action='store_true',
                        help='Run optimiser on interpolation')

    parser = parser.parse_args()

    return parser


def main(*args, settings=None):
    if settings is None:
        parser = parse_inputs()

        settings = interface.load_yaml(parser.input_file)  # dict

    siminfo = SimulationInfo(directory=settings['directory'],
                             name=settings['simulation_name'],
                             settings=settings)
    log_file_name = siminfo.path + f'/{siminfo.name}.log'

    logger_utils.load_logger_settings(log_name=log_file_name,
                                      file_level='info',
                                      console_level='info')

    starting_msg = '#############################################\n' +\
                   'Starting Interpolation and Optimisation Suite\n' + \
                   '#############################################'

    logging.info(f'Starting Suite - log file created at {log_file_name}')
    print(starting_msg)

    flow = siminfo.settings['flow']

    switcher = FlowSwitcher()
    program_exec = switcher.switch(flow)
    program_exec(siminfo)

    return 0


class FlowSwitcher:
    def switch(self, flow):
        method = self.__getattribute__(flow)
        return method

    @staticmethod
    def interpolation(siminfo):
        interpolation_suite = suite.Suite()
        interpolation_suite.create(siminfo)
        interpolation_suite.interpolate()

    @staticmethod
    def optimisation(siminfo):
        optimisation_suite = optimiser.Optimiser(siminfo)
        optimisation_suite.initialise_interpolation_suite()
        optimisation_suite.run()

    @staticmethod
    def adaptive(siminfo):
        adaptive_suite = adaptivelhs.AdaptiveLatinSampling(siminfo)
        adaptive_suite.initialise_interpolation_suite()
        adaptive_suite.run()


class SimulationInfo:

    def __init__(self, name, directory, settings):
        self.name = name  # str
        self.__directory = directory  # str
        self.settings = settings  # dict

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.parameters = dict()  # dict: id: {name: v_name, fmt_scaling: v_fmt_scaling}
        self.parameter_index = dict()  # dict: name: id
        for ith, parameter_info in enumerate(self.settings['parameters']):
            self.parameters[ith] = parameter_info
            self.parameter_index[parameter_info['name']] = parameter_info.get('index', ith)

        self.n_parameters = len(self.parameters)

        self.save_settings_file()

    @property
    def parameter_names(self):
        names = []
        for idx in range(self.n_parameters):
            names.append(self.parameters[idx]['name'])

        return names

    @property
    def path(self):
        return os.path.abspath(self.__directory + '/' + self.name + '/')

    def case_name_generator(self, parameter_values):
        """
        Generates a case name based on the setting ``case_name_format`` and its multiplying factor given in the
        parameter setting

        Args:
            parameter_values (list or np.array or dict): Parameter values that make up the case name

        Returns:
            str: formatted case name
        """
        fmt_string = self.settings['case_name_format']
        param_order = self.settings.get('case_name_parameters', [param_info['name']
                                                                 for param_info in self.parameters.values()])

        str_list = []
        for param_name in param_order:
            param_index = self.parameter_index[param_name]
            param_info = self.parameters[param_index]
            if type(parameter_values) is not dict:
                str_list.append(parameter_values[param_index] * param_info.get('format_scaling', 1))
            else:
                str_list.append(parameter_values[param_name] * param_info.get('format_scaling', 1))
        return fmt_string.format(*str_list)

    def run_sharpy(self, parameter_values):
        """
        Runs sharpy with the provided ``simulation_settings`` at the ``parameter_values`` specified

        Args:
            parameter_values (list or np.array or dict): Parameter information

        """
        if type(parameter_values) is dict:
            param_dict = parameter_values  # as required for run_sharpy
        else:
            param_dict = {p_info['name']: parameter_values[p_index] for p_index, p_info in self.parameters.items()}

        interface.run_sharpy(case_name=self.case_name_generator(parameter_values),
                             parameters=param_dict,
                             simulation_settings=self.settings['simulation_settings'])

    def parameter_sigfig(self, point_info):
        """

        Args:
            point_info (list or dict or np.array): point information

        Returns:
            dict: Containing parameter name and value appropriate to the specified significant figures specified in
                  the parameter setting ``sigfig``
        """

        out_dict = {}
        for param_idx, param_info in self.parameters.items():
            if type(point_info) is dict:
                point_val = point_info[param_info['name']]  #input is dict
            else:
                try:
                    r, c = point_info.shape
                except AttributeError:
                    point_val = point_info[param_idx]  # input is list
                except ValueError:
                    point_val = point_info[param_idx]  # input is a (1,) array where it is not able to unpack into r, c
                else:
                    point_val = point_info[0, param_idx] # input is 2D array

            try:
                sigfig = param_info['sigfig']
            except KeyError:
                out_dict[param_info['name']] = point_val
            else:
                out_dict[param_info['name']] = np.round(point_val, decimals=sigfig)
        return out_dict

    def save_settings_file(self):
        new_name = self.path + '/input_' + self.name + '.yaml'
        with open(new_name, 'w') as f:
            f.write(yaml.dump(self.settings))


def interpolation_run():
    """
    This is a wrapper function for the console command ``interpolation``
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main(sys.argv)


if __name__ == '__main__':
    main()
