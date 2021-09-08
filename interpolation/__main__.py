import argparse
import logging
import interpolation.suite as suite
import interpolation.interface as interface
import sharpy.io.logger_utils as logger_utils
import os


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


def main(settings=None):
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

    interpolation_suite = suite.Suite()
    interpolation_suite.create(siminfo)

    return 0


class SimulationInfo:

    def __init__(self, name, directory, settings):
        self.name = name  # str
        self.directory = directory  # str
        self.settings = settings  # dict

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.parameters = dict()
        self.parameter_index = dict()
        for ith, parameter_info in enumerate(self.settings['parameters']):
            self.parameters[ith] = parameter_info
            self.parameter_index[parameter_info['name']] = parameter_info.get('index', ith)

        self.n_parameters = len(self.parameters)

    @property
    def path(self):
        return os.path.abspath(self.directory + '/' + self.name + '/')

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

if __name__ == '__main__':
    main()
