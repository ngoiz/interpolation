import os

class Suite:
    """
    Interpolation Suite
    """
    def __init__(self):

        self.simulation_name = None  # str: simulation name
        self.directory = None  # str: simulation directory

        self.parameters = dict()  # dict: idx: param_name
        self.parameter_index = dict()  # dict: param_name: idx

    @property
    def path(self):
        return os.path.abspath(self.directory + self.simulation_name)

    def create(self, input_data):
        """
        Creates an interpolation suite.

        1. Create folder structure

        Args:
            input_data (dict): With keys:
                simulation_name
                directory
                parameters: list of parameter names

        Returns:

        """
        self.simulation_name = input_data['simulation_name']
        self.directory = input_data['directory']

        self._create_directories()

        for ith, parameter_name in input_data['parameters']:
            self.parameters[ith] = parameter_name
            self.parameter_index[parameter_name] = ith

    def _create_directories(self):
        directory_names = ['training_cases', 'training_output', 'testing_output', 'testing_cases']
        for dir in directory_names:
            if not os.path.isdir(self.path + '/' + dir):
                os.makedirs(self.path + '/' + dir)

