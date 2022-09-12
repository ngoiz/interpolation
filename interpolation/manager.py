"""
Optimisation and interpolation simulation manager

Write to a csv, assign log number to simulation run, save settings
"""
import os
import pandas as pd


class Manager:

    def __init__(self, simulation_settings, root_directory, name=None):
        self.directory = root_directory
        self.simulation_settings = simulation_settings  # dict: with all sim settings

        if name is not None:
            try:
                self.name = name.split('.')[0]  # no extension
            except IndexError:
                self.name = name
        else:
            self.name = 'manager'

        self.data = None

        self._create_or_load()

    @property
    def path(self):
        return self.directory + '/' + self.name + '.csv'

    def _create_or_load(self):

        # open(self.path, 'a')
        if os.path.exists(self.path):
            df = pd.read_csv(self.path)
        else:
            df = pd.DataFrame(data=self.simulation_settings)

        self.data = df

    # def get_next_simulation_id(self):
