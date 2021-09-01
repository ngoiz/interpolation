import numpy as np
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary
import os


class Grid:

    def __init__(self, settings):
        self.library = None  # pmor.library
        self.settings = settings
        self.source_case_directory = None  # str: directory from where to source other cases

    def add_point(self, point_parameters):
        # to-do; add point to grid, if already exists do not add
        pass


class Database(Grid):

    def __init__(self, settings):
        super().__init__(settings)

    def load(self):
        """
        Creates or loads a pmor library pkl from the settings
            library_path and library_name

        """
        # load from existing pkl
        # create pkl
        # source cases from a specific directory - in case new cases are needed

        library_pickle_folder = self.settings['library_path']
        library_pickle_name = self.settings['library_name']
        if library_pickle_name.split('.')[-1] == 'pkl':
            library_pickle_name = ''.join(library_pickle_name.split('.')[:-1])
        library_settings = {'folder': library_pickle_folder,
                            'library_name': library_pickle_name}

        self.library = pmorlibrary.ROMLibrary()

        if os.path.exists(library_settings['folder'] + '/' + library_settings['library_name'] + '.pkl'):
            self.library.load_library(library_settings['folder'] + '/' + library_settings['library_name'] + '.pkl')
            self.library.folder = library_settings['folder']
            self.library.library_name = library_settings['library_name']
        else:
            self.library.create(library_settings)

        # add extra points if needed - from points setting
