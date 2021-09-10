import logging
import os
import GPyOpt
import interpolation.suite as suite


class Optimiser:

    def __init__(self, sim_info):
        """

        Args:
            sim_info (interpolation.__main__.SimInfo):

        Optimiser specific settings provided through ``optimiser_settings``:
            - model_type: Optimiser acquisition model (def GP_MCMC)
            - acquisition_type: optimisier acquisition type (def EI_MCMC)
        """

        self.settings = sim_info.settings['optimiser_settings']

        self.logger = logging.getLogger(__name__)

        self.optimisation_info = OptimisationInfo()
        self.optimisation_info.optimisation_bounds = sim_info
        self.sim_info = sim_info

        self.interpolation = None

    def initialise_interpolation_suite(self):
        self.interpolation = suite.Suite()
        self.interpolation.create(self.sim_info)
        self.interpolation.initialise_cost_function('EigenvalueComparison')

    def run(self):

        self.optimise()

    def optimise(self):
        self._new_optimisation()
        self.logger.info(f'Starting optimisation {self.optimisation_info.optimisation_id}')

        optimiser = GPyOpt.methods.bayesian_optimization.BayesianOptimization(
            f=self.interpolation.evaluate,
            domain=self.optimisation_info.optimisation_bounds,
            normalize_Y=False,
            exact_feval=True,
            model_type=self.settings.get('model_type', 'GP_MCMC'),
            acquisition_type='EI_MCMC',
            verbosity=True
        )

        opti_id = self.optimisation_info.optimisation_id
        opti_path = self.optimisation_info.optimisation_data[opti_id]['output_directory']
        optimiser.run_optimization(max_iter=10,
                                   verbosity=True,
                                   report_file=opti_path + '/optimisation_report.txt',
                                   evaluations_file=opti_path + '/evaluations_report.txt',
                                   models_file=opti_path + '/models_report.txt')

        import pdb; pdb.set_trace()

    def _new_optimisation(self):
        self.optimisation_info.optimisation_id += 1  # maybe add a next_optimisation method?
        output_directory = self.sim_info.directory + '/optimisation_{:02g}'.format(
            self.optimisation_info.optimisation_id)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            self.logger.info(f'Optimisation {self.optimisation_info.optimisation_id} output directory created '
                             f'and set: {output_directory}')
        self.optimisation_info.optimisation_data[self.optimisation_info.optimisation_id] = \
            {'output_directory': output_directory}


class OptimisationInfo:
    """Class to store information on the optimisation process
    """

    def __init__(self):
        self.optimisation_id = -1  # int: number of optimisations performed

        self.optimisation_data = {} # something on optimisation history
        # self.evaluations # history on evaluated points

        self._optimisation_bounds = []

    @property
    def optimisation_bounds(self):
        """

        Returns:
            list: containing optimisation variables bounds information
        """
        return self._optimisation_bounds

    @optimisation_bounds.setter
    def optimisation_bounds(self, sim_info):
        """
        Sets the bounds on the optimised variables following the inputs of GPyOpt

        Ensure the parameter settings include an ``optimisation_domain`` as list specifying the bounds on said variable
        By default the type of variable is set as ``'type' = 'continuous'``

        Args:
            sim_info (interpolation.__main__.SimInfo):
        """
        param_order = []
        bounds = []
        for param_index, param_info in sim_info.parameters.items():
            param_order.append(param_index)
            try:
                opti_var_info = {'name': param_info['name'],
                                 'type': param_info.get('type', 'continuous'),
                                 'domain': param_info['optimisation_domain']}
            except KeyError:
                raise KeyError('Ensure a setting ``optimisation_domain`` is provided as list for the variable '
                               f'{param_info["name"]}')
            else:
                bounds.append(opti_var_info)

        self._optimisation_bounds = [bounds[i_param] for i_param in param_order]




