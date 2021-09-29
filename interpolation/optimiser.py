import logging
import os
import GPyOpt
import interpolation.suite as suite
from interpolation.plotutils import plot_optimisation_data, plot_acquisition_functions
from interpolation.grid import sort_point_input
import pickle
import numpy as np


class Optimiser:

    def __init__(self, sim_info):
        """

        Args:
            sim_info (interpolation.__main__.SimulationInfo):

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

        iter_max = self.settings.get('max_iter', 1)

        while self.optimisation_info.optimisation_id < iter_max:
            opti_point = self.optimise()

            self.add_to_training(opti_point)

        import pdb; pdb.set_trace()

    def optimise(self):
        self._new_optimisation()
        self.logger.info(f'Starting optimisation {self.optimisation_info.optimisation_id}')

        prev_x, prev_y = self.generate_previous_xy()

        self.logger.info(f'Previous X {prev_x}')
        self.logger.info(f'Previous Y {prev_y}')

        opti = GPyOpt.methods.bayesian_optimization.BayesianOptimization(
            f=self.interpolation.evaluate,
            domain=self.optimisation_info.optimisation_bounds,
            initial_design_numdata=self.settings.get('initial_design_numdata', 1),
            initial_design_type=self.settings.get('initial_design_type', 'random'),
            normalize_Y=False,
            exact_feval=True,
            model_type=self.settings.get('model_type', 'GP_MCMC'),
            acquisition_type=self.settings.get('acquisition_type', 'EI_MCMC'),
            verbosity=True,
            maximize=False,
            X=prev_x,
            Y=prev_y,
            model_update_interval=self.settings.get('model_update_interval', 1)
        )

        opti_id = self.optimisation_info.optimisation_id
        opti_path = self.optimisation_info.optimisation_data[opti_id]['output_directory']
        np.savetxt(opti_path + '/priors_evaluations.txt', np.column_stack((prev_x, prev_y)))
        opti.run_optimization(max_iter=self.settings.get('max_opti_iterations', 10),
                              verbosity=True,
                              report_file=opti_path + '/optimisation_report.txt',
                              evaluations_file=opti_path + '/evaluations_report.txt',
                              models_file=opti_path + '/models_report.txt')

        with open(opti_path + '/optimiser.pkl', 'wb') as f:
            try:
                pickle.dump(opti, f, protocol=pickle.HIGHEST_PROTOCOL)
            except AttributeError as err:
                self.logger.warning(f'Unable to save optimisation pickle: {err}')

        self.interpolation.training_data(
            self.optimisation_info.optimisation_data[self.optimisation_info.optimisation_id]['output_directory']
            + '/training_data.yaml')

        self.optimisation_info.finish_optimisation(opti)

        plot_optimisation_data(self.optimisation_info, self.optimisation_info.optimisation_id)
        # plot_acquisition_functions(self.optimisation_info, self.optimisation_info.optimisation_id, opti)

        self.logger.info('Optimisation Complete')
        self.logger.info(f'Worst case interpolation at {opti.x_opt} with cost {opti.fx_opt}')

        return opti.x_opt

    def _new_optimisation(self):
        self.optimisation_info.optimisation_id += 1  # maybe add a next_optimisation method?
        output_directory = self.sim_info.path + '/optimisation_{:02g}'.format(
            self.optimisation_info.optimisation_id)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            self.logger.info(f'Optimisation {self.optimisation_info.optimisation_id} output directory created '
                             f'and set: {output_directory}')
        self.optimisation_info.optimisation_data[self.optimisation_info.optimisation_id] = \
            {'output_directory': output_directory}

    def add_to_training(self, new_point):
        self.interpolation.training_data.add_point(new_point, self.sim_info)

    def generate_previous_xy(self):
        x_training = self.interpolation.training_data()
        y_training = np.zeros((x_training.shape[0], 1))
        x_training = np.column_stack((x_training[:, 1], x_training[:, 0]))

        prev_x = []
        prev_y_temp = []
        for prev_opti in range(self.optimisation_info.optimisation_id):
            evals = self.optimisation_info.optimisation_data[prev_opti]['evaluations']
            for point in evals:
                point = self.sim_info.parameter_sigfig(point)
                new_entry = [point[param_name] for param_name in self.sim_info.parameter_names]
                if new_entry not in prev_x:
                    prev_x.append(new_entry)
                # cost = self.interpolation.evaluate(point)
                # prev_y_temp.append(cost)

        if len(prev_x) != 0:
            cost = self.interpolation.evaluate(prev_x)
            prev_y_temp = cost
            prev_x = np.array(prev_x)
            prev_y = np.zeros((prev_x.shape[0], 1))
            for i, entry in enumerate(prev_y_temp):
                prev_y[i] = entry

            x = np.concatenate((x_training, prev_x))
            y = np.concatenate((y_training, prev_y))

            return x, y
        else:
            return x_training, y_training


class OptimisationInfo:
    """Class to store information on the optimisation process
    """

    def __init__(self):
        self.optimisation_id = -1  # int: number of optimisations performed

        self.optimisation_data = {}  # something on optimisation history
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
            sim_info (interpolation.__main__.SimulationInfo):
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

    def finish_optimisation(self, optimiser):
        """

        Args:
            optimiser (GPyOpt.methods.bayesian_optimisation.BayesianOptimisation):

        Returns:

        """

        self.optimisation_data[self.optimisation_id]['opti'] = optimiser

        self.optimisation_data[self.optimisation_id] = {**self.optimisation_data[self.optimisation_id],
                                                        **{'x': optimiser.X, 'y': optimiser.Y,
                                                           'fx_opt': optimiser.fx_opt,
                                                           'x_opt': optimiser.x_opt,
                                                           'evaluations': self.generate_evals_dict_from_opt(optimiser)}}

    def generate_evals_dict_from_opt(self, opt):
        """

        Args:
            opt (GPyOpt.methods.bayesian_optimisation.BayesianOptimisation):

        Returns:
            list: List of dictionaries containing evaluated points
        """
        out = []
        for ith in range(opt.Y.shape[0]):
            if opt.Y[ith] != 0.:
                out.append({param_info['name']: opt.X[ith, param_index]
                            for param_index, param_info in enumerate(self.optimisation_bounds)})

        return out
