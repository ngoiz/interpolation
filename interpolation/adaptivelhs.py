import logging
import os
import interpolation.suite as suite
import numpy as np
import pyDOE2
import interpolation.grid as grid
import matplotlib.pyplot as plt
import matplotlib as mpl


class AdaptiveLatinSampling:

    def __init__(self, sim_info):
        """

        Args:
            sim_info (interpolation.__main__.SimulationInfo):
        """

        self.settings = sim_info.settings['adaptive_lhs_settings']
        self.logger = logging.getLogger(__name__)
        self.sim_info = sim_info

        self.interpolation = None

        self.lhs_data = {}  # dict containing info at every iteration
        self.lhs_iter = -1  # iteration counter

    def initialise_interpolation_suite(self):
        self.interpolation = suite.Suite()
        self.interpolation.create(self.sim_info)
        self.interpolation.initialise_cost_function('EigenvalueComparison')

    def run(self):

        iter_max = self.settings.get('max_iter', 1)
        while self.lhs_iter < iter_max:
            x_opt = self.new_iteration()
            self.end_iteration()

            self.add_to_training(x_opt)

    def add_to_training(self, new_point):
        self.interpolation.training_data.add_point(new_point, self.sim_info)

    def _new_iteration_admin(self):
        self.lhs_iter += 1
        self.logger.info(f'Starting iteration {self.lhs_iter}')

        output_directory = self.sim_info.path + '/iteration_{:02g}/'.format(self.lhs_iter)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            self.logger.info(f'Iteration {self.lhs_iter} set output directory {output_directory}')

        self.lhs_data[self.lhs_iter] = {'output_directory': output_directory}

    def new_iteration(self):
        self._new_iteration_admin()
        testing_points = self.set_up_testing_grid()

        if self.settings['with_previous_evals']:
            try:
                testing_points = np.concatenate((self.lhs_data[self.lhs_iter - 1]['evaluations'],
                                                 testing_points))
            except KeyError:
                pass
        cost_at_testing = self.interpolation.evaluate(testing_points)

        # find min cost (i.e. worst match)
        min_cost_idx = np.argmin(cost_at_testing)
        min_cost = cost_at_testing[min_cost_idx]
        min_cost_params = testing_points[min_cost_idx, :]

        min_cost_dict = {}
        for param_index, param_info in self.sim_info.parameters.items():
            min_cost_dict[param_info['name']] = min_cost_params[param_index]

        self.logger.info(f'Worst match at {min_cost_dict} with value: {min_cost}')

        self.lhs_data[self.lhs_iter]['evaluations'] = testing_points
        self.lhs_data[self.lhs_iter]['cost'] = cost_at_testing
        self.lhs_data[self.lhs_iter]['x_opt'] = min_cost_params
        self.lhs_data[self.lhs_iter]['cost_opt'] = min_cost

        return min_cost_dict

    def end_iteration(self):
        output_dir = self.lhs_data[self.lhs_iter]['output_directory']
        evals = self.lhs_data[self.lhs_iter]['evaluations']
        cost = self.lhs_data[self.lhs_iter]['cost']
        import pdb; pdb.set_trace()
        np.savetxt(output_dir + '/evals.txt', np.column_stack((cost, evals)))

        plt.figure()
        abs_cost = np.abs(cost)
        cmap = plt.get_cmap('Reds')
        nrm = plt.Normalize(vmin=0, vmax=np.ceil(np.max(abs_cost)))
        plt.scatter(evals[:, 0], evals[:, 1], facecolor=cmap(nrm(abs_cost)), edgecolor='k', lw=0.2)
        training_points = self.interpolation.training_data()
        plt.scatter(training_points[:, 1], training_points[:, 0], marker='x', color='k')
        plt.colorbar(mpl.cm.ScalarMappable(nrm, cmap), label='Cost Function')

        plt.savefig(output_dir + '/iter_path.pdf')

        self.interpolation.training_data(output_dir + '/training_data.yaml')

    def set_up_testing_grid(self):
        raw_lhs = pyDOE2.lhs(self.sim_info.n_parameters,
                             samples=self.settings.get('initial_test_samples', 4),
                             criterion=self.settings.get('lhs_criterion', 'centermaximin'))

        # scale lhs
        lhs = np.zeros_like(raw_lhs)
        for param_index, param_info in self.sim_info.parameters.items():
            vmin = param_info['optimisation_domain'][0]
            scale = param_info['optimisation_domain'][1] - vmin
            lhs[:, param_index] = raw_lhs[:, param_index] * scale + vmin

        return lhs


