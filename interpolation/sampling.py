from scipy.spatial import Delaunay
import itertools
import numpy as np
import interpolation.grid as grid
import interpolation.suite as suite
import logging
import GPyOpt.models as gmodels
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import interpolation.interface as interface
import h5py
import random
# ---- Sampling utils


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def set_chooser(known_parameter_points, edges=None, max_n_sets=None):
    """
    Find sets Z given the known parameter points, X
    Produce sets X in convex hull of Z
    Args:
        known_parameter_points:

    Returns:
        list(AssociatedSet): lists of sets containing X in convex hull of Z and Z
    """

    known_parameter_points = {tuple(i) for i in known_parameter_points}
    edges = {tuple(i) for i in edges}  # Z
    known_parameter_points = known_parameter_points.union(edges)  # X

    exclusive_testing_points = known_parameter_points.difference(edges)  # M

    combinations = generate_combinations(exclusive_testing_points)

    list_of_N_sets = []
    for elem_set in combinations:
        # elem_set is what we refer to as Z
        if edges is not None:
            elem_set = elem_set.union(edges)  # R u Z

        points_in_set = known_parameter_points.difference(elem_set)  # S

        if np.abs(len(points_in_set) - len(elem_set)) > 3:  # change to 3
        #     if not len(points_in_set) >= 8:  # new attempt
            continue

        if len(points_in_set) != 0:
            z_set = ParameterPointSet(elem_set)
            # create set. This is X in S_Z, and we now need to associate a Z
            new_set = AssociatedSet(points_in_set, z_set)
            list_of_N_sets.append(new_set)
            if max_n_sets is not None:
                if len(list_of_N_sets) > max_n_sets:
                    break

    if len(list_of_N_sets) == 0:
        logging.warning('Unable to generate any more sets!')
        list_of_N_sets.append(AssociatedSet(exclusive_testing_points, edges))

    return list_of_N_sets


def generate_combinations(known_parameter_points):
    combis = []  # list of all possible sets

    n_points = len(known_parameter_points)
    min_points_in_set = 0  # verify because this is not correct

    # for points_in_set in range(min_points_in_set, n_points):
    #     combis.append(list(itertools.combinations(known_parameter_points, points_in_set)))
    for points_in_set in range(min_points_in_set, n_points):
        combis += [set(i) for i in itertools.combinations(known_parameter_points, points_in_set)]

    # output = []
    return combis
    # return [set(i) for i in grid.flatten(combis)[::-1]]


class ParameterPointSet:

    n_instances = 0

    def __init__(self, points):

        ParameterPointSet.n_instances += 1
        self.id = ParameterPointSet.n_instances

        self.points = points  # set

    @property
    def n_points(self):
        return len(self.points)


class AssociatedSet(ParameterPointSet):
    def __init__(self, points, associated_set):
        super().__init__(points)

        self.z_set = associated_set  # ParameterPointSet containing training data

        self.interpolation = None

        self.gp_model = None  # gmodels.gpmodel.GPModel store own GP model

        self.cost = None

        self.scaling_factor = 1.  # float to normalize cost and GP function

        self.plot_data = dict()  # dict containing mean and variance of each

        self.parameters = None  # dict containing the sampling parameter info from sim_info

        # scaling aspect ratio for gp_model
        # need to see how to get it in here from the optimisation settings
        self._gaussian_aspect_ratio = 1

    @property
    def gaussian_aspect_ratio(self):
        """Two dimensional aspect ratio defined as param1/param0"""
        return self._gaussian_aspect_ratio

    @gaussian_aspect_ratio.setter
    def gaussian_aspect_ratio(self, aspect_ratio):
        self._gaussian_aspect_ratio = aspect_ratio
        logging.info(f'Set the Gaussian aspect ratio = {self.gaussian_aspect_ratio}')

    def initialise_interpolation(self, sim_info):
        # clean up
        files_to_remove = [sim_info.path + '/training_library.pkl',
                           sim_info.path + '/interpolation_output/']
        for item in files_to_remove:
            if os.path.isdir(item):
                shutil.rmtree(item)
            elif os.path.isfile(item):
                os.remove(item)

        self.interpolation = suite.Suite()

        training_data_settings = sim_info.settings['training_data'].copy()
        training_data_settings['points'] = list(self.z_set.points)
        print('Current training points', training_data_settings['points'])
        self.interpolation.create(sim_info,
                                  training_data_settings=training_data_settings,
                                  save_pickle=True)

        self.parameters = sim_info.settings['parameters']

    def train_gaussian(self):
        """
        Interpolates at valid points in set and trains GP model
        Returns:

        """
        cost = self.interpolation.evaluate(list(self.points))
        try:
            self.scaling_factor = np.max(np.abs(cost)) * np.sign(cost[np.argmax(np.abs(cost))])
        except IndexError:
            self.scaling_factor = cost  # assuming it is a single number...
            assert type(cost) is float or type(cost) is np.float64, f'Cost is actually different to a 1 number {type(cost)}'

        self.cost = cost

        self.gp_model = gmodels.gpmodel.GPModel(optimize_restarts=50, verbose=True, exact_feval=True, noise_var=0)

        c_shape = cost.shape
        if len(c_shape) != 2:
            if type(cost) is not np.ndarray:
                cost = np.array([cost, ])
            cost = cost.reshape(cost.shape[0], 1)

        # prepare for GP model - including cost normalisation
        X_all = np.array([entry for entry in self.points])
        if len(X_all.shape) != 2:
            X_all = X_all.reshape(X_all.shape[0], 1)
        Y_all = cost

        X_all = np.concatenate((X_all, np.array([i for i in self.z_set.points])))
        Y_all = np.concatenate((Y_all, np.zeros((self.z_set.n_points, 1))))

        # Scale X into a unit domain
        print('Absolute X_all', X_all)
        X_all = self.scale_x(X_all)
        print('Scaled X_all', X_all)
        self.gp_model.updateModel(X_all, Y_all / self.scaling_factor, None, None)

        return cost

    def scale_x(self, x):
        param_star = [1, self.gaussian_aspect_ratio]
        for i_param in range(x.shape[1]):
            vmin, vmax = self.parameters[i_param]['optimisation_domain']
            x[:, i_param] = (x[:, i_param] - vmin) / (vmax - vmin) * param_star[i_param]
        return x

    def report(self):
        out_dict = {'n_training_points': self.z_set.n_points,
                    'n_points_in_hull': self.n_points,
                    'set_id': self.id,
                    'training_points': str(self.z_set.points),
                    'points_in_hull': str(self.points)}
        if self.cost is not None:
            out_dict['cost'] = str(self.cost)

        return out_dict

    def expected_error(self, x):
        """

        Args:
            x (np.array): ``(n_point x n_param)`` array in absolute value

        Returns:

        """
        x_scaled = self.scale_x(x)
        posterior_mean, variance = self.gp_model.predict(x_scaled)
        return posterior_mean * self.scaling_factor, variance * np.abs(self.scaling_factor)

    def initial_error(self, out_path=None):
        # x1_dom = np.linspace(2, 5, 50)
        # x2_dom = np.linspace(10, 70, 50)
        x1_dom = np.linspace(*self.parameters[0]['optimisation_domain'], 70)
        x2_dom = np.linspace(*self.parameters[1]['optimisation_domain'], 70)

        X1_dom, X2_dom = np.meshgrid(x1_dom, x2_dom)

        Z = np.array([i for i in self.z_set.points])
        X = np.array([i for i in self.points])

        Y_dom = self.expected_error(np.column_stack((X1_dom.reshape(-1), X2_dom.reshape(-1))))
        m = Y_dom[0].reshape(X1_dom.shape)
        v = Y_dom[1].reshape(X1_dom.shape)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        nrm = plt.Normalize(vmin=np.min(m), vmax=np.max(m))
        sm = plt.cm.ScalarMappable(nrm, cmap=cmap)
        ax[0].contourf(X1_dom, X2_dom, m)

        ax[0].scatter(Z[:, 0], Z[:, 1])
        ax[0].scatter(X[:, 0], X[:, 1], marker='o', facecolor=cmap(nrm(self.cost)),
                      edgecolor='k', lw=0.2)
        ax[0].set_title('Posterior Mean')
        plt.colorbar(sm, ax=ax[0])

        ax[1].contourf(X1_dom, X2_dom, v)
        nrm = plt.Normalize(vmin=np.min(v), vmax=np.max(v))
        sm = plt.cm.ScalarMappable(nrm)
        ax[1].scatter(Z[:, 0], Z[:, 1])
        ax[1].scatter(X[:, 0], X[:, 1], marker='x', color='r')
        plt.colorbar(sm, ax=ax[1])

        ax[1].set_title('Standard Deviation')

        if out_path is not None:
            plt.savefig(out_path + '/error_{:02g}.pdf'.format(self.id))

        self.plot_data = {'X1_dom': X1_dom,
                          'X2_dom': X2_dom,
                          'm': m,
                          'v': v}
        return X1_dom, X2_dom, m, v

    def add_to_h5file(self, hdf5handle):
        """

        Args:
            hdf5handle (h5py.File()):

        Returns:

        """

        dict_of_names = {'mean': 'm',
                         'variance': 'v'}
        for k, v in dict_of_names.items():
            hdf5handle.create_dataset('{:s}_{:03g}'.format(k, self.id), data=self.plot_data[v])

        # add points
        # hdf5handle.create_dataset('training_points_{:03g}'.format(self.id), data=self.z_set.points)
        # hdf5handle.create_dataset('testing_points_{:03g}'.format(self.id), data=self.points)


class BayesianSampling:
    def __init__(self, sim_info):
        self.settings = sim_info.settings['sampling_settings']
        self.logger = logging.getLogger(__name__)
        self.sim_info = sim_info

        self.sets = None  # list(AssociatedSet)

        self.plot_path = self.sim_info.path + '/plots/'
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)

        self.sampling_info = []  # list(SamplingInfo) containing all the information at the different iterations

        # method to restart
        try:
            training_points_yaml = self.settings['training_points_yaml']
            known_points_yaml = self.settings['known_points_yaml']
        except KeyError:
            # might need to add edges as a boolean...
            # self.edges = np.array([[0, 30],
            #                        [0, 90],
            #                        [0, 50],
            #                        [3.5, 50],
            #                        [5, 50],
            #                        [3.5, 30],
            #                        [3.5, 90],
            #                        [5, 30],
            #                        [5, 90]])
            # bounds:
            param0 = self.sim_info.parameters[0]['optimisation_domain']
            param1 = self.sim_info.parameters[1]['optimisation_domain']
            self.edges = np.array([
                                   [param0[0], param1[0]],
                                   [param0[0], param1[1]],
                                   [param0[1], param1[0]],
                                   [param0[1], param1[1]]
            ])
            self.known_parameter_points = self.sim_info.settings['training_data']['points']
        else:
            self.edges = interface.yaml_to_array(training_points_yaml)  # Z
            self.logger.info('Successfully loaded training points set')
            self.known_parameter_points = interface.yaml_to_array(known_points_yaml)  # X
            self.logger.info('Successfully loaded known points set')

        self.iter_max = None

    def initialise(self):

        self.iter_max = self.settings['iter_max']

    def run(self):
        while len(self.sampling_info) < self.iter_max:
            current_iteration = self.new_iteration(self.settings.get('max_n_sets', None))
            # do as above, then finish it

            x_opt, x_std = current_iteration.evaluate()

            filtered_x_opt = np.array([[val for val in self.sim_info.parameter_sigfig(x_opt).values()]])
            filtered_x_std = np.array([[val for val in self.sim_info.parameter_sigfig(x_std).values()]])

            # add max std deviation
            self.known_parameter_points = np.concatenate((self.known_parameter_points, filtered_x_std))

            # add new fixed param
            self.edges = np.concatenate((self.edges, filtered_x_opt))

            self.logger.info(f'New known parameter points {filtered_x_opt}')

            # save training data set
            interface.array_to_yaml(['alpha', 'u_inf'], np.array(self.known_parameter_points),
                                    current_iteration.path + '/known_datapoints.yaml')
            interface.array_to_yaml(['alpha', 'u_inf'], np.array(self.edges),
                                    current_iteration.path + '/fixed_interpolation_datapoints.yaml')

        print('This is the end')

    def new_iteration(self, max_n_sets):
        self.logger.info(f'Initialising a new sampling iteration - Current iteration {len(self.sampling_info)}')
        iteration_sets = set_chooser(known_parameter_points=self.known_parameter_points, edges=self.edges,
                                     max_n_sets=max_n_sets)
        for a_set in iteration_sets:
            a_set.gaussian_aspect_ratio = self.settings.get('gaussian_aspect_ratio', 1)
        self.report(iteration_sets)
        new_sample = SamplingInfo(self.sim_info, iteration_sets)

        new_sample.train_sets(self.sim_info)

        self.sampling_info.append(new_sample)

        return new_sample

    def report(self, list_of_sets):

        full_info = [entry.report() for entry in list_of_sets]
        df = pd.DataFrame(full_info)

        df.to_pickle(self.sim_info.path + '/sampling_report.pkl')


class SamplingInfo:
    iteration_id = -1

    logger = logging.getLogger(__name__)

    def __init__(self, sim_info, sets):
        SamplingInfo.iteration_id += 1
        self.x_opt = None
        self.cost_opt = None
        self.sets = sets

        self.path = sim_info.path + '/sampling_iteration{:02g}/'.format(SamplingInfo.iteration_id)
        self.plot_path = self.path + '/plots/'
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            os.makedirs(self.plot_path)

    def train_sets(self, sim_info):
        self.logger.info(f'Training sets for iteration {SamplingInfo.iteration_id}')
        for ith_entry, entry in enumerate(self.sets):
            self.logger.info(f'Processing {ith_entry} of {len(self.sets)}')
            entry.initialise_interpolation(sim_info)
            entry.train_gaussian()

        self.report()

    def report(self):

        full_info = [entry.report() for entry in self.sets]
        df = pd.DataFrame(full_info)

        df.to_pickle(self.path + '/sampling_report.pkl')

    def evaluate(self):
        n_valid_sets = 0
        valid_m = []
        self.logger.info('Evaluating performance of the statistical surrogates...')
        hdffile = h5py.File(self.path + '/set_data.h5', 'w')
        for ith_entry, entry in enumerate(self.sets):
            self.logger.info(f'Processing {ith_entry} of {len(self.sets)}')
            n_valid_sets += 1
            X1_dom, X2_dom, m_single, v_single = entry.initial_error(self.plot_path)
            valid_m.append(m_single)
            try:
                m_all += m_single
                v_all += v_single
            except NameError:
                m_all = m_single
                v_all = v_single
                hdffile.create_dataset('X1_dom', data=X1_dom)
                hdffile.create_dataset('X2_dom', data=X2_dom)
            entry.add_to_h5file(hdffile)
        try:
            # m_all /= n_valid_sets
            # v_all /= n_valid_sets
            m_all
        except NameError:
            import pdb; pdb.set_trace()

        hdffile.create_dataset('mean_all', data=m_all)
        hdffile.create_dataset('var_all', data=v_all)
        hdffile.create_dataset('n_sets', data=n_valid_sets, dtype=int)

        std_deviation = np.zeros_like(m_all)
        for m_single in valid_m:
            std_deviation += 1/n_valid_sets * np.sqrt((m_single - m_all) ** 2)

        plot_mean_variance(X1_dom, X2_dom, std_deviation, std_deviation,
                           filename=self.plot_path + '/total_deviation.pdf',
                           ax0_title='Standard Deviation',
                           ax1_title='Standard Deviation')

        # could be m_all + std_deviation with some weights
        # interpolation_evaluation_function = np.abs(m_all) / np.max(np.abs(m_all)) * 0.5 + \
        #                                     0.5 * std_deviation / np.max(std_deviation)
        # interpolation_evaluation_function = -1 * m_all * np.sqrt(v_all)
        interpolation_evaluation_function = np.abs(m_all) / np.max(np.abs(m_all)) * 0.7 + v_all / np.max(v_all) * 0.3
        interpolation_evaluation_function *= np.max(np.abs(m_all))

        # deviation_evaluation_function = v_all
        deviation_evaluation_function = np.abs(m_all) / np.max(np.abs(m_all)) * 0.2 + v_all / np.max(v_all) * 0.8
        deviation_evaluation_function *= np.max(np.abs(v_all))

        # find worst case:
        idx_x_opt = np.argmax(interpolation_evaluation_function)
        x_opt = np.array([dom_variable.reshape(-1)[idx_x_opt] for dom_variable in [X1_dom, X2_dom]])
        cost_opt = np.max(interpolation_evaluation_function)

        plot_mean_variance(X1_dom, X2_dom, interpolation_evaluation_function, deviation_evaluation_function,
                           X=np.array([x_opt]),
                           filename=self.plot_path + '/interpolation_function.pdf',
                           ax0_title='Interpolation Func', ax1_title='Interpolation Function')

        self.x_opt = x_opt
        self.cost_opt = cost_opt

        self.logger.info(f'Worst case found at {self.x_opt} with cost {self.cost_opt}')

        # max std deviation place
        # idx_max_std = np.argmax(v_all)
        # x_max_std = np.array([dom_variable.reshape(-1)[idx_max_std] for dom_variable in [X1_dom, X2_dom]])
        # max_std = np.max(v_all)

        idx_max_std = np.argmax(deviation_evaluation_function)
        x_max_std = np.array([dom_variable.reshape(-1)[idx_max_std] for dom_variable in [X1_dom, X2_dom]])
        max_std = np.max(deviation_evaluation_function)
        self.logger.info(f'Maximum standard deviation found at {x_max_std} with deviation {max_std}')
        plot_mean_variance(X1_dom, X2_dom, m_all, v_all, filename=self.plot_path + '/total.pdf',
                           X=np.array([x_max_std]))

        hdffile.create_dataset('x_opt', data=x_opt)
        hdffile.create_dataset('x_max_std', data=x_max_std)

        hdffile.close()
        return x_opt, x_max_std


def plot_mean_variance(X1_dom, X2_dom, m, v, X=None, training_cost=None, filename=None, **kwargs):
        fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
        ax[0].contourf(X1_dom, X2_dom, m)

        ax[0].set_title(kwargs.get('ax0_title', 'Posterior Mean'))
        nrm = plt.Normalize(vmin=np.min(m), vmax=np.max(m))
        sm = plt.cm.ScalarMappable(nrm)
        plt.colorbar(sm, ax=ax[0])

        cmap = plt.get_cmap('viridis')
        if X is not None:
            if training_cost is not None:
                facecolor = cmap(nrm(training_cost))
            else:
                facecolor = 'r'
            ax[0].scatter(X[:, 0], X[:, 1], marker='o',
                          facecolor=facecolor,
                          lw=0.2,
                          edgecolor='k')
            ax[1].scatter(X[:, 0], X[:, 1])

        # VARIANCE
        nrm = plt.Normalize(vmin=np.min(v), vmax=np.max(v))
        sm = plt.cm.ScalarMappable(nrm, cmap=cmap)
        ax[1].contourf(X1_dom, X2_dom, v)
        plt.colorbar(sm, ax=ax[1])

        ax[1].set_title(kwargs.get('ax1_title', 'Standard Deviation'))

        if filename is not None:
            plt.savefig(filename)
