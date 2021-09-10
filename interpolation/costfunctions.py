import numpy as np
import scipy.spatial.distance as distance

dict_of_cost_functions = {}
dict_of_damping_functions = {}


def cost_function(arg):
    global dict_of_cost_functions
    try:
        arg.cost_function_name
    except AttributeError:
        raise AttributeError('Class defined as cost function has no given name')
    dict_of_cost_functions[arg.cost_function_name] = arg
    return arg


def damping_penalty(func):
    try:
        func.name
    except AttributeError:
        raise AttributeError('Class classified as DampingPenalty has not been given a .name')
    dict_of_damping_functions[func.name] = func

    return func


class BaseCostFunction:

    def __init__(self):
        self.settings = {}

    def initialise(self, settings=None):
        if settings is not None:
            self.settings = settings

    def __call__(self, t_case, i_case):
        cost = 0
        return cost


@cost_function
class EigenvalueComparison(BaseCostFunction):
    cost_function_name = 'EigenvalueComparison'

    def __init__(self):
        super().__init__()

        self.damping_penalty = None

    def initialise(self, settings=None):
        super().initialise(settings)

        self.damping_penalty = get_damping_penalty(self.settings.get('damping_penalty_name', 'BaseDamping'),
                                                   self.settings.get('damping_penalty_settings', None))

        print('Set the damping penalty to ', self.damping_penalty.name)

    def __call__(self, t_case, i_case):
        min_eigs = np.min([i_case.eigs.shape[0], t_case.eigs.shape[0]])

        t_eigs = self._filter_eigenvalues(t_case.eigs[:min_eigs, :])
        i_eigs = self._filter_eigenvalues(i_case.eigs[:min_eigs, :])

        eig_error = distance.cdist(i_eigs, t_eigs, 'euclidean')

        damp = t_eigs[:, 0] / np.sqrt(t_eigs[:, 0] ** 2 + t_eigs[:, 1] ** 2)

        cost = 0
        for i_node in range(i_eigs.shape[0]):
            idx_t_eig = np.argmin(eig_error[i_node])
            cost_contribution = self.damping_penalty(damp[idx_t_eig]) * eig_error[i_node, idx_t_eig]
            cost += cost_contribution

        return cost

    def _filter_eigenvalues(self, eigs):

        conditions = (eigs[:, 0] > self.settings.get('min_real', -10)) * \
                     (eigs[:, 1] >= 0) * \
                     (eigs[:, 1] < self.settings.get('max_imag', 600))

        return eigs[conditions, :]


@damping_penalty
class BaseDampingPenalty:
    name = 'BaseDamping'

    def __init__(self, settings=None):
        self.settings = dict()
        if settings is not None:
            self.settings = settings

    def __call__(self, damping):

        return 1


@damping_penalty
class GaussianPenalty(BaseDampingPenalty):
    name = 'GaussianPenalty'

    def __init__(self, settings=None):
        super().__init__(settings=settings)

        self.multiplying_factor = self.settings.get('multiplying_factor', 1)
        self.std_deviation_factor = self.settings.get('std_deviation_factor', 0.03)

    def __call__(self, damping):

        return self.multiplying_factor * np.exp(-damping ** 2 / (2 * self.std_deviation_factor ** 2))


def get_cost_function(cost_function_name):
    return dict_of_cost_functions[cost_function_name]()


def get_damping_penalty(damping_penalty_name, settings=None):
    return dict_of_damping_functions[damping_penalty_name](settings)
