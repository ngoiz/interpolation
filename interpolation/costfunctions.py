import numpy as np
import scipy.spatial.distance as distance
import logging
import sharpytools.linear.stability as stability

dict_of_cost_functions = {}
dict_of_damping_functions = {}


def cost_function(arg):
    global dict_of_cost_functions
    try:
        arg.name
    except AttributeError:
        raise AttributeError('Class defined as cost function has no given name')
    dict_of_cost_functions[arg.name] = arg
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
        self.logger = logging.getLogger(__name__)

    def initialise(self, settings=None):
        if settings is not None:
            self.settings = settings

    def __call__(self, t_case, i_case):
        cost = 0
        return cost


@cost_function
class RootLoadsComparison(BaseCostFunction):
    name = 'RootLoadsComparison'

    def __call__(self, t_case, i_case):
        # Assert that the frequency response is correct - i.e. that the size is as desired
        #
        # evaluate the steady state loads
        try:
            tss0 = t_case.bode.ss0[-6:, 0]
            iss0 = i_case.bode.ss0[-6:, 0]
        except AttributeError:
            import pdb; pdb.set_trace()

        assert tss0.shape == iss0.shape

        mag_tss0 = 20 * np.log10(np.abs(tss0))
        mag_iss0 = 20 * np.log10(np.abs(iss0))

        # print(f'Interpolated loads:')
        # print(mag_iss0)
        # print('RealLoads')
        # print(mag_tss0)

        rel_error_shear = np.max(np.abs(mag_iss0[:3] - mag_tss0[:3])) / np.max(np.abs(mag_tss0[:3]))
        rel_error_moment = np.max(np.abs(mag_iss0[3:] - mag_tss0[3:])) / np.max(np.abs(mag_tss0[3:]))

        max_error = max([rel_error_shear, rel_error_moment])
        # print('RootLoads max relative error: ', max_error)
        return -max_error


@cost_function
class FrobeniusNorm(BaseCostFunction):
    name = 'FrobeniusNorm'

    def __call__(self, t_case, i_case):

        # # frobenius norm of each matrix
        # t_ss = t_case.ss
        # i_ss = i_case.ss
        #
        # if t_ss is None:
        #     raise ValueError('Unable to locate state-space attribute for testing case')
        # if i_ss is None:
        #     raise ValueError('Unable to locate state-space attribute for interpolated case')
        #
        # rel_frobenius_error = np.linalg.norm(t_ss.A - i_ss.A, ord='fro') / np.linalg.norm(t_ss.A, ord='fro')

        tss0 = t_case.bode.ss0[-6:, :]
        iss0 = i_case.bode.ss0[-6:, :]

        assert tss0.shape == iss0.shape

        rel_frobenius_error = np.linalg.norm(tss0 - iss0, ord='fro') / np.linalg.norm(tss0, ord='fro')

        return rel_frobenius_error

@cost_function
class FrequencyRelativeError(BaseCostFunction):
    name = 'FrequencyRelativeError'

    def __call__(self, t_case, i_case):

        err = self.error_metric(t_case.bode.yfreq[-6:, :, :], i_case.bode.yfreq[-6:, :, :])
        return err

    @staticmethod
    def error_metric(y1, y2):
        p, m, _ = y1.shape
        err = np.zeros((p, m))
        for pi in range(p):
            for mi in range(m):
                err[pi, mi] = np.max(np.abs(y1[pi, mi, :] - y2[pi, mi, :])) / np.max(
                    np.abs(y1[pi, mi, :]))
        return np.max(err)


@cost_function
class EigenvalueComparison(BaseCostFunction):
    name = 'EigenvalueComparison'

    def __init__(self):
        super().__init__()

        self.damping_penalty = None

    def initialise(self, settings=None):
        super().initialise(settings)

        self.damping_penalty = get_damping_penalty(self.settings.get('damping_penalty_name', 'ConstantDamping'),
                                                   self.settings.get('damping_penalty_settings', None))

        self.logger.info(f'Set the damping penalty to {self.damping_penalty.name}')

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

        return cost / i_eigs.shape[0]

    def _filter_eigenvalues(self, eigs):

        # backward compatibility
        try:
            self.settings['min_real']
        except KeyError:
            pass
        else:
            self.settings['remin'] = self.settings['min_real']
        try:
            self.settings['max_imag']
        except KeyError:
            pass
        else:
            self.settings['wdmax'] = self.settings['max_imag']

        conditions = stability.filter_velocity_eigenvalues(np.ones(eigs.shape[0]), eigs,
                                                           **self.settings)

        # conditions = (eigs[:, 0] > self.settings.get('min_real', -10)) * \
        #              (eigs[:, 1] >= 0) * \
        #              (eigs[:, 1] < self.settings.get('max_imag', 600))

        if all([not c for c in conditions]):  # if all conditions are false
            self.logger.error('Settings to filter eigenvalues are too strict. No eigenvalues fall within the desired range')
            raise IndexError('Settings to filter eigenvalues are too strict. No eigenvalues fall within the desired range')

        return eigs[conditions, :]


@damping_penalty
class ConstantDampingPenalty:
    name = 'ConstantDamping'

    def __init__(self, settings=None):
        self.settings = dict()
        if settings is not None:
            self.settings = settings

    def __call__(self, damping):

        return 1


@damping_penalty
class GaussianPenalty(ConstantDampingPenalty):
    name = 'GaussianPenalty'

    def __init__(self, settings=None):
        super().__init__(settings=settings)

        self.multiplying_factor = self.settings.get('multiplying_factor', 1)
        self.std_deviation_factor = self.settings.get('std_deviation_factor', 0.03)  # pct damping

    def __call__(self, damping):

        return -self.multiplying_factor * np.exp(-damping ** 2 / (2 * self.std_deviation_factor ** 2))


def get_cost_function(cost_function_name):
    return dict_of_cost_functions[cost_function_name]()


def get_damping_penalty(damping_penalty_name, settings=None):
    return dict_of_damping_functions[damping_penalty_name](settings)
