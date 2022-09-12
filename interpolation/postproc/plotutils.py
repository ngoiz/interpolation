import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import interpolation.postproc.manager as manager
import interpolation.utils as utils

def test_data_deflection_to_meshgrid(testing_data, alpha_vec=None, u_inf_vec=None):
    x1_dom = []
    x2_dom = []
    z_deflection = []
    if alpha_vec is None:
        alpha_vec = np.linspace(0, 5, 11)
    if u_inf_vec is None:
        u_inf_vec = np.linspace(30, 90, 31)


    z_deflection = np.zeros((len(alpha_vec), len(u_inf_vec)))
    for i, alpha in enumerate(alpha_vec):
        for j, u_inf in enumerate(u_inf_vec):
            case = testing_data.aeroelastic.find_param({'alpha': alpha, 'u_inf': u_inf})
            case.load_deflection()
            if case.deflection is None:
                print('No deflection for case:')
                print(alpha, u_inf)
                continue
            else:
                z_deflection[i, j] = case.deflection[-1, -1] / 0.55
    return alpha_vec, u_inf_vec, z_deflection.T


def plot_set(iter_info, set_id, output_dir=None):
    set_data = iter_info['data']
    set_info = iter_info['info']

    fig, ax = plot_mean_or_variance(set_data, set_info, set_id, 'mean')

    if output_dir is not None:
        plt.savefig(output_dir + 'mean_{:03g}.pdf'.format(set_id))

    plt.close(fig)

    fig, ax = plot_mean_or_variance(set_data, set_info, set_id, 'variance')
    if output_dir is not None:
        plt.savefig(output_dir + 'variance_{:03g}.pdf'.format(set_id))
    plt.close(fig)


def plot_mean_or_variance(set_data, set_info, set_id, dataset='mean'):
    x1 = set_data['X1_dom']
    x2 = set_data['X2_dom']
    if dataset == 'mean':
        data = np.abs(set_data['mean_{:03g}'.format(set_id)])
    elif dataset == 'variance':
        data = set_data['variance_{:03g}'.format(set_id)]

    w = 12 / 2.54
    h = w * 0.8
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()

    cmap = plt.get_cmap('viridis')
    ax.contourf(x1, x2, data)
    nrm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))
    sm = plt.cm.ScalarMappable(nrm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.05)
    cb.ax.tick_params(labelsize=8)

    training_points = manager.get_points(set_info, set_id, which_set='training')
    testing_points = manager.get_points(set_info, set_id, which_set='testing')
    testing_cost = np.abs(manager.get_cost(set_info, set_id))

    ax.scatter(training_points[:, 0], training_points[:, 1], facecolor='tab:red',
               edgecolor='tab:blue', label='Training Set', marker='^')

    ax.scatter(testing_points[:, 0], testing_points[:, 1],
               facecolor=cmap(nrm(testing_cost)),
               edgecolor='r',
               label='Testing Set', lw=0.4)

    plt.legend(fontsize=8)
    ax.set_xlabel('Angle of attack, deg', fontsize=8)
    ax.set_ylabel('Free stream velocity, m/s', fontsize=8)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)

    return fig, ax


def plot_single_total(case_info, iter_id, function='interpolation'):
    x1 = case_info[0]['data']['X1_dom']
    x2 = case_info[0]['data']['X2_dom']

    m_all = np.abs(case_info[iter_id]['data']['mean_all'])
    v_all = case_info[iter_id]['data']['var_all']

    if function == 'interpolation':
        interpolation_function = m_all * v_all
    elif function == 'mean':
        interpolation_function = m_all
    elif function == 'variance':
        interpolation_function = v_all
    else:
        raise NameError

    w = 12 / 2.54
    h = w * 0.8
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()

    plt.contourf(x1, x2, interpolation_function)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)

    ax.set_xlabel('Angle of attack, deg', fontsize=8)
    ax.set_ylabel('Free stream velocity, m/s', fontsize=8)

    idx_max_std = np.argmax(interpolation_function)
    x_max_std = np.array([dom_variable.reshape(-1)[idx_max_std] for dom_variable in [x1, x2]])
    ax.scatter(x_max_std[0], x_max_std[1], s=40, facecolor='r', edgecolor='k')

    ax.set_xlim(2, 5)
    ax.set_ylim(10, 70)

    return fig, ax, x_max_std


def plot_totals(case_info, iter_id, iter_output_directory):
    fig, ax, x_opt_func = plot_single_total(case_info, iter_id, function='interpolation')
    plt.savefig(iter_output_directory + '/interpolation_function.pdf')
    plt.close(fig)

    fig, ax, x_opt_var = plot_single_total(case_info, iter_id, function='variance')
    plt.savefig(iter_output_directory + '/variance_function.pdf')
    plt.close(fig)

    fig, ax, x_opt_mean = plot_single_total(case_info, iter_id, function='mean')
    plt.savefig(iter_output_directory + '/mean_function.pdf')
    plt.close(fig)

    np.savetxt(iter_output_directory + '/optima.txt', np.column_stack((x_opt_func, x_opt_mean, x_opt_var)),
               header='Interp Func \tMean \tVariance')


def meshgrid_from_array(out_array):
    x, y, z = out_array[:, 1], out_array[:, 0], out_array[:, 2]

    ngridx = 100
    ngridy = 100
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = mpl.tri.Triangulation(x, y)
    interpolator = mpl.tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    return Xi, Yi, zi


def flutter_envelope_plot(e_case, xt, yt, alpha_vec):
    w = 12 / 2.54
    h = w * 0.8
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()

    interpolated_flutter_speeds = []
    testing_flutter_speeds = []
    tp = plt.scatter(xt, yt, marker='o', facecolor='tab:orange', label='Training Points', edgecolor='y', s=45, lw=2)

    for i, alpha in enumerate(alpha_vec):
        v, g, f = utils.produce_vgf(e_case.testing_data, 'alpha', alpha)
        vi, gi, fi = utils.produce_vgf(e_case.interpolated_data, 'alpha', alpha)

        interpolated_flutter_speeds.append(
            utils.compute_flutter(e_case.interpolated_data, 'alpha', alpha, vmin=40, wnmax=50 * np.pi * 2))
        testing_flutter_speeds.append(
            utils.compute_flutter(e_case.testing_data, 'alpha', alpha, vmin=40, wnmax=50 * np.pi * 2))
        if i == 0:
            ilab = 'Interpolated $\zeta=0$'
            tlab = 'True $\zeta=0$'
        else:
            ilab = None
            tlab = None
        ip = plt.scatter(np.ones_like(interpolated_flutter_speeds[i]) * alpha, interpolated_flutter_speeds[i],
                         marker='x', color='b', label=ilab)
    #         plt.fill_between(np.ones_like(testing_flutter_speeds[i]) * alpha, testing_flutter_speeds[i], marker='o', edgecolor='k', facecolor='none', label=tlab)
    print(testing_flutter_speeds)
    testing_flutter_speeds = np.array(testing_flutter_speeds)
    print(testing_flutter_speeds.shape)
    plt.fill_between(alpha_vec, testing_flutter_speeds[:, 0], testing_flutter_speeds[:, 1], interpolate=True,
                     facecolor='r', alpha=0.5,
                     ls='--', edgecolor='k', lw=2)  # , marker='o', edgecolor='k', facecolor='none', label=tlab)

    l = plt.Line2D([], [], ls='--', color='k', lw=2, label='True $\zeta=0$')

    plt.xlim(2., 5.)
    plt.ylim(9, 71)
    plt.legend([tp, ip, l], ['Training Points', 'Interpolated $\zeta=0$', 'True $\zeta=0$'], fontsize=8,
               loc='lower right')
    plt.xlabel('Angle of Attack, deg')
    plt.ylabel('Free stream velocity, m/s')
    plt.grid()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)

    return fig, ax