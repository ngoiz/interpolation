import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import interpolation.interface as interface


def plot_optimisation_data(optimisation_info, optimisation_id, opt=None):
    path_to_data = optimisation_info.optimisation_data[optimisation_id]['output_directory']
    if opt is None:
        report_file = path_to_data + f'/evaluations_report.txt'

        evaluations = np.loadtxt(report_file, skiprows=1)
        iter_id = evaluations[:, 0]
        cost = -evaluations[:, 1]
        param_values = evaluations[:, 2:]
    else:
        cost = -opt.Y
        param_values = opt.X
    source_points = param_values[cost == 0.]
    n_source_points = source_points.shape[0]

    training_data = interface.yaml_to_array(path_to_data + '/training_data.yaml')

    plt.figure()
    cmap = plt.get_cmap('Reds')
    nrm = plt.Normalize(vmin=0, vmax=np.ceil(np.max(cost)))
    plt.scatter(training_data[:, 0], training_data[:, 1], marker='x', color='k')

    plt.plot(param_values[n_source_points:, 0], param_values[n_source_points:, 1], ls='--')
    plt.scatter(param_values[n_source_points:, 0], param_values[n_source_points:, 1], marker='o',
                facecolor=cmap(nrm(cost[n_source_points:])), edgecolor='k', lw=0.2)
    plt.colorbar(mpl.cm.ScalarMappable(nrm, cmap), label='Cost Function')

    # plt.text(param_values[n_source_points + 1, 0], param_values[n_source_points + 1, 1], '1')
    # plt.text(param_values[-1, 0], param_values[-1, 1], 'end')
    plt.savefig(path_to_data + '/optimisation_path2d.pdf')

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(param_values[n_source_points:, 0], param_values[n_source_points:, 1], ls='--')
    ax.scatter(param_values[:n_source_points, 0], param_values[:n_source_points, 1], cost[:n_source_points],
               marker='o', facecolor='k')
    ax.scatter(param_values[n_source_points:, 0], param_values[n_source_points:, 1], cost[n_source_points:],
               marker='o', facecolor='r')
    for i_point in range(n_source_points, param_values.shape[0]):
        ax.plot([param_values[i_point, 0], param_values[i_point, 0]],
                [param_values[i_point, 1], param_values[i_point, 1]], [0., cost[i_point]], color='tab:blue')
    ax.set_xlabel('Angle of Attack')
    ax.set_ylabel('Free stream velocity')
    ax.set_zlabel('Cost function')
    plt.savefig(path_to_data + '/optimisation_path3d.pdf')


def plot_acquisition_functions(optimisation_info, optimisation_id, optimiser):
    path_to_data = optimisation_info.optimisation_data[optimisation_id]['output_directory']

    fig = plt.figure()
    optimiser.plot_acquisition(label_x='Angle of Attack', label_y='Free stream velocity')
    plt.tight_layout()
    plt.savefig(path_to_data + '/optimisation_acquisition.pdf')

    fig = plt.figure()
    optimiser.plot_convergence()
    plt.savefig(path_to_data + '/optimisation_convergence.pdf')
