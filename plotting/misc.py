import matplotlib.pyplot as plt
import numpy as np

def plot_2d_array_of_config_results_traces(configs_array, results_array, key_order):

    n_x = len(configs_array[key_order[0]])
    n_y = len(configs_array[key_order[1]])
    fig, ax = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))
    n_seeds = results_array.shape[2]
    for i_x in range(n_x):
        for i_y in range(n_y):
            ax[i_x, i_y].plot(results_array[i_x, i_y])

def plot_2d_array_of_config_results(configs_array, results_array, key_order,
                                    log_scale=False, tick_rounding=3, **imshow_kwargs):
    """Given an array of configs (must be 2D) and corresponding results as
    floats, plots the result in a 2D grid averaging over random seeds."""

    fig = plt.figure()

    plt.imshow(results_array.mean(-1), **imshow_kwargs)

    if log_scale:
        plt.yticks(range(results_array.shape[0]),
                   np.round(np.log10(configs_array[key_order[0]]),
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(np.log10(configs_array[key_order[1]]),
                            tick_rounding))
    else:
        plt.yticks(range(results_array.shape[0]),
                   np.round(configs_array[key_order[0]],
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(configs_array[key_order[1]],
                            tick_rounding))

    plt.ylabel(key_order[0])
    plt.xlabel(key_order[1])
    plt.colorbar()

    return fig

def plot_3d_or_4d_array_of_config_results(configs_array, results_array, key_order,
                                          tick_rounding=3, color_bar=True,
                                          **imshow_kwargs):
    """Given an array of configs (must be 3-4D) and corresponding results as
    floats, plots the result in a grid averaging over random seeds."""

    d_grid = len(results_array.shape) - 3

    if d_grid == 1:
        n_x = 1
        n_y = results_array.shape[2]
    elif d_grid == 2:
        n_x, n_y = results_array.shape[2:4]
    else:
        #from pdb import set_trace
        #set_trace()
        raise ValueError('Configs must be 3 or 4 dimensional')

    fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))

    for i_x in range(n_x):
        for i_y in range(n_y):

            if d_grid == 1:
                results_slice = results_array[:,:,i_y,:].mean(-1)
                ax = axes[i_y]
            if d_grid == 2:
                results_slice = results_array[:,:,i_x,i_y,:].mean(-1)
                ax = axes[i_x, i_y]

            mappable = ax.imshow(results_slice, **imshow_kwargs)

            ax.set_yticks(list(range(len(configs_array[key_order[0]]))))
            ax.set_xticks(list(range(len(configs_array[key_order[1]]))))
            if type(configs_array[key_order[0]][0]) != str:
                ax.set_yticklabels(np.round(configs_array[key_order[0]],
                                            tick_rounding))
            else:
                ax.set_yticklabels(configs_array[key_order[0]])
            if type(configs_array[key_order[1]][0]) != str:
                ax.set_xticklabels(np.round(configs_array[key_order[1]],
                                            tick_rounding))
            else:
                ax.set_xticklabels(configs_array[key_order[1]])

            ax.set_ylabel(key_order[0])
            ax.set_xlabel(key_order[1])

            if d_grid == 1:
                y_param = key_order[2]
                title = y_param + '= {}'.format(configs_array[y_param][i_y])
            if d_grid == 2:
                x_param = key_order[2]
                y_param = key_order[3]
                title = '{} = {}, {} = {}'.format(x_param,
                                                  configs_array[x_param][i_x],
                                                  y_param,
                                                  configs_array[y_param][i_y])

            ax.set_title(title)

    if color_bar:
        fig.colorbar(mappable=mappable, ax=ax)

    return fig