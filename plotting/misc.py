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


def compute_average_flow_field_cartesian_batches(all_batches, n_x_bins=20, n_y_bins=20,
                                                 x_range=(-2.0, 2.0), y_range=(-2.0, 2.0)):
    """
    Computes the average flow field over a Cartesian grid from a list of time series batches.
    Each batch is treated independently so that the end of one batch and the start of the next
    do not get connected.

    Parameters:
      all_batches : list of np.ndarray
          Each array is of shape (T, 2) representing a 2D trajectory for one batch.
      n_x_bins : int
          Number of bins in the x direction.
      n_y_bins : int
          Number of bins in the y direction.
      x_range : tuple (x_min, x_max)
          Range in the x direction.
      y_range : tuple (y_min, y_max)
          Range in the y direction.

    Returns:
      avg_flow : np.ndarray, shape (n_x_bins, n_y_bins, 2)
          The average velocity vector in each grid cell.
      counts : np.ndarray, shape (n_x_bins, n_y_bins)
          Number of velocity samples in each cell.
      x_edges, y_edges : np.ndarray
          The bin edges for x and y.
    """
    # Prepare arrays for accumulation.
    velocity_sum = np.zeros((n_x_bins, n_y_bins, 2))
    counts = np.zeros((n_x_bins, n_y_bins))

    # Define the grid edges.
    x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    # Process each batch separately.
    for batch in all_batches:
        # Skip batches that are too short.
        if batch.shape[0] < 2:
            continue
        # Compute velocities within this batch.
        velocities = np.diff(batch, axis=0)
        # Use midpoints of consecutive points for binning.
        positions = 0.5 * (batch[:-1] + batch[1:])

        for pos, vel in zip(positions, velocities):
            x, y = pos
            # Skip if outside region.
            if (x < x_range[0]) or (x > x_range[1]) or (y < y_range[0]) or (y > y_range[1]):
                continue
            # Determine bin indices.
            x_bin = int(np.floor((x - x_range[0]) / (x_range[1] - x_range[0]) * n_x_bins))
            y_bin = int(np.floor((y - y_range[0]) / (y_range[1] - y_range[0]) * n_y_bins))
            # Handle edge cases.
            if x_bin == n_x_bins:
                x_bin = n_x_bins - 1
            if y_bin == n_y_bins:
                y_bin = n_y_bins - 1
            # Accumulate.
            velocity_sum[x_bin, y_bin] += vel
            counts[x_bin, y_bin] += 1

    # Compute the average velocities where counts > 0.
    avg_flow = np.zeros_like(velocity_sum)
    mask = counts > 0
    avg_flow[mask] = velocity_sum[mask] / counts[mask, None]

    return avg_flow, counts, x_edges, y_edges


def plot_flow_field_cartesian_stream(avg_flow, x_edges, y_edges, N, density=1.5, linewidth=1, arrowsize=1,
                                     Z=None, limit_cycle=None, plot_only_first=True, fp=None, Z_long=None,
                                     z_labels=None, sqrt_ticks=True, plot_arrows=False, plot_endpoints=False, fgsz=1.5,
                                     xlimylim=None):
    """
    Plots the average flow field on a Cartesian grid using streamplot.

    Parameters:
      avg_flow : np.ndarray, shape (n_x_bins, n_y_bins, 2)
          The average velocity vector in each grid cell.
      x_edges, y_edges : np.ndarray
          The bin edges for x and y.
      density : float
          Controls the closeness of streamlines.
      linewidth : float
          Line width of the streamlines.
      arrowsize : float
          Size of the arrows.
    """
    # Compute bin centers.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Extract averaged velocity components.
    U = avg_flow[:, :, 0]
    V = avg_flow[:, :, 1]

    # For streamplot, provide 1D coordinate arrays and transpose U and V.
    fig = plt.figure(figsize=(fgsz, fgsz))
    if Z is not None:
        for z in Z:
            plt.plot(z[:, 0], z[:, 1], color='#AC85BC', alpha=1, linewidth=1, zorder=2)
            if plot_arrows:
                # Compute directional differences between successive points
                dx = np.diff(z[:, 0])
                dy = np.diff(z[:, 1])
                speed = np.sqrt(dx ** 2 + dy ** 2)
                dx = dx / speed
                dy = dy / speed

                # Choose arrow placement interval (adjust based on your data density)
                arrow_interval = max(1, len(z) // 20)

                # Plot arrows using quiver. We use z[:-1] since np.diff returns one fewer element.
                plt.quiver(z[:-1:arrow_interval, 0], z[:-1:arrow_interval, 1],
                           dx[::arrow_interval], dy[::arrow_interval],
                           color='#AC85BC', scale_units='xy', angles='xy', scale=0.8,
                           width=0.008,  # Arrow shaft thickness
                           headwidth=15,  # Width of the arrow head
                           headlength=15,  # Length of the arrow head
                           headaxislength=15, zorder=3)

            if plot_only_first:
                break
            if plot_endpoints:
                plt.plot([z[-1, 0]], [z[-1, 1]], 'x', markersize=3, color='k', zorder=3, alpha=1)
    speed = np.log10(np.sqrt(U ** 2 + V ** 2) + 0.001)
    plt.streamplot(x_centers, y_centers, U.T, V.T, density=density,
                   color=speed, cmap='Greys', linewidth=linewidth, arrowsize=arrowsize)

    if limit_cycle is not None:
        plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], color='k', linestyle='--', linewidth=0.8)
    if fp is not None:
        for fp_ in fp:
            plt.plot([fp_[0]], [fp_[1]], 'x', color='k', markersize=3)
    if Z_long is not None:
        plt.plot(Z_long[:, 0], Z_long[:, 1], color='#AC85BC', linewidth=0.7)
    if z_labels == 1:
        plt.xlabel('$z^{(1)}_1(t)$', fontsize=8)
        plt.ylabel('$z^{(1)}_2(t)$', fontsize=8)
    elif z_labels == 2:
        plt.xlabel('$z^{(2)}_1(t)$', fontsize=8)
        plt.ylabel('$z^{(2)}_2(t)$', fontsize=8)
    else:
        plt.xlabel('$z_1(t)$', fontsize=8)
        plt.ylabel('$z_2(t)$', fontsize=8)
    # plt.title('Average Flow Field (Cartesian Grid, Batches)')
    if sqrt_ticks:
        plt.xticks([-np.sqrt(N), np.sqrt(N)], ['$-\sqrt{N}$', '$\sqrt{N}$'], fontsize=8)
        plt.yticks([-np.sqrt(N), np.sqrt(N)], ['$-\sqrt{N}$', '$\sqrt{N}$'], fontsize=8)
    else:
        pass
    plt.axis('equal')
    if xlimylim is not None:
        plt.xlim([-xlimylim, xlimylim])
        plt.ylim([-xlimylim, xlimylim])
    plt.show()

    return fig