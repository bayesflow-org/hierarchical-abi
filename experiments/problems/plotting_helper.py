import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_simulation_output(sim_output, title_prefix="Time", cmap="viridis",
                                same_scale=True, scales=None, mask=None, save_path=None, add_scale_bar=False):
    """
    Visualize the full simulation trajectory on a grid of subplots.

    Parameters:
        sim_output (np.ndarray): Simulation trajectory output.
            For a single simulation, it can be either:
              - 2D: shape (grid_size, n_time_points)
              - 3D: shape (n_grid, n_grid, n_time_points)
        title_prefix (str, list): Prefix for subplot titles.
        cmap (str): Colormap for imshow when visualizing 2D grid outputs.
        same_scale (bool): Whether to use the same color scale for all subplots.
        scales (list): List of tuples specifying the color scale for each subplot.
        mask (np.ndarray): Binary 2D mask of shape (n_grid, n_grid) where 0 indicates blacked-out areas.
        save_path (str): Path to save the figure.
        add_scale_bar (bool): Whether to add a scale bar to the plots.
    """
    if sim_output.ndim == 2:
        # (n_grid, n_time_points)
        n_grid = int(np.sqrt(sim_output.shape[0]))
        sim_output = sim_output[:n_grid**2, :]
        sim_output = sim_output.reshape(n_grid, n_grid, -1)
    elif sim_output.ndim == 3:
        n_grid = sim_output.shape[1]
        sim_output = sim_output.reshape(n_grid, n_grid, -1)
    else:
        raise ValueError("Simulation output must be 2D or 3D.")

    # Determine number of time points.
    n_time_points = sim_output.shape[-1]

    # Automatically choose grid layout (approximate square).
    n_cols = n_time_points
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), tight_layout=True)
    # Flatten axes array in case it's 2D.
    axes = axes.flatten()
    cmap = plt.get_cmap(cmap).copy()

    for i in range(n_time_points):
        ax = axes[i]
        img = sim_output[:, :, i].copy()

        # Check if the grid is 1D or 2D.
        # 2D grid: shape (n_grid, n_grid, n_time_points)
        if mask is not None:
            # Set masked regions to a distinct value (black later via colormap)
            img[~mask] = np.nan
            cmap.set_bad(color="black")
        if scales is not None:
            # Use provided scales‚
            vmin, vmax = scales[i]
            cmap.set_under(color='black')
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        elif same_scale:
            cmap.set_under(color='black')
            im = ax.imshow(img, cmap=cmap, vmin=sim_output.min(), vmax=sim_output.max())
        else:
            im = ax.imshow(img, cmap=cmap)
        if isinstance(title_prefix, list):
            ax.set_title(title_prefix[i])
        else:
            ax.set_title(f"{title_prefix} {i}")
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_xticks([])
        ax.set_yticks([])

        if add_scale_bar:
            # Define the pixel-to-micron ratio
            microns_per_pixel = 135./512
            scale_bar_length_um = 10
            scale_bar_length_px = int(scale_bar_length_um / microns_per_pixel)

            # Position the scale bar in the upper-right corner
            x0 = img.shape[1] - scale_bar_length_px - 20
            y0 = 30

            # Add the scale bar
            ax.hlines(y=y0, xmin=x0, xmax=x0 + scale_bar_length_px, color='white', linewidth=2)

            # Add label
            ax.text(x0 + scale_bar_length_px / 2, y0 - 5, f'{scale_bar_length_um} µm',
                    color='white', ha='center', va='bottom', fontsize=8)

    # Hide any unused subplots.
    for j in range(n_time_points, len(axes)):
        axes[j].axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.show()
    return


def plot_shrinkage(global_samples, local_samples, ci=95, min_max=None):
    """
    Plots the shrinkage of local estimates toward the global mean for each n_data.

    Parameters:
      global_samples: np.ndarray of shape (n_data, n_samples, 2)
                      The last dimension holds [global_mean, log_std].
      local_samples:  np.ndarray of shape (n_data, n_samples, n_individuals, 1)
                      The last dimension holds the local parameter.
      ci:             Confidence interval percentage (default 95).
    """
    n_data, n_samples, _ = global_samples.shape
    n_individuals = local_samples.shape[2]

    # Create a subplot for each n_data
    nrows, ncols = int(np.ceil(n_data / 4)), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, int(np.ceil(n_data / 4))*2),
                             sharex=True, sharey=True, tight_layout=True)
    axes = axes.flatten()

    # If there is only one subplot, wrap it in a list for consistent indexing.
    if n_data == 1:
        axes = [axes]

    for i in range(n_data):
        ax = axes[i]

        # Process global posterior for this n_data:
        global_mean_samples = global_samples[i, :, 0]
        global_mean_est = np.mean(global_mean_samples)
        global_ci = [global_mean_est-1.96*np.mean(np.exp(global_samples[i, :, 1])),
                     global_mean_est+1.96*np.mean(np.exp(global_samples[i, :, 1]))]

        # Process local posterior for each individual at data index i:
        local_means = np.zeros(n_individuals)
        local_cis = np.zeros((n_individuals, 2))

        for j in range(n_individuals):
            samples_j = local_samples[i, :, j, 0]
            local_means[j] = np.mean(samples_j)
            local_cis[j, :] = np.percentile(samples_j, [50 - ci/2, 50 + ci/2])

        indices = np.arange(n_individuals)
        # Plot local estimates with error bars
        h1 = ax.errorbar(indices, local_means,
                    yerr=[local_means - local_cis[:, 0], local_cis[:, 1] - local_means],
                    fmt='o', capsize=5, label='Local posterior mean')

        # Plot the global estimate as a horizontal dashed line
        h2 = ax.axhline(global_mean_est, color='red', linestyle='--', label='Global posterior mean')
        # Shade the global CI
        h3 = ax.fill_between(indices, global_ci[0], global_ci[1],
                        color='red', alpha=0.2, label='Global 95% CI')

        ax.set_ylabel("Parameter Value")
        ax.set_title(f"Data {i}")
        if min_max is not None:
            ax.set_ylim(min_max)
    fig.legend(handles=[h1, h2, h3], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.05))
    axes[-1].set_xlabel("Individual Index")
    for i in range(n_data, len(axes)):
        # disable axis
        axes[i].set_visible(False)
    plt.show()

