import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def visualize_simulation_output(sim_output, title_prefix="Time", cmap="viridis", same_scale=True):
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
    """
    if sim_output.ndim == 2:
        # (n_time_points, n_grid)
        n_grid = int(np.sqrt(sim_output.shape[0]))
        sim_output = sim_output[:n_grid**2, :]
        sim_output = sim_output.reshape(n_grid, n_grid, -1)
    elif sim_output.ndim == 3:
        n_grid = sim_output.shape[0]
        sim_output = sim_output.reshape(n_grid, n_grid, -1)
    else:
        raise ValueError("Simulation output must be 2D or 3D.")

    # Determine number of time points.
    n_time_points = sim_output.shape[-1]

    # Automatically choose grid layout (approximate square).
    n_cols = n_time_points
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    # Flatten axes array in case it's 2D.
    axes = axes.flatten()

    for i in range(n_time_points):
        ax = axes[i]
        # Check if the grid is 1D or 2D.
        # 2D grid: shape (n_grid, n_grid, n_time_points)
        if same_scale:
            im = ax.imshow(sim_output[:, :, i], cmap=cmap, vmin=sim_output.min(), vmax=sim_output.max())
        else:
            im = ax.imshow(sim_output[:, :, i], cmap=cmap)
        if isinstance(title_prefix, list):
            ax.set_title(title_prefix[i])
        else:
            ax.set_title(f"{title_prefix} {i}")
        fig.colorbar(im, ax=ax)

    # Hide any unused subplots.
    for j in range(n_time_points, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    return


def plot_shrinkage(global_samples, local_samples, ci=95, min_max=None):
    """
    Plots the shrinkage of local estimates toward the global mean for each n_data.

    Parameters:
      global_samples: np.ndarray of shape (n_data, n_samples, 3)
                      The last dimension holds [alpha, global_mean, log_std].
      local_samples:  np.ndarray of shape (n_data, n_samples, n_individuals, 1)
                      The last dimension holds the local parameter.
      ci:             Confidence interval percentage (default 95).
    """
    if global_samples.shape[-1] == 3:
        global_samples = global_samples[:, :, 1:]
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


def petcolormap(m=256):
    """
    Generates a PET colormap similar to the MATLAB version.

    Args:
        m (int): Number of colors in the colormap.

    Returns:
        ListedColormap: A matplotlib colormap.
    """
    # Base colormap array (each row is an [R, G, B] triplet).
    # This is taken directly from your MATLAB function.
    c = np.array([
        [0,	0,	0],
        [0,	0,	2],
        [0,	0,	4],
        [0,	0,	6],
        [0,	0,	8],
        [0,	0,	10],
        [0,	0,	12],
        [0,	0,	14],
        [0,	0,	16],
        [0,	0,	17],
        [0,	0,	19],
        [0,	0,	21],
        [0,	0,	23],
        [0,	0,	25],
        [0,	0,	27],
        [0,	0,	29],
        [0,	0,	31],
        [0,	0,	33],
        [0,	0,	35],
        [0,	0,	37],
        [0,	0,	39],
        [0,	0,	41],
        [0,	0,	43],
        [0,	0,	45],
        [0,	0,	47],
        [0,	0,	49],
        [0,	0,	51],
        [0,	0,	53],
        [0,	0,	55],
        [0,	0,	57],
        [0,	0,	59],
        [0,	0,	61],
        [0,	0,	63],
        [0,	0,	65],
        [0,	0,	67],
        [0,	0,	69],
        [0,	0,	71],
        [0,	0,	73],
        [0,	0,	75],
        [0,	0,	77],
        [0,	0,	79],
        [0,	0,	81],
        [0,	0,	83],
        [0,	0,	84],
        [0,	0,	86],
        [0,	0,	88],
        [0,	0,	90],
        [0,	0,	92],
        [0,	0,	94],
        [0,	0,	96],
        [0,	0,	98],
        [0,	0,	100],
        [0,	0,	102],
        [0,	0,	104],
        [0,	0,	106],
        [0,	0,	108],
        [0,	0,	110],
        [0,	0,	112],
        [0,	0,	114],
        [0,	0,	116],
        [0,	0,	117],
        [0,	0,	119],
        [0,	0,	121],
        [0,	0,	123],
        [0,	0,	125],
        [0,	0,	127],
        [0,	0,	129],
        [0,	0,	131],
        [0,	0,	133],
        [0,	0,	135],
        [0,	0,	137],
        [0,	0,	139],
        [0,	0,	141],
        [0,	0,	143],
        [0,	0,	145],
        [0,	0,	147],
        [0,	0,	149],
        [0,	0,	151],
        [0,	0,	153],
        [0,	0,	155],
        [0,	0,	157],
        [0,	0,	159],
        [0,	0,	161],
        [0,	0,	163],
        [0,	0,	165],
        [0,	0,	167],
        [3,	0,	169],
        [6,	0,	171],
        [9,	0,	173],
        [12,	0,	175],
        [15,	0,	177],
        [18,	0,	179],
        [21,	0,	181],
        [24,	0,	183],
        [26,	0,	184],
        [29,	0,	186],
        [32,	0,	188],
        [35,	0,	190],
        [38,	0,	192],
        [41,	0,	194],
        [44,	0,	196],
        [47,	0,	198],
        [50,	0,	200],
        [52,	0,	197],
        [55,	0,	194],
        [57,	0,	191],
        [59,	0,	188],
        [62,	0,	185],
        [64,	0,	182],
        [66,	0,	179],
        [69,	0,	176],
        [71,	0,	174],
        [74,	0,	171],
        [76,	0,	168],
        [78,	0,	165],
        [81,	0,	162],
        [83,	0,	159],
        [85,	0,	156],
        [88,	0,	153],
        [90,	0,	150],
        [93,	2,	144],
        [96,	4,	138],
        [99,	6,	132],
        [102,	8,	126],
        [105,	9,	121],
        [108,	11,	115],
        [111,	13,	109],
        [114,	15,	103],
        [116,	17,	97],
        [119,	19,	91],
        [122,	21,	85],
        [125,	23,	79],
        [128,	24,	74],
        [131,	26,	68],
        [134,	28,	62],
        [137,	30,	56],
        [140,	32,	50],
        [143,	34,	47],
        [146,	36,	44],
        [149,	38,	41],
        [152,	40,	38],
        [155,	41,	35],
        [158,	43,	32],
        [161,	45,	29],
        [164,	47,	26],
        [166,	49,	24],
        [169,	51,	21],
        [172,	53,	18],
        [175,	55,	15],
        [178,	56,	12],
        [181,	58,	9],
        [184,	60,	6],
        [187,	62,	3],
        [190,	64,	0],
        [194,	66,	0],
        [198,	68,	0],
        [201,	70,	0],
        [205,	72,	0],
        [209,	73,	0],
        [213,	75,	0],
        [217,	77,	0],
        [221,	79,	0],
        [224,	81,	0],
        [228,	83,	0],
        [232,	85,	0],
        [236,	87,	0],
        [240,	88,	0],
        [244,	90,	0],
        [247,	92,	0],
        [251,	94,	0],
        [255,	96,	0],
        [255,	98,	3],
        [255,	100,	6],
        [255,	102,	9],
        [255,	104,	12],
        [255,	105,	15],
        [255,	107,	18],
        [255,	109,	21],
        [255,	111,	24],
        [255,	113,	26],
        [255,	115,	29],
        [255,	117,	32],
        [255,	119,	35],
        [255,	120,	38],
        [255,	122,	41],
        [255,	124,	44],
        [255,	126,	47],
        [255,	128,	50],
        [255,	130,	53],
        [255,	132,	56],
        [255,	134,	59],
        [255,	136,	62],
        [255,	137,	65],
        [255,	139,	68],
        [255,	141,	71],
        [255,	143,	74],
        [255,	145,	76],
        [255,	147,	79],
        [255,	149,	82],
        [255,	151,	85],
        [255,	152,	88],
        [255,	154,	91],
        [255,	156,	94],
        [255,	158,	97],
        [255,	160,	100],
        [255,	162,	103],
        [255,	164,	106],
        [255,	166,	109],
        [255,	168,	112],
        [255,	169,	115],
        [255,	171,	118],
        [255,	173,	121],
        [255,	175,	124],
        [255,	177,	126],
        [255,	179,	129],
        [255,	181,	132],
        [255,	183,	135],
        [255,	184,	138],
        [255,	186,	141],
        [255,	188,	144],
        [255,	190,	147],
        [255,	192,	150],
        [255,	194,	153],
        [255,	196,	156],
        [255,	198,	159],
        [255,	200,	162],
        [255,	201,	165],
        [255,	203,	168],
        [255,	205,	171],
        [255,	207,	174],
        [255,	209,	176],
        [255,	211,	179],
        [255,	213,	182],
        [255,	215,	185],
        [255,	216,	188],
        [255,	218,	191],
        [255,	220,	194],
        [255,	222,	197],
        [255,	224,	200],
        [255,	226,	203],
        [255,	228,	206],
        [255,	229,	210],
        [255,	231,	213],
        [255,	233,	216],
        [255,	235,	219],
        [255,	237,	223],
        [255,	239,	226],
        [255,	240,	229],
        [255,	242,	232],
        [255,	244,	236],
        [255,	246,	239],
        [255,	248,	242],
        [255,	250,	245],
        [255,	251,	249],
        [255,	253,	252],
        [255,	255,	255]
    ])

    n = c.shape[0]
    # Create a mapping from the original colormap positions to the new m points.
    xp = np.linspace(0, m-1, n)
    x = np.arange(m)
    r = np.interp(x, xp, c[:, 0])
    g = np.interp(x, xp, c[:, 1])
    b = np.interp(x, xp, c[:, 2])
    colormap = np.stack([r, g, b], axis=1) / 255.0
    # Normalize so that the maximum value is 1 (this mimics the MATLAB division by max(map(:))).
    colormap = colormap / colormap.max()
    return ListedColormap(colormap)
