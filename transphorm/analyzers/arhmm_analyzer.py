from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl
from pathlib import Path
import ssm
from dotenv import load_dotenv
from transphorm.framework_helpers import setup_comet_experimet
from typing import List, Optional
from torch import Tensor

from transphorm.preprocessors.loaders import AADataLoader
import polars as pl


def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1.0, 1.0))

    cdict = {
        "red": tuple(reds),
        "green": tuple(greens),
        "blue": tuple(blues),
        "alpha": tuple(alphas),
    }

    cmap = LinearSegmentedColormap("grad_colormap", cdict, nsteps)
    return cmap


class ARHMMAnalyzer:
    def __init__(self, model, lls, x, labels):
        self.model = model
        self.lls = lls
        self.x = x
        self.labels = labels
        self.z_hat_list = None
        self.z_hat_array = None
        self.x_array = None
        self.mean_state_df = None
        self.data_dict = None
        self.num_states = model.K
        self.agg_data = None

    def compute_most_likely_states(self):
        self.z_hat_list = [
            self.model.most_likely_states(self.x[i]) for i in range(len(self.x))
        ]
        self.z_hat_array = np.array(self.z_hat_list)
        self.x_array = np.array(self.x)
        self.x_array = self.x_array.reshape(-1, self.x_array.shape[1])

    def get_sample_data(self):
        # for plotting
        idx = 1
        sample_x = self.x_array[idx][:3000]
        sample_z = self.z_hat_array[idx][:3000]
        return sample_x, sample_z

    def get_mean_durations(self, z_hat, num_states):
        state_lst, state_dur = ssm.util.rle(z_hat)
        dur_stack = []
        for s in range(num_states):
            dur_stack.append(state_dur[state_lst == s])
        return {f"state {s}": np.mean(v) for s, v in zip(range(num_states), dur_stack)}

    def compile_mean_data(
        self,
        label,
        z_hat,
        num_states,
    ):
        data_dict = self.get_mean_durations(z_hat, num_states)
        data_dict["label"] = float(label)
        return data_dict

    def compute_aggregate_mean(self):
        agg_data = [
            self.compile_mean_data(l, z, self.num_states)
            for l, z in zip(self.labels, self.z_hat_list)
        ]
        self.agg_data = pl.concat(
            [pl.DataFrame(d) for d in agg_data], how="vertical"
        ).melt(id_vars="label", variable_name="state", value_name="mean_duration")

    def compute_metrics(self):
        self.compute_most_likely_states()
        self.compute_aggregate_mean()

    def plot_lls(self):
        fig, ax = plt.subplots()
        ax.plot(self.lls, label="EM 10 states")
        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Log Probability")
        ax.legend(loc="lower right")
        return fig

    def plot_states(self):
        sample_x, sample_z = self.get_sample_data()

        # Adjust the plot to cover the full vertical range of x_s
        y_min, y_max = sample_x.min(), sample_x.max()

        color_names = [
            "windows blue",
            "red",
            "amber",
            "faded green",
            "dusty purple",
            "orange",
            "teal",
            "coral",
            "light blue",
            "sage green",
        ]
        colors = sns.xkcd_palette(color_names)
        cmap = gradient_cmap(colors)

        fig, ax = plt.subplots()

        ax.plot(sample_x, c="k")
        ax.imshow(
            sample_z[None, :],
            cmap=cmap,
            aspect="auto",
            extent=[0, len(sample_x) - 1, y_min, y_max],
            alpha=0.3,
        )
        ax.set_ylim(y_min * 1.2, y_max * 1.2)
        return fig

    def plot_mean_state_duration(self):
        fig = sns.barplot(
            self.agg_data.to_pandas(),
            x="state",
            y="mean_duration",
            hue="label",
            errorbar="se",
        )
        return fig
