import numpy as np
from alphacsc import learn_d_z, BatchCDL
from alphacsc.utils import construct_X_multi, check_univariate_signal
import matplotlib.pyplot as plt
import random


def calculate_mse_list(x, x_hat):
    """
    Calculates MSE for each trial between original and reconstructed signals.

    Parameters:
    - x (np.ndarray): Original signals of shape (n_trials, n_times).
    - reconstructions (np.ndarray): Reconstructed signals of shape (n_trials, n_times).

    Returns:
    - mse_list (list): List of MSE values for each trial.
    """
    return [np.mean((x[trial] - x_hat[trial]) ** 2) for trial in range(x.shape[0])]


class CDLTrainer:
    def __init__(
        self,
        n_atoms,
        n_times_atom,
        n_iter,
        reg,
        n_jobs,
        solver_d_kwargs,
        random_state,
        verbose,
    ):
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.n_iter = n_iter
        self.reg = reg
        self.n_jobs = n_jobs
        self.solver_d_kwargs = solver_d_kwargs
        self.random_state = random_state
        self.verbose = verbose
        self.pobjective = None
        self.times = None
        self.d_hat = None
        self.z_hat = None
        self._cdl = None
        self.n_channels = 1

    @property
    def cdl_params(self):
        params = {
            "n_times_atom": self.n_times_atom,
            "n_atoms": self.n_atoms,
            "n_iter": self.n_iter,
            "reg": self.reg,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "solver_d_kwargs": self.solver_d_kwargs,
        }
        return params

    def fit_csc(self, X):
        self.pobjective, self.times, self.d_hat, self.z_hat, self.reg = learn_d_z(
            X, **self.cdl_params
        )

    @property
    def cdl(self):
        if self._cdl is None:
            self.build_batch_cdl()
        return self._cdl

    def build_batch_cdl(self):
        params = {k: v for k, v in self.cdl_params.items() if k != "solver_kwgs"}
        self._cdl = BatchCDL(**params)
        self._cdl._D_hat = self.d_hat
        self._cdl.reg_ = self.reg
        self._cdl.n_channels_ = self.n_channels

    def transform(self, X):

        if len(X.shape) != 3 or X.shape[1] != 1:
            X = check_univariate_signal(X)
        return self.cdl.transform(X)

    def reconstruct(self, z_hat):
        return construct_X_multi(z_hat, self.d_hat, n_channels=1)

    def transform_and_reconstruct(self, X):

        z_hat = self.transform(X)
        return self.reconstruct(z_hat)


class CDLAnalyzer:
    def __init__(self, trainer, loader):
        self.trainer = trainer
        self.loader = loader
        self.mse_list = None

    def compute_mse(self, x, x_hat):
        self.mse_list = calculate_mse_list(x, x_hat)
        return np.mean(self.mse_list)

    def plot_pobjective(self):
        plt.figure(figsize=(5, 3))
        plt.plot(self.trainer.pobjective)
        plt.title("Objective Function")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.show()

    def _get_subplot_arrangement(self, n_atoms):
        """
        Determines the best subplot arrangement given the number of atoms.

        Parameters:
        - n_atoms (int): Number of atoms to plot.

        Returns:
        - n_rows (int): Number of rows for subplots.
        - n_cols (int): Number of columns for subplots.
        """
        if n_atoms == 1:
            return 1, 1

        n_cols = int(np.ceil(np.sqrt(n_atoms)))
        n_rows = int(np.ceil(n_atoms / n_cols))

        return n_rows, n_cols

    def plot_atoms(self, d_hat):
        """
        Plots the atoms learned by the CDL model.

        Parameters:
        - d_hat (np.ndarray): Learned atoms of shape (n_atoms, n_channels, n_times_atom).
        """
        n_atoms, _ = d_hat.shape

        # Determine the best subplot arrangement
        n_rows, n_cols = self._get_subplot_arrangement(n_atoms)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex=True, sharey=True
        )
        axes = axes.flatten() if n_atoms > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < n_atoms:
                ax.plot(d_hat[i])
                ax.set_title(f"Atom {i+1}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")
            else:
                ax.axis("off")  # Hide unused subplots

        plt.tight_layout()
        plt.show()

    def plot_mse_distribution(self):
        plt.figure(figsize=(5, 3))
        plt.hist(self.mse_list, bins=20, edgecolor="black")
        plt.title("Distribution of MSE across trials")
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.show()

    def mse_boxplot(self):
        plt.figure(figsize=(5, 3))
        plt.boxplot(self.mse_list)
        plt.title("MSE")
        plt.ylabel("MSE")
        plt.show()

    def plot_mse_by_trial(self):
        plt.figure(figsize=(6, 3))
        plt.plot(self.mse_list, marker="o")
        plt.title("MSE for Each Trial")
        plt.xlabel("Trial Number")
        plt.ylabel("MSE")
        plt.show()

    def plot_best_and_worst_reconstructions(self, X, X_hat):
        # Ensure X and X_hat are 2D
        if X.ndim != 2:
            X = X.reshape(1, -1)
        if X_hat.ndim != 2:
            X_hat = X_hat.reshape(X.shape[0], -1)
        # Find the trial with the minimum and maximum MSE
        min_mse_trial = np.argmin(self.mse_list)
        max_mse_trial = np.argmax(self.mse_list)

        # Get the original signal and its reconstruction for these trials
        x_best = X[min_mse_trial]
        x_best_hat = X_hat[min_mse_trial]
        x_worst = X[max_mse_trial]
        x_worst_hat = X_hat[max_mse_trial]

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))

        # Plot original signal
        axes[0].plot(x_best)
        axes[0].plot(x_best_hat)
        axes[0].set_title("Original Signal (Best)")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Amplitude")

        # Plot reconstructed signal
        axes[1].plot(x_worst)
        axes[1].plot(x_worst_hat)
        axes[1].set_title("Original Signal (Worst)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    def plot_activation_by_trial(self, x, z_hat):
        # get random idx to plot
        random_idx = random.randint(0, len(x) - 1)
        x = x[random_idx]
        z_hat = z_hat[random_idx]

        n_atoms, _ = z_hat.shape

        fig, axes = plt.subplots(
            n_atoms + 1, 1, figsize=(4, 1 * (n_atoms + 1)), sharex=True
        )
        # Plot original signal
        axes[0].plot(x, color="black")
        axes[0].set_title("Original Signal")
        axes[0].set_ylabel("Amplitude")

        # Plot activation strengths for each atom
        # Generate a list of random colors for each atom

        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(n_atoms)
        ]
        for atom in range(n_atoms):
            axes[atom + 1].plot(z_hat[atom, :], color=colors[atom])
            axes[atom + 1].set_title(f"Atom {atom + 1} Activation")
            axes[atom + 1].set_ylabel("Activation")

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()
