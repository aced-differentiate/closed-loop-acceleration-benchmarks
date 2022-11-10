"""
Reproduce "acceleration from sequential learning"-related figures from the associated
publication (corresponding to Figure 5).

Note that this script is provided for reproduction purposes only: the data aggregation,
averaging, and plotting functionalities implemented here contain several hard-coded,
problem-specific parameters. Please do not expect these functions to work as-is without
modifications for generating similar figures for a different problem.

The `make_figure` function encapsulates the overall task of generating Figure 5, with
functionality related to plotting individual panels delegated to
`["targets"/"distances"/"iterations"]_per_iteration_plot` functions.
A typical workflow can be found in the module execution block at the bottom.
"""

import os
import gzip
import json
import random
from typing import List
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# custom types for type hints
Array = Union[List[float], np.ndarray]

# general plotting styles-related settings
plt.style.use("seaborn-pastel")
rcParams.update(
    {
        "font.family": ["Helvetica", "sans-serif"],
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "font.size": 18,
        "axes.linewidth": 2.0,
        "lines.dashed_pattern": (5, 2.5),
    }
)


def _get_plot_options(acq: str = "mli") -> Dict:
    options = {
        "random": {"color": "#aaaaaa", "lw": 2.0, "ls": "--", "label": "Random"},
        "mli": {"color": "C0", "lw": 2.5, "ls": "-", "label": "SL (MLI)"},
        "mu": {"color": "C2", "ls": "-", "lw": 2.5, "label": "SL (MU)"},
    }
    return options[acq]


def _get_average_candidate_distance_to_window(
    df: pd.DataFrame,
    train_mask: List[bool],
    /,
    target_column: str = "binding_energy_of_adsorbed",
    target_window: List[float] = None,
    *,
    avg: str = "mean",
) -> float:
    distances = []
    # distance to lower bound of window
    lb_distances = np.array(df[train_mask][target_column] - target_window[0])
    lb_distances[lb_distances > 0] = 0
    # distance to upper bound of window
    ub_distances = np.array(df[train_mask][target_column] - target_window[1])
    ub_distances[ub_distances < 0] = 0
    distances = np.maximum(np.abs(lb_distances), np.abs(ub_distances))

    if avg == "mean":
        return np.mean(distances)
    elif avg == "median":
        return np.median(distances)
    else:
        msg = f'Failed to interpret averaging method "{avg}"'
        raise NotImplementedError(msg)


def _get_number_of_candidates_in_window(
    df: pd.DataFrame,
    train_mask: List[bool],
    /,
    target_column: str = "binding_energy_of_adsorbed",
    target_window: List[float] = None,
) -> int:
    _df = df[train_mask]
    ciw = _df[
        (_df[target_column] >= target_window[0])
        & (_df[target_column] <= target_window[1])
    ]
    return len(ciw)


def _aggregate_per_trial_candidates_data(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_of_adsorbed",
    target_window: List[float] = None,
    *,
    metric: str = "successes",
    n_iterations: int = 1,
) -> List:
    per_trial_data = []
    for trial in histories:
        data = []
        for train_mask in histories[trial]["train_history"][:n_iterations]:
            if metric == "successes":
                data.append(
                    _get_number_of_candidates_in_window(
                        df, train_mask, target_column, target_window,
                    )
                )
            elif metric == "distances":
                data.append(
                    _get_average_candidate_distance_to_window(
                        df, train_mask, target_column, target_window,
                    )
                )
            else:
                raise NotImplementedError
        per_trial_data.append(data)
    return per_trial_data


def _aggregate_per_trial_accuracy_data(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_of_adsorbed",
    *,
    bootstrap_size: int = 100,
    n_bootstrap_samples: int = 20,
    n_iterations: int = 1,
) -> pd.DataFrame:
    y = np.array(df[target_column].values)
    records = []
    for trial in histories:
        for iteration, (train_mask, pred) in enumerate(
            zip(
                histories[trial]["train_history"][: n_iterations + 1],
                histories[trial]["pred_history"][: n_iterations + 1],
            )
        ):
            pred = np.array(pred)
            test_mask = [not idx for idx in train_mask]
            for bs_sample in range(n_bootstrap_samples):
                bs_idx = np.random.choice(
                    np.arange(len(test_mask))[test_mask], bootstrap_size, replace=True,
                )
                bs_mask = np.zeros(len(test_mask), dtype=bool)
                bs_mask[bs_idx] = True
                records.append(
                    {
                        "trial": trial,
                        "iteration": iteration,
                        "bs_sample": bs_sample,
                        "mae": np.mean(np.abs(y[bs_mask] - pred[bs_mask])),
                    }
                )
    acc_df = pd.DataFrame(records)
    return acc_df


def per_iteration_plot(
    per_trial_data: Array,
    /,
    ax: Axes = None,
    *,
    color: str = "C0",
    ls: str = "-",
    lw: float = 2.0,
):
    """
    Plot the average statistic (e.g., # of candidates found, model accuracy) as a
    function of SL iteration as a line, and color an area around that line to indicate
    uncertainty (calculated as +/- std over independent SL trials).

    Args:
        per_trial_data (Array): Target statistics aggregated over trials and SL
            iterations. Usually of shape (# of trials, # of SL iterations).
        ax (Axes, optional): Matplotlib axes on which to plot. If not specified, a new
            figure and axes will be created.
        color (str, optional): Color of the statistic line (and the surrounding colored
            area of uncertainty). Defaults to "C0".
        ls (str, optional): Line style (dotted, dashed, etc.). Defaults to "-".
        lw (float, optional): Line width. Defaults to 2.0.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    mean = np.mean(per_trial_data, axis=0)
    unct = np.std(per_trial_data, axis=0)

    ax.plot(range(len(mean)), mean, color=color, ls=ls, lw=lw)
    ax.fill_between(
        range(len(mean)), mean + unct, mean - unct, facecolor=color, alpha=0.3,
    )


def targets_per_iteration_plot(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_of_adsorbed",
    target_window: List[float] = None,
    *,
    n_iterations: int = None,
    ax: Axes = None,
):
    """
    Plot the average number of candidates found in the target window as a function of SL
    iteration, and color an area around the average to indicate uncertainty (calculated
    as +/- std over independent SL trials). Compare the performance of maximum
    likelihood of improvement (MLI) and random acquisition functions.

    Args:
        df (pd.DataFrame): The full dataset.
        histories (Dict): A record of training masks, predictions, and prediction
            uncertainties made at every iteration of sequential learning (SL) during
            every independent trial, for each input acquisition function.
            See the docstring for the `make_figure` function for details.
        target_column (str, optional): Name of the column with the target property that
            was optimized over. Defaults to "binding_energy_of_adsorbed".
        target_window (List[float], optional): Window [lower limit, upper limit] of the
            target property that was used to rank and select candidates during SL.
            Defaults to [-0.7, -0.5] (the window that was used in the accompanying SL
            script).
        n_iterations (int, optional): Number of SL iterations to plot. Defaults to all
            SL iterations present in the input `histories` data.
        ax (Axes, optional): Matplotlib axes on which to plot. If not specified, a new
            figure and axes will be created.
    """
    print("Plotting number of targets found per SL iteration...")

    if n_iterations is None:
        acq = random.choice(list(histories.keys()))
        n_iterations = len(histories[acq]["1"]["train_history"])
    print(f"  Number of frames (SL iterations): {n_iterations}")

    n_targets = _get_number_of_candidates_in_window(
        df,
        np.ones(len(df), dtype=bool),
        target_column=target_column,
        target_window=target_window,
    )
    print(f"  Total number of candidates in the target window: {n_targets}")

    for acq in histories:
        # this plot does not use results using MU acquisition
        if acq not in ["random", "mli"]:
            continue
        per_trial_data = _aggregate_per_trial_candidates_data(
            df,
            histories[acq],
            target_column=target_column,
            target_window=target_window,
            metric="successes",
            n_iterations=n_iterations,
        )
        per_iteration_plot(
            per_trial_data,
            ax=ax,
            color=_get_plot_options(acq)["color"],
            ls=_get_plot_options(acq)["ls"],
            lw=_get_plot_options(acq)["lw"],
        )

    # annotate the total number of targets in the dataset
    ax.axhline(y=n_targets, ls="--", color="k")
    ax.text(
        n_iterations * 0.4,
        n_targets + 2,
        f"Total # of targets = {n_targets}",
        color="k",
        ha="center",
        va="center",
    )

    ax.set_xlim([0, n_iterations - 1])
    ax.set_ylim([0, int(np.ceil(0.1 * n_targets)) * 10])
    ax.set_xlabel("Sequential learning iteration")
    ax.set_ylabel("Number of target compounds found")
    print("Done.\n")


def distances_per_iteration_plot(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_of_adsorbed",
    target_window: List[float] = None,
    *,
    n_iterations: int = None,
    ax: Axes = None,
):
    """
    Plot the average candidate distance to the target window as a function of SL
    iteration, and color an area around the average to indicate uncertainty (calculated
    as +/- std over independent SL trials). Compare the performance of maximum
    likelihood of improvement (MLI) and random acquisition functions.

    Args:
        df (pd.DataFrame): The full dataset.
        histories (Dict): A record of training masks, predictions, and prediction
            uncertainties made at every iteration of sequential learning (SL) during
            every independent trial, for each input acquisition function.
            See the docstring for the `make_figure` function for details.
        target_column (str, optional): Name of the column with the target property that
            was optimized over. Defaults to "binding_energy_of_adsorbed".
        target_window (List[float], optional): Window [lower limit, upper limit] of the
            target property that was used to rank and select candidates during SL.
            Defaults to [-0.7, -0.5] (the window that was used in the accompanying SL
            script).
        n_iterations (int, optional): Number of SL iterations to plot. Defaults to all
            SL iterations present in the input `histories` data.
        ax (Axes, optional): Matplotlib axes on which to plot. If not specified, a new
            figure and axes will be created.
    """
    print("Plotting candidate distance to target window per SL iteration...")

    if n_iterations is None:
        acq = random.choice(list(histories.keys()))
        n_iterations = len(histories[acq]["1"]["train_history"])
    print(f"  Number of frames (SL iterations): {n_iterations}")

    for acq in histories:
        # this plot does not use results using MU acquisition
        if acq not in ["random", "mli"]:
            continue
        per_trial_data = _aggregate_per_trial_candidates_data(
            df,
            histories[acq],
            target_column=target_column,
            target_window=target_window,
            metric="distances",
            n_iterations=n_iterations,
        )
        per_iteration_plot(
            per_trial_data,
            ax=ax,
            color=_get_plot_options(acq)["color"],
            ls=_get_plot_options(acq)["ls"],
            lw=_get_plot_options(acq)["lw"],
        )

    ax.set_xlim([0, n_iterations - 1])
    ax.set_xlabel("Sequential learning iteration")
    ax.set_ylabel("Candidate distance to target window (eV)")
    print("Done.\n")


def accuracy_per_iteration_plot(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_of_adsorbed",
    *,
    init_train_size: int = 10,
    n_iterations: int = None,
    ax: Axes = None,
):
    """
    Plot model accuracy as a function of SL iteration for the maximum uncertainty (MU)
    acquisition function, and color an area around the average to indicate uncertainty
    (calculated as +/- std over independent SL trials).

    Args:
        df (pd.DataFrame): The full dataset.
        histories (Dict): A record of training masks, predictions, and prediction
            uncertainties made at every iteration of sequential learning (SL) during
            every independent trial, for each input acquisition function.
            See the docstring for the `make_figure` function for details.
        target_column (str, optional): Name of the column with the target property that
            was optimized over. Defaults to "binding_energy_of_adsorbed".
        init_train_size (int, optional): Number of examples used to build the initial ML
            models during each SL run. Defaults to 10, consistent with the accompanying
            sequential learning script.
        n_iterations (int, optional): Number of SL iterations to plot. Defaults to all
            SL iterations present in the input `histories` data.
        ax (Axes, optional): Matplotlib axes on which to plot. If not specified, a new
            figure and axes will be created.
    """
    print("Plotting ML model accuracy per SL iteration...")

    if n_iterations is None:
        n_iterations = len(histories["mu"]["1"]["train_history"])
    print(f"  Number of frames (SL iterations): {n_iterations}")

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    # aggregate data and calculate stats to plot
    _df = _aggregate_per_trial_accuracy_data(
        df, histories["mu"], target_column=target_column, n_iterations=n_iterations
    )
    iterations = sorted(list(set(_df["iteration"].values)))
    mean = np.array([_df[_df["iteration"] == i]["mae"].mean() for i in iterations])
    unct = np.array([_df[_df["iteration"] == i]["mae"].std() for i in iterations])
    size = init_train_size + np.array(range(len(mean)))

    ax.plot(
        size,
        mean,
        color=_get_plot_options("mu")["color"],
        ls=_get_plot_options("mu")["ls"],
        lw=_get_plot_options("mu")["lw"],
    )
    ax.fill_between(
        size,
        mean + unct,
        mean - unct,
        alpha=0.3,
        color=_get_plot_options("mu")["color"],
    )

    # annotate example target accuracy
    target_mae = 0.1
    ax.axhline(y=target_mae, ls="--", color="k")
    ax.text(
        init_train_size + (n_iterations * 0.5),
        target_mae - 0.025,
        f"Target accuracy = {target_mae} eV",
        ha="center",
        va="center",
        color="k",
    )

    ax.set_xlim([init_train_size, n_iterations + init_train_size])
    ax.set_ylim([0, max(mean) * 1.05])
    ax.set_xlabel("Size of training data")
    ax.set_ylabel("Model test accuracy (MAE in eV)")
    print("Done.\n")


def make_figure(
    df: pd.DataFrame,
    histories: Dict,
    /,
    target_column: str = "binding_energy_per_adsorbed",
    target_window: List[float] = None,
    *,
    filename: str = "acceleration_from_sequential_learning.png",
):
    """
    Reproduce the 3-panel Figure 5 (acceleration from using sequential learning) from
    the associated publication. Panels:
    - a: number of candidates found in the target window as f(SL iteration)
    - b: candidate distance to the target window as f(SL iteration)
    - c: model accuracy as f(SL iteration)
    Add labels to each panel and a legend for the overall plot.

    Args:
        df (pd.DataFrame): The full dataset.
        histories (Dict): A record of training masks, predictions, and prediction
            uncertainties made at every iteration of sequential learning (SL) during
            every independent trial, for each input acquisition function.

            For each iteration of SL, the corresponding lists are of length = number of
            examples/rows in the full dataset. The training masks are boolean values
            indicating whether a particular index was used for training or not, while
            the predictions and prediction uncertainties are floating point numbers.

            An example for a (very much hypothetical) dataset with 4 examples, 2 SL
            iterations, 2 examples used to train the initial models, 10 independent
            trials and acquisition functions ["random", "mli"]:

            {
                "random": {
                    1: {
                        "train_history": [[True, False, False, True], [True, True, False, True]],
                        "pred_history": [[0.02, 0.01, -0.45, -0.36], [0.12, -0.22, -0.26, -0.36]],
                        "unct_history": [[0.08, 0.34, 0.12, 0.20], [0.01, 0.02, 0.32, 0.05]]
                    },
                    2: {
                        ...
                    },
                    ...,
                    ...,
                    10: {
                        ...
                    }
                },
                "mli": {
                    1: {
                        "train_history": [[True, False, False, True], [True, False, True, True]],
                        "pred_history": [[0.02, 0.01, -0.45, -0.36], [0.10, -0.32, -0.46, -0.36]],
                        "unct_history": [[0.08, 0.34, 0.12, 0.20], [0.01, 0.05, 0.02, 0.05]]
                    },
                    2: {
                        ...
                    },
                    ...,
                    ...,
                    10: {
                        ...
                    }
                }
            }

            Note that the output of the accompanying `sequential_learning.py` script is
            already in the above format.
        target_column (str, optional): Name of the column with the target property that
            was optimized over. Defaults to "binding_energy_of_adsorbed".
        target_window (List[float], optional): Window [lower limit, upper limit] of the
            target property that was used to rank and select candidates during SL.
            Defaults to [-0.7, -0.5] (the window that was used in the accompanying SL
            script).
        filename (str, optional): Path to the file to which the plot has to be saved.
            Defaults to "./acceleration_from_sequential_learning.png".
    """
    # 3-panel plot corresponding to Figure 5 in the associated publication
    fig = plt.figure(figsize=(21, 6))
    gs = GridSpec(nrows=1, ncols=3, wspace=0.225)

    # panel a: number of candidates found in window
    ax = fig.add_subplot(gs[0])
    targets_per_iteration_plot(
        df, histories, target_column=target_column, target_window=target_window, ax=ax
    )
    ax.text(0.875, 0.875, "a", fontsize=44, fontweight=1000, transform=ax.transAxes)

    # panel b: candidate distance to window
    ax = fig.add_subplot(gs[1])
    distances_per_iteration_plot(
        df, histories, target_column=target_column, target_window=target_window, ax=ax
    )
    ax.text(0.875, 0.875, "b", fontsize=44, fontweight=1000, transform=ax.transAxes)

    # panel c: ML model accuracy
    ax = fig.add_subplot(gs[2])
    accuracy_per_iteration_plot(
        df, histories, target_column=target_column, n_iterations=70, ax=ax
    )
    ax.text(0.875, 0.875, "c", fontsize=44, fontweight=1000, transform=ax.transAxes)

    # plot the consolidated legend
    print("Adding legend...")
    handles = []
    for acq in ["mli", "random", "mu"]:
        handle = Line2D(
            [],
            [],
            color=_get_plot_options(acq)["color"],
            ls=_get_plot_options(acq)["ls"],
            lw=_get_plot_options(acq)["lw"],
            label=_get_plot_options(acq)["label"],
        )
        handles.append(handle)
    ax.legend(handles=handles, ncol=3, bbox_to_anchor=(0.10, -0.15), fontsize=24)
    print("Done.\n")

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f'The figure has been saved to "{filename}".')


if __name__ == "__main__":
    # directory with the bimetallic catalysts data
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(thisdir, "..", "data", "bimetallic_catalysts_dataset")

    # read in the bimetallics adsorption energy dataset
    csv_path = os.path.join(datadir, "bimetallics_data.csv")
    df = pd.read_csv(csv_path)

    # read in the training/prediction history
    with gzip.open("histories.json.gz", "rt", encoding="utf-8") as fr:
        histories = json.load(fr)

    # property and target window to reproduce figures in the associated publication:
    # column in the dataset with the property of interest
    target_column = "binding_energy_of_adsorbed"
    # window of property values that was optimized over in the SL runs
    target_window = [-0.7, -0.5]

    # make the 3-panel figure and write it to disk
    make_figure(df, histories, target_column=target_column, target_window=target_window)
