# This file contains the main functions used to compare and visualize ensembles and their scores.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from itertools import combinations
import json
import math

from fetch import *

# Code to fetch the score array for a given state, chamber, ensemble_type and score.
# and also creates the lists of ensemble types, score, and state-chamber combinations.

state_chamber_list = [
    (state, chamber)
    for state in state_list
    for chamber in ["congress", "upper", "lower"]
]

base_list = ["base0", "base1", "base2", "base3", "base4"]
ensemble_list = [
    "pop_minus",
    "pop_plus",
    "distpair",
    "ust",
    "distpair_ust",
    "reversible",
    "county25",
    "county50",
    "county75",
    "county100",
]

# convert into LaTex notation for tables in our paper
ensemble_name_dict = {
    "base0": "$\RA_0$",
    "base1": "$\RA_1$",
    "base2": "$\RA_2$",
    "base3": "$\RA_3$",
    "base4": "$\RA_4$",
    "pop_minus": "$\popm$",
    "pop_plus": "$\popp$",
    "ust": "$\RC$",
    "distpair": "$\RB$",
    "distpair_ust": "$\RD$",
    "reversible": r"\makecell{Rev \\ ReCom}",
    "county25": "$\C$",
    "county50": "$\CC$",
    "county75": "$\CCC$",
    "county100": "$\CCCC$",
}

# convert to names compatable with Matplotlib
ensemble_name_dict_for_plots = {
    "base0": r"$A_0$",
    "base1": r"$A_1$",
    "base2": r"$A_2$",
    "base3": r"$A_3$",
    "base4": r"$A_4$",
    "pop_minus": r"$Pop_{-}$",
    "pop_plus": r"$Pop_{+}$",
    "ust": r"$C$",
    "distpair": r"$B$",
    "distpair_ust": r"$D$",
    "reversible": r"RevReCom",
    "county25": r"$R_{25}$",
    "county50": r"$R_{50}$",
    "county75": r"$R_{75}$",
    "county100": r"$R_{100}$",
}

# statistical tests


def t_test(
    a0, a1
):  # runs the t-test of the hypotheses that two arrays were drawn from distributions with the same means.
    result = stats.ttest_ind(a0, a1, equal_var=False)
    return (
        result.statistic,
        result.pvalue,
    )  # the statistic is positive if a0 has a larger mean than a1


def ks_test(
    a0, a1
):  # runs the Kolmogorov-Smirnov test that the two arrays were drawn from the same distribution
    result = stats.ks_2samp(a0, a1)
    return (
        result.statistic,
        result.pvalue,
        result.statistic_sign,
    )  # the statistic_sign is positive if a1 has larger values than a0


def gelman_rubin_rhat(a1, a2):
    n = len(a1)
    assert len(a2) == n, "Both chains must have the same length"

    # Means and variances
    mu1, mu2 = np.mean(a1), np.mean(a2)
    s1_sq, s2_sq = np.var(a1, ddof=1), np.var(a2, ddof=1)

    W = (s1_sq + s2_sq) / 2
    B = n * ((mu1 - mu2) ** 2) / 2
    V_hat = ((n - 1) / n) * W + (1 / n) * B
    R_hat = np.sqrt(V_hat / W)

    return R_hat


# visualization functions


def kde_plot(
    state, chamber, ensemble_list, score, average_lines=True, filename=None, ax=None
):
    """
    For the given state, chamber, and score, this plots one KDE for each ensemble in ensemble_list.
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ensemble in enumerate(ensemble_list):
        color = prop_cycle[i % len(prop_cycle)]
        a = fetch_score_array(state, chamber, ensemble, score)
        sns.kdeplot(a, label=ensemble_name_dict_for_plots[ensemble], color=color, ax=ax)
        if average_lines:
            ax.axvline(np.mean(a), linestyle="--", color=color)

    ax.set_title(f"{state} {chamber}: {score}")
    ax.set_xlabel(score)
    ax.set_ylabel("Density")
    ax.legend(title="Ensemble")

    if filename is not None:
        plt.savefig(filename)

    if created_ax:
        plt.show()


def box_whisker_plot(
    state, chamber, ensemble_list, score, filename=None
):  # box plot of any given list of ensembles
    """
    For the given state, chamber, and score, this plots one box plot for each ensembles in ensemble_list.
    """
    data = pd.DataFrame(columns=["ensemble", "score"])
    for ensemble in ensemble_list:
        a = fetch_score_array(state, chamber, ensemble, score)
        data_for_a = pd.DataFrame({"ensemble": [ensemble] * len(a), "score": a})
        data = pd.concat([data, data_for_a], ignore_index=True)
    ax = sns.boxplot(data=data, x="ensemble", y="score", hue="ensemble", fliersize=0)
    ax.set_title(f"{state} {chamber}: {score} boxplots by ensemble type")
    ax.set_xlabel("Ensemble")
    ax.set_ylabel(score)
    # Set LaTeX labels for each tick on the x-axis
    ax.set_xticklabels(
        [ensemble_name_dict_for_plots[ensemble] for ensemble in ensemble_list],
        rotation=45,
    )
    ax.grid(axis="y")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    plt.show()


def box_whisker_plots_grid(
    state, chamber, ensemble_list, score_list, filename=None, cols=2
):
    """
    For the given state and chamber, this plots one box plot per score across all ensembles.
    Plots are arranged in a grid with a configurable number of columns.

    Parameters:
        state (str): The state name.
        chamber (str): The chamber name.
        ensemble_list (list of str): List of ensemble names.
        score_list (list of str): List of scores to plot.
        filename (str, optional): If provided, saves the plot to this file.
        cols (int): Number of columns in the subplot grid.
    """
    num_scores = len(score_list)
    rows = math.ceil(num_scores / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = (
        axes.flatten() if num_scores > 1 else [axes]
    )  # Make sure axes is always iterable

    for idx, score in enumerate(score_list):
        data = pd.DataFrame(columns=["ensemble", "score"])
        for ensemble in ensemble_list:
            a = fetch_score_array(state, chamber, ensemble, score)
            data_for_a = pd.DataFrame({"ensemble": [ensemble] * len(a), "score": a})
            data = pd.concat([data, data_for_a], ignore_index=True)

        ax = axes[idx]
        sns.boxplot(
            data=data, x="ensemble", y="score", hue="ensemble", fliersize=0, ax=ax
        )
        ax.set_title(f"{score}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticklabels(
            [ensemble_name_dict_for_plots[ensemble] for ensemble in ensemble_list],
            rotation=45,
        )
        ax.grid(axis="y")

    # Hide any unused axes
    for j in range(len(score_list), len(axes)):
        fig.delaxes(axes[j])

    # Add a single centered title for the whole grid
    # plt.tight_layout()
    fig.suptitle(f"{state} {chamber}: Boxplots of scores by Ensemble Type", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

    if filename is not None:
        plt.savefig(filename)
    plt.show()


def ordered_seats_plot(
    state, chamber, ensemble_list, competitive_window=0.05, filename=None
):
    """
    For the given state and chamber, this superimposes ordered-seats-plots for the two ensembles in ensemble_list.
    Each ensemble is colored differently.
    The size of ensemble_list must be 2.
    It only includes seats that are competitive for at least one of the ensembles.
    (this means that the dem seat share is within competitive_window of 0.5 for at least one ensemble)
    """
    if len(ensemble_list) != 2:
        raise ValueError("ensemble_list must have length 2")
    X0 = fetch_score_array(state, chamber, ensemble_list[0], "by_district")
    X1 = fetch_score_array(state, chamber, ensemble_list[1], "by_district")

    seats_list = []
    for i in range(1, X0.shape[1] + 1):
        if (
            abs(np.mean(X0[:, i - 1]) - 0.5) < competitive_window
            or abs(np.mean(X1[:, i - 1]) - 0.5) < competitive_window
        ):
            seats_list.append(i)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in seats_list:
        ax.boxplot(
            X0[:, i - 1],
            positions=[i - 0.15],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="black"),
            flierprops=dict(markerfacecolor="white", marker=""),
        )
        ax.boxplot(
            X1[:, i - 1],
            positions=[i + 0.15],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="lightgreen", color="black"),
            medianprops=dict(color="black"),
            flierprops=dict(markerfacecolor="white", marker=""),
        )
    plt.xticks(np.arange(1, X0.shape[1] + 1), np.arange(1, X0.shape[1] + 1))
    plt.axhline(y=0.5, color="red", linestyle="-")
    if competitive_window == 1:
        plt.axhline(y=0.45, color="red", linestyle="--")
        plt.axhline(y=0.55, color="red", linestyle="--")
    else:
        plt.axhline(y=0.5 - competitive_window, color="red", linestyle="--")
        plt.axhline(y=0.5 + competitive_window, color="red", linestyle="--")
    plt.xlabel("Ordered Districts")
    plt.ylabel("Democrat Vote Share")
    plt.title(
        f"{state} {chamber}: Ordered Seats Plots for {ensemble_name_dict_for_plots[ensemble_list[0]]} and {ensemble_name_dict_for_plots[ensemble_list[1]]}"
    )
    plt.legend(
        [
            plt.Line2D([0], [0], color="lightblue", lw=4),
            plt.Line2D([0], [0], color="lightgreen", lw=4),
        ],
        [
            ensemble_name_dict_for_plots[ensemble_list[0]],
            ensemble_name_dict_for_plots[ensemble_list[1]],
        ],
        loc="upper left",
    )
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def kde_jointplot(
    state,
    chamber,
    score1,
    score2,
    my_ensemble_list=ensemble_list,
    filename=None,
    step_size=1,
):
    """
    Returns a KDE plot for the scores score1 and score2 over the ensembles in my_ensemble_list.
    Increase step_size to use a subsample and hense speed up the plot.
    """
    # Build the dataframe
    all_rows = []
    for ensemble in my_ensemble_list:
        score_arrays = {
            score: fetch_score_array(state, chamber, ensemble, score)[::step_size]
            for score in [score1, score2]
        }
        num_plans = len(next(iter(score_arrays.values())))
        for i in range(num_plans):
            row = [score_arrays[score][i] for score in [score1, score2]] + [ensemble]
            all_rows.append(row)
    df = pd.DataFrame(all_rows, columns=[score1, score2, "ensemble"])

    # Create the KDE plot
    ax = sns.kdeplot(df, x=score1, y=score2, hue="ensemble")
    plt.title(f"KDE plot of {score1} vs {score2} for {state} {chamber} ensembles")

    # Fix the legend labels by modifying the existing legend
    if ax.legend_:
        legend = ax.legend_
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        new_labels = [
            ensemble_name_dict_for_plots.get(label, label) for label in labels
        ]

        ax.legend(
            handles=handles, labels=new_labels, title="Ensemble", loc="lower right"
        )

    if filename:
        plt.savefig(filename)
    plt.show()


# Correlation table function
def correlation_table(
    state,
    chamber,
    my_ensemble_list=ensemble_list,
    my_score_list=primary_score_list,
    step_size=1,
    rounding=None,
    return_dataframe=False,
):
    """
    Returns a correlation table for the scores in my_score_list over the ensembles in my_ensemble_list.
    Set step_size to 1 to use all the plans, or a larger number to subsample the data (to use less memory).
    Optionally returns the dataframe used to create the correlation table.
    """
    all_rows = []

    for ensemble in my_ensemble_list:
        score_arrays = {
            score: fetch_score_array(state, chamber, ensemble, score)[::step_size]
            for score in my_score_list
        }
        num_plans = len(next(iter(score_arrays.values())))
        for i in range(num_plans):
            row = [score_arrays[score][i] for score in my_score_list] + [ensemble]
            all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=my_score_list + ["ensemble"])

    corr = df.corr(numeric_only=True)
    if rounding is not None:
        corr = corr.round(rounding)

    return (corr, df) if return_dataframe else corr
