from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
        "legend.edgecolor": "black",
        "lines.markersize": 8,
        "lines.linewidth": 2.0
    }
)

AUTOMATED = np.array([2, 1, 1, 1, 1, 9, 1])
TRADITIONAL = 60 * np.array([16, 9, 10, 9, 9, 3, 2])


def cumulative_time_trajectory(combination_of_auto: List[bool]) -> List[int]:
    """
    Given a combination of using the automation tools, estimates a cumulative time
    for evaluation per catalyst. 

    Parameters
    ----------

    combination_of_auto:
        List of bools indicating whether the step in AUTOMATED is considered to be
        automated. True indicates that step is automated where False indicates
        the traditional approach is used.
    """
    cumulative_time = 0
    history = []
    for idx, a in enumerate(combination_of_auto):
        if a:
            cumulative_time += AUTOMATED[idx]
        else:
            cumulative_time += TRADITIONAL[idx]
        history.append(cumulative_time)
    return history


fig, ax = plt.subplots(figsize=(8,6))
ax.set_ylabel("Cumulative time (s)")
ax.set_xlabel("Step")

# manually specify combinations
# fully automated
ax.plot(range(1,len(AUTOMATED)+1), cumulative_time_trajectory([True] * len(AUTOMATED)), "-o", label="Fully automated")
# fully traditional
ax.plot(range(1,len(AUTOMATED)+1), cumulative_time_trajectory([False] * len(AUTOMATED)), "--o", label="Fully traditional")
# autocat but no dftparse or dftinputgen
combo = [True, False, True, True, False, False, False]
ax.plot(range(1,len(AUTOMATED)+1), cumulative_time_trajectory(combo), "-.o", label="Automated structure generation")
# no autocat but dftparse and dftinputgen
combo = [False, True, False, False, True, True, True]
ax.plot(range(1,len(AUTOMATED)+1), cumulative_time_trajectory(combo), ":o", label="Automated DFT pre- and post-processing")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), markerscale=0., fontsize=14, ncol=2, framealpha=1, borderpad=0.5)

ax.set_yscale("log")

plt.savefig("levels_of_automation.png", bbox_inches="tight", dpi=200)
