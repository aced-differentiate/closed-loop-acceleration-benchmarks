"""Estimate human lag in job management using Monte Carlo sampling."""

from typing import List

import numpy as np
from numpy import random


def get_job_lag(checkpoints: List[int], /, n_jobs: int = 10 ** 6) -> float:
    """
    Run a Monte Carlo sampling to estimate mean lag per job.

    Args:
        checkpoints (List[int]): Job checkpoints (i.e., times in a week when a
            job is checked on) as an offset from Monday 8am.
            E.g., [2, 4, 24, 26, 48, 54] corresponds to checking job status at Monday
            10am, Monday 12pm, Tuesday 8am, Tuesday 10am, Wednesday 8am, and Wednesday
            2pm, respectively.
            Note that the time from which the offset is calculated (Monday 8am) is
            immaterial/arbitrary but this particular implementation relies on it for
            calculating the average time between when a job finishes and when it is
            checked on.
        n_jobs (int, optional): Number of jobs (or MC trials). Defaults to 10**6.

    Returns:
        mean_lag (float): The mean lag per job in hours.
    """
    rng = random.default_rng(42)
    # assume that a randomly-chosen job can complete at any random time during the week
    job_stop_times = rng.uniform(0, 24 * 7, n_jobs)
    time_to_checkpoints = checkpoints - job_stop_times
    # ignore checkpoints in the past
    time_to_checkpoints[time_to_checkpoints < 0] = np.infty
    time_to_nearest_checkpoint = np.min(time_to_checkpoints, axis=0)
    mean_lag = np.mean(time_to_nearest_checkpoint, axis=0)
    return mean_lag


if __name__ == "__main__":
    # checkpoints = points in time when a researcher is expected to check on the status
    # of ongoing jobs/calculations
    # list of job checkpoints to reproduce the data in the associated publication:
    checkpoints = np.array(
        [
            2,  # Monday (start at 8am)
            4,
            6,
            8,
            14,
            24,  # Tuesday
            26,
            28,
            30,
            32,
            38,
            48,  # Wednesday
            50,
            52,
            54,
            56,
            62,
            72,  # Thursday
            74,
            76,
            78,
            80,
            86,
            96,  # Friday (NB: end at 11pm)
            98,
            100,
            102,
            104,
            110,
            120,  # Saturday (8am)
            168,  # Monday (8am)
        ]
    )
    checkpoints = checkpoints.reshape((-1, 1))
    mean_lag = get_job_lag(checkpoints=checkpoints)
    print(f"Mean lag/job (in hours): {mean_lag:0.2f}")
