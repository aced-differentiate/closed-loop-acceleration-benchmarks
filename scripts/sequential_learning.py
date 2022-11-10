"""Perform simulated sequential learning on the bimetallic catalysts dataset."""

import os
import json
import gzip
from typing import List
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from lolopy.learners import RandomForestRegressor

# custom type hints
Array = Union[List[float], np.ndarray]


def _df_to_magpie_features(df: pd.DataFrame, column: str = "formula") -> pd.DataFrame:
    # get pymatgen composition objects from a string formula or dictionaries
    df = StrToComposition(target_col_id="pmg_comp").featurize_dataframe(
        df=df, col_id=column
    )
    # convert composition into magpie features
    featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie")])
    df = featurizer.featurize_dataframe(df=df, col_id="pmg_comp")
    # remove the pmg_composition (extraneous) column
    df = df[[c for c in df.columns if c not in ["pmg_comp"]]]
    print(f"Featurized dataframe:\n{df}")
    return df


def _get_overlap_score(mean: float, std: float, x1: float, x2: float) -> float:
    # integrate distribution from x1 to x2
    norm_dist = stats.norm(loc=mean, scale=std)
    return norm_dist.cdf(x2) - norm_dist.cdf(x1)


def choose_next_candidate(
    df: pd.DataFrame,
    train_mask: Array,
    pred: Array,
    unct: Array,
    /,
    acquisition_function: str = "random",
    *,
    target_window: List[float] = None,
) -> int:
    """
    Choose the next candidate from the pool using the specified acquisition function.

    Args:
        df (pd.DataFrame): The full dataset.
        train_mask (Array): A mask for the indices of examples used for training.
            Usually of shape (len(df),) with each entry a boolean value indicating
            whether the example corresponding to that index was used for training.
        pred (Array): Predicted values for the property of interest.
        unct (Array): Uncertainties in the predicted values for the property of interest.
        acquisition_function (str, optional): Label of the acquisition function to use to
            rank and choose candidates from the pool of possible candidates (i.e., from
            all the examples in the full dataset that were not used for training).
            Options:
                - "random": select a candidate at random from the pool of candidates.
                - "mli": select the candidate predicted to have the maximum likelihood
                  of improvement (MLI) over a specified baseline.
                  In this particular problem, this is equivalent to maximum likelihood
                  of the value falling in the target window.
                - "mu": select the candidate with the maximum prediction uncertainty.
            Defaults to "random".
        target_window (List[float], optional): Window [lower limit, upper limit] of
            property values that is optimal. Used to calculate candidate scores for the
            MLI acquisition function; is ignored if any other acquisition function is
            specified.
            Defaults to [-0.7, -0.5] (the volcano peak that is relevant for CO2
            reduction reaction).

    Raises:
        NotImplementedError: If an acquisition function apart from the ones listed above
            is specified.

    Returns:
        parent_idx (int): The index of the selected candidate in the input dataframe.
    """
    if target_window is None:
        target_window = [-0.7, -0.5]
    print(f"  Acquisition function (AF): {acquisition_function}")
    if acquisition_function.lower() == "random":
        next_idx = np.random.choice(len(df["binding_energy_of_adsorbed"][~train_mask]))
        parent_idx = np.arange(df["binding_energy_of_adsorbed"].shape[0])[~train_mask][
            next_idx
        ]
        max_score = 0
    elif acquisition_function.lower() == "mli":
        # get scores of all compounds in the dataset
        scores = np.array(
            [
                _get_overlap_score(mean, std, *target_window)
                for mean, std in zip(pred, unct)
            ]
        )
        # choose next candidate based on maximum likelihood of improvement MLI
        next_idx = np.argmax(scores[~train_mask])
        # some index wrangling because the `next_idx` in the previous step is in
        # "masked" array excluding candidates in the training set; this next line
        # uses the same mask to recover the index of the chosen candidate in the
        # original, unmasked dataset.
        # a simple example here:
        # http://seanlaw.github.io/2015/09/10/numpy-argmin-with-a-condition/
        parent_idx = np.arange(scores.shape[0])[~train_mask][next_idx]
        # get the actual value of the MLI score for record keeping
        max_score = np.max(scores[~train_mask])
    elif acquisition_function.lower() == "mu":
        # choose next candidate based on maximum (prediction) uncertainty MU
        next_idx = np.argmax(unct[~train_mask])
        # get the parent index (see above for details of why this is needed)
        parent_idx = np.arange(unct.shape[0])[~train_mask][next_idx]
        # get the actual value of the MU score for record keeping
        max_score = np.max(unct[~train_mask])
    else:
        msg = f'Acquisition function "{acquisition_function}" not supported'
        raise NotImplementedError(msg)
    print(f"  Next index: {next_idx:4d}; Parent index: {parent_idx:4d}")
    print(f"  Maximum AF score: {max_score:.3f}")
    print(f"  Formula: {df['formula'][parent_idx]:10s}")
    print(f"  Actual value: {df['binding_energy_of_adsorbed'][parent_idx]:.2f}")
    print(f"  Predicted value: {pred[parent_idx]:.2f}")
    print(f"  Uncertainty: {unct[parent_idx]:.2f}")
    print("")
    return parent_idx


def do_simulated_sl(
    df: pd.DataFrame,
    column_to_magpie: str = "formula",
    target_column: str = "target",
    /,
    *,
    exclude_columns: List[str] = None,
    init_train_size: int = 10,
    n_iterations: int = 100,
    acquisition_function: str = "random",
    target_window: List[float] = None,
    model_kwargs: Dict = None,
) -> Dict:
    """
    Run a simulated sequential learning pipeline for a specified number of iterations.

    Args:
        df (pd.DataFrame): The full dataset.
        column_to_magpie (str, optional): Name of the column used to generate magpie
            features from. Defaults to "formula".
        target_column (str, optional): Name of the column with the target property to
            optimize over. Defaults to "target".
        exclude_columns (List[str], optional): Names of the columns that should be
            excluded from the dataframe to obtain the feature vectors.
            Defaults to ["id", "name", "formula", "composition"].
            Note that the `target_column` is automatically excluded from the feature
            vectors, so specifying it again as part of this argument is not necessary.
        init_train_size (int, optional): Number of examples to use build the initial ML
            models. The specified number of examples will be chosen randomly from the
            full dataset. Defaults to 10.
        n_iterations (int, optional): Number of sequential learning iterations to
            perform (i.e., number of times the ML retrained, predictions made, new
            candidate chosen). Defaults to 100.
        acquisition_function (str, optional): Label of the acquisition function to use
            to rank and choose candidates from the pool of possible candidates (i.e.,
            from all the examples in the full dataset that were not used for training).
            Options:
                - "random": select a candidate at random from the pool of candidates.
                - "mli": select the candidate predicted to have the maximum likelihood
                  of improvement (MLI) over a specified baseline.
                  In this particular problem, this is equivalent to maximum likelihood
                  of the value falling in the target window.
                - "mu": select the candidate with the maximum prediction uncertainty.
            Defaults to "random".
        target_window (List[float], optional): Window [lower limit, upper limit] of
            property values that is optimal. Used to calculate candidate scores for the
            MLI acquisition function; is ignored if any other acquisition function is
            specified. Defaults to [-0.7, -0.5] (the volcano peak that is relevant for
            CO2 reduction reaction).
        model_kwargs (Dict, optional): Any random forest model-related parameters. Will
            be passed on to the `RandomForestRegressor` class of the `lolo` library
            during model instantiation. Defaults to {}.
    Returns:
        history (Dict): A record of training masks, predictions, and prediction
            uncertainties made at every iteration of sequential learning (SL).
            For each iteration of SL, the corresponding lists are of length = number of
            examples/rows in the full dataset. The training masks are boolean values
            indicating whether a particular index was used for training or not, while
            the predictions and prediction uncertainties are floating point numbers.

            An example for a (very much hypothetical) dataset with 4 examples, 2 SL
            iterations, and 2 examples used to train the initial models:

            {
                "train_history": [[True, False, False, True], [True, False, True, True]],
                "pred_history": [[0.02, 0.01, -0.45, -0.36], [0.10, -0.32, -0.46, -0.36]],
                "unct_history": [[0.08, 0.34, 0.12, 0.20], [0.01, 0.05, 0.02, 0.05]]
            }
    """
    if exclude_columns is None:
        exclude_columns = {"id", "name", "formula", "composition"}
    _exclude_columns = set(exclude_columns)
    _exclude_columns.add(target_column)

    if model_kwargs is None:
        model_kwargs = {}

    df = _df_to_magpie_features(df, column=column_to_magpie)
    feature_columns = [c for c in df.columns if c not in _exclude_columns]
    X = np.array(df[feature_columns].values)
    y = np.array(df[target_column].values)

    # training masks and their per-iteration history
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[np.random.choice(len(X), init_train_size, replace=False)] = 1
    train_history = [train_mask]

    # store history of all predictions and uncertainties
    pred_history = []
    unct_history = []

    # build initial ML models
    model = RandomForestRegressor(**model_kwargs)
    model.fit(X[train_mask], y[train_mask])

    # predict, add candidate to training set, train, repeat
    ctr = 0
    while len(train_history) <= n_iterations:
        ctr += 1
        print(f"SL iteration #{ctr}")

        print("Predicting properties for candidates in the design space...")
        pred, unct = model.predict(X, return_std=True)
        pred_history.append(pred)
        unct_history.append(unct)

        train_mask = np.copy(train_history[-1])
        print("Choosing the next candidate...")
        next_idx = choose_next_candidate(
            df,
            train_mask,
            pred,
            unct,
            acquisition_function=acquisition_function,
            target_window=target_window,
        )

        train_mask[next_idx] = True
        train_history.append(train_mask)

        # re-train the ML model
        model.fit(X[train_mask], y[train_mask])

    # make predictions using the model trained in the last iteration
    pred, unct = model.predict(X, return_std=True)
    pred_history.append(pred)
    unct_history.append(unct)

    # the `numpy.tolist()` is here only to help JSONify later
    history = {
        "train_history": [t.tolist() for t in train_history],
        "pred_history": [p.tolist() for p in pred_history],
        "unct_history": [u.tolist() for u in unct_history],
    }
    return history


def do_multiple_simulated_sl_trials(
    df: pd.DataFrame,
    column_to_magpie: str = "formula",
    target_column: str = "target",
    /,
    *,
    exclude_columns: List[str] = None,
    init_train_size: int = 10,
    n_iterations: int = 100,
    n_trials: int = 20,
    acquisition_functions: List[str] = None,
    target_window: List[float] = None,
    model_kwargs: Dict = None,
) -> Dict:
    """
    Run a specified number of end-to-end independent trials of simulated sequential
    learning pipeline (for a specified number of iterations).

    Args:
        df (pd.DataFrame): The full dataset.
        column_to_magpie (str, optional): Name of the column used to generate magpie
            features from. Defaults to "formula".
        target_column (str, optional): Name of the column with the target property to
            optimize over. Defaults to "target".
        exclude_columns (List[str], optional): Names of the columns that should be
            excluded from the dataframe to obtain the feature vectors.
            Defaults to ["id", "name", "formula", "composition"].
            Note that the `target_column` is automatically excluded from the feature
            vectors, so specifying it again as part of this argument is not necessary.
        init_train_size (int, optional): Number of examples to use build the initial ML
            models. The specified number of examples will be chosen randomly from the
            full dataset. Defaults to 10.
        n_iterations (int, optional): Number of sequential learning iterations to
            perform (i.e., number of times the ML retrained, predictions made, new
            candidate chosen). Defaults to 100.
        n_trials (int, optional): Number of independent end-to-end trials of the
            simulated sequential learning pipelines to run, e.g., to derive statistics.
            Defaults to 20.
        acquisition_functions (List[str], optional): List of acquisition functions to
            use to rank and choose candidates from the pool of possible candidates
            (i.e., from all the examples in the full dataset that were not used for
            training). An independent sequential learning run is performed for each
            acquisition function specified in this argument.
            Options for acquisition functions:
                - "random": select a candidate at random from the pool of candidates.
                - "mli": select the candidate predicted to have the maximum likelihood
                  of improvement (MLI) over a specified baseline.
                  In this particular problem, this is equivalent to maximum likelihood
                  of the value falling in the target window.
                - "mu": select the candidate with the maximum prediction uncertainty.
            Defaults to ["random"].
        target_window (List[float], optional): Window [lower limit, upper limit] of
            property values that is optimal. Used to calculate candidate scores for the
            MLI acquisition function; is ignored if any other acquisition function is
            specified. Defaults to [-0.7, -0.5] (the volcano peak that is relevant for
            CO2 reduction reaction).
        model_kwargs (Dict, optional): Any random forest model-related parameters. Will
            be passed on to the `RandomForestRegressor` class of the `lolo` library
            during model instantiation. Defaults to {}.
    Returns:
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
    """
    if acquisition_functions is None:
        acquisition_functions = ["random"]

    # multiple full SL pipeline trials for extracting robust stats
    histories = {}
    for acquisition_function in acquisition_functions:
        history_per_trial = {}
        for trial in range(1, n_trials + 1):
            print(f"TRIAL #{trial}")
            print("")
            history_per_trial[trial] = do_simulated_sl(
                df,
                column_to_magpie=column_to_magpie,
                target_column=target_column,
                exclude_columns=exclude_columns,
                init_train_size=init_train_size,
                n_iterations=n_iterations,
                acquisition_function=acquisition_function,
                target_window=target_window,
                model_kwargs=model_kwargs,
            )
        histories[acquisition_function] = history_per_trial
    return histories


if __name__ == "__main__":
    # directory with the bimetallic catalysts data
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(thisdir, "..", "data", "bimetallic_catalysts_dataset")

    # read in the bimetallics adsorption energy dataset
    csv_path = os.path.join(datadir, "bimetallics_data.csv")
    df = pd.read_csv(csv_path)

    # SL parameters to reproduce the data/analysis/figures in the associated publication:
    parameters = {
        "column_to_magpie": "formula",
        "target_column": "binding_energy_of_adsorbed",
        "exclude_columns": [
            "name",
            "formula",
            "comp",
            "composition",
            "top_monolayer_formula",
            "bulk_formula",
        ],
        "init_train_size": 10,
        "n_iterations": 100,
        "n_trials": 20,
        "acquisition_functions": ["random", "mli", "mu"],
        "target_window": [-0.7, -0.5],
    }

    # run multiple independent SL pipelines and get history of training/predictions
    histories = do_multiple_simulated_sl_trials(df, **parameters)

    # write training/prediction histories to disk
    with gzip.open("histories.json.gz", "wt", encoding="utf-8") as fw:
        json.dump(histories, fw, indent=2)
