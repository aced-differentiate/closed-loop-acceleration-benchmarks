"""Converts a list of PIFs into tabular data for easier manipulation."""

import os
import gzip
import json
import random
from typing import List
from typing import Union

import pandas as pd

from pymatgen.core import Composition


def _read_pifs(json_file: str) -> List[dict]:
    print(f"Reading PIFs from {json_file}...")
    if os.path.splitext(json_file)[-1] == ".gz":
        with gzip.open(json_file, "rb") as fr:
            pifs = json.load(fr)
    else:
        with open(json_file, "rb") as fr:
            pifs = json.load(fr)
    print(f"Successfully read {len(pifs)} PIFs")
    return pifs


def _get_overall_composition(pif: dict) -> dict:
    comp = {e["element"]: e["idealAtomicPercent"] for e in pif["composition"]}
    if not comp:
        return None
    return comp


def _get_prop_value(pif: dict, prop_name: str) -> Union[str, float]:
    prop = filter(lambda x: prop_name in x["name"], pif["properties"])
    try:
        val = list(prop)[0]["scalars"]["value"]
    except IndexError:
        val = None
    return val


def get_input_data_as_df(
    pifs: List[dict],
    /,
    properties: List[str] = None,
    *,
    write_to_disk: bool = True,
    fname: str = "data.csv",
) -> pd.DataFrame:
    """
    Transform the input list of PIFs to a pd.DataFrame.

    Args:
        pifs (List[dict]): List of PIFs.
        properties (List[str]): List of properties to extract from each PIF.
            Defaults to the following:
                [
                    "Top monolayer formula",
                    "Bulk formula",
                    "Binding energy of adsorbed",
                    "Filling of a d-band",
                    "Center of a d-band",
                    "Width of a d-band",
                    "Skewness of a d-band",
                    "Kurtosis of a d-band",
                    "Work function",
                    "Atomic radius",
                    "Spatial extent of d-orbitals",
                    "Ionization potential",
                    "Electron affinity",
                    "Pauling electronegativity",
                    "Local Pauling electronegativity",
                    "d coupling matrix"
                ]
        write_to_disk (bool, optional): Whether to write the dataframe to disk.
            Defaults to True.
        fname (str, optional): Path to the file to write the dataframe to.
            Defaults to "data.csv".

    Returns:
        pd.DataFrame: Input PIFs data as a dataframe.
    """
    if properties is None:
        properties = [
            "Top monolayer formula",
            "Bulk formula",
            "Binding energy of adsorbed",
            "Filling of a d-band",
            "Center of a d-band",
            "Width of a d-band",
            "Skewness of a d-band",
            "Kurtosis of a d-band",
            "Work function",
            "Atomic radius",
            "Spatial extent of d-orbitals",
            "Ionization potential",
            "Electron affinity",
            "Pauling electronegativity",
            "Local Pauling electronegativity",
            "d coupling matrix",
        ]

    ddict = {}
    for pif in pifs:
        names = set(pif["name"])

        # only one name (to be used as UID)
        assert len(names) == 1
        name = list(names)[0]

        # get alloy composition
        comp = _get_overall_composition(pif)

        # get pymatgen reduced formula for later labeling
        formula = Composition(comp).reduced_formula

        # add this entry to the data dictionary
        ddict[name] = {"name": name, "formula": formula, "comp": comp}

        # get all properties to use for training, add to data dictionary
        for prop in properties:
            prop_key = "_".join(prop.lower().split())
            ddict[name].update({prop_key: _get_prop_value(pif, prop)})

    print(f"# of entries in the parsed input data: {len(ddict)}")

    # convert to df
    # list of dictionary keys as df column names
    sample_key = random.choice(list(ddict.keys()))
    # note: columns here are unsorted (no need to)
    columns = list(ddict[sample_key].keys())
    data = []
    for name, props in ddict.items():
        data.append([props[c] for c in columns])

    df = pd.DataFrame(data=data, columns=columns)
    print(df)

    if write_to_disk:
        print(f"Input data written to {fname}.")
        df.to_csv(fname, index=False)
    return df


if __name__ == "__main__":
    pifs_file = "ma_2015_bimetallics_raw.json.gz"
    pifs = _read_pifs(pifs_file)
    df = get_input_data_as_df(pifs, fname="bimetallics_data.csv")
    print(df)
