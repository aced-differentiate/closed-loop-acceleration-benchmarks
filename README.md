# By how much can closed-loop frameworks accelerate computational materials discovery?

Data and scripts in support of the publication "By how much can closed-loop frameworks accelerate computational materials discovery?", Kavalsky et al., arXiv:2211.10533 [cond-mat.mtrl-sci] (2022). DOI: [10.48550/arXiv.2211.10533](https://doi.org/10.48550/arXiv.2211.10533).

The repository is organized as follows:

1. [data/](data)

    * `benchmark_calculations_record.xlsx`: Excel spreadsheet containing a record of DFT calculations, associated raw timestamps, and a tabulation of the acceleration estimates.

    * [bimetallic\_catalysts\_dataset/](data/bimetallic_catalysts_dataset)

        * `ma_2015_bimetallics_raw.json.gz`: Dataset of bimetallic alloys for CO2 reduction, in the [Physical Information File (PIF)](https://citrineinformatics.github.io/pif-documentation) format, obtained from [Dataset 153450](https://citrination.com/datasets/153450) on Citrination.
  
            Original data source: "Machine-Learning-Augmented Chemisorption Model for CO2 Electroreduction Catalyst Screening", Ma et al., *J. Phys. Chem. Lett.* **6** 3528-3533 (2015). DOI: [10.1021/acs.jpclett.5b01660](http://dx.doi.org/10.1021/acs.jpclett.5b01660)

        * `transform.py`: Python script for converting from the PIF format into tabular data.

        * `bimetallics_data.csv`: Bimetallics catalysts dataset mentioned above in a tabular format.

    * [runtime\_geometries/](data/runtime_geometries)

        "Chemically-informed" and naive structures and settings in the form of `ase.traj` files, corresponding to the discussion surrounding FIG. 3 in the main text. The files can be read using ASE package (using [`ase.io.read`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read)).

2. [scripts/](scripts)

    * `human_lagtime.py`: Script for estimating human lagtime in job management, calculated using a Monte Carlo sampling method.

    * `sequential_learning.py`: Script for running multiple independent trials of sequential learning (SL) and recording a history of training examples, model predictions and prediction uncertainties.
    
        If run as-is, the script performs 20 independent trials of 100 SL iterations to optimize the `binding_energy_of_adsorbed` property in the bimetallic catalysts dataset mentioned above, using four acquisition functions (results from each recorded separately): random, maximum likelihood of improvement (MLI), maximum uncertainty (MU), and space-filling.

    * `plot_acceleration_from_sequential_learning.py`: Script to aggregate results from the `sequential_learning.py` script, calculate and plot statistics related to acceleration from SL over a baseline.

        If run as-is, the script reproduces the 3-paneled FIG. 5 in the main text.

    * `plot_acceleration_from_sequential_learning__ALL_ACQ.py`: Similar to the previous script; plots and compares statistics from all acquisition functions considered (MLI, MU, random, space-filling) for all SL tasks.

        If run as-is, the script reproduces the 3-paneled FIG. S1 in the Supplementary Information.

    * `plot_levels_of_automation.py`: Script for plotting the cumulative time  of executing the DFT pipeline at varying levels of automation.

        If run as-is, the script reproduces the bottom panel from FIG. 2 in the main text.

## Running the scripts

The required packages for executing the scripts are specified in `requirements.txt`,
and can be installed in a new environment (e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/index.html))
as follows:

```py
$ conda create -n accel_benchmarking python=3.10
$ conda activate accel_benchmarking
$ pip install -r requirements.txt
```

The scripts are all in python, and can be run from the command line. For example:
```py
$ cd scripts
$ python sequential_learning.py
```
