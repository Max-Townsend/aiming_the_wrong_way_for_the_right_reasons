# Reproducing the SHIFT analyses

[Back to the project overview](../README.md)

These commands reproduce the Chapter 3 analyses. Figure numbers retain the thesis numbering. Raw data, processed data and saved fits require Git LFS (approximately 744 MiB).

## Setup

Install Python 3.11 and Git LFS, then clone the repository and create the environment:

```powershell
git lfs install
git clone https://github.com/Max-Townsend/aiming_the_wrong_way_for_the_right_reasons.git
cd aiming_the_wrong_way_for_the_right_reasons
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py verify
```

On macOS/Linux, create the environment with `python3.11 -m venv .venv` and activate it with `source .venv/bin/activate`.

## Figures and statistics

The supplied fits can be used directly:

```powershell
python reproduce.py list
python reproduce.py run figure33
python reproduce.py all
```

| Task | Output |
|---|---|
| `figure31` | Model and target-geometry schematics |
| `figure32` | Ding learning, generalisation and model comparisons |
| `figure33` | Savings learning curves and model comparisons |
| `figure34` | Wildcard learning, target-specificity and model comparisons |
| `geometry` | Target-geometry illustration |
| `statistics` | Ding parameter and structural-belief tests; wildcard learning statistics |
| `manuscript-statistics` | Generalisation, early structural-belief and parameter analyses |
| `demographics` | Wildcard participant summary |

`all` runs the four figure tasks and all three statistical summaries. Full figure runs include participant plots and 10,000-sample energy scores, which can take substantial time. The exported panels support the thesis figures; final page composition and labels were arranged in Illustrator.

Code and inputs are copied to `outputs/<workspace>/`, with `reproduction` as the default workspace. Figures, tables, checkpoints and run logs are written there. Use `--workspace <name>` to create a separate run, and choose a new name after changing code or inputs.

## Fit from raw data

```powershell
$env:SHIFT_CORES = '4'
python reproduce.py all --from-scratch --workspace fresh-refit
```

On macOS/Linux, set the worker count with `export SHIFT_CORES=4`.

This preprocesses the three datasets, fits both models, recovers wildcard weights, and runs the figures and statistics. It requires a new workspace. Demographics use the supplied linked participant table. Full fitting can take days, and stochastic optimisation can produce different estimates from the supplied fits.

To resume an interrupted fit, run its task in the same workspace, then continue with the remaining tasks:

```powershell
python reproduce.py run fit-ding-shift --workspace fresh-refit
```

Fitting task names follow `fit-<dataset>-<model>`, where datasets are `ding`, `savings`, and `wildcard`, and models are `shift` and `qhmm`. `preprocess` prepares the data, and `recover-weights` calculates wildcard local/global weights from the SHIFT fit.

## Checks

`python reproduce.py verify` checks required files, Git LFS downloads and Python syntax. The fitting smoke test uses small trial subsets to check all six fitting tasks, saved fits and checkpoint resume:

```powershell
python scripts/smoke_test.py --workspace fit-check
```

Use a new workspace or `--run-name` for each smoke test. `check-figure32` and `check-figure34` run reduced plotting checks; their sample counts are for testing.

## Project layout

| Location | Contents |
|---|---|
| `src/chapter3/` | Data preparation, fitting, figures and statistics |
| `src/models/` | SHIFT and Q-HMM implementations |
| `src/preprocessing/` | Data conversion and demographic summaries |
| `src/analysis/` | Energy scores, learning metrics and plotting helpers |
| `data/raw/` | Experimental inputs |
| `data/processed/` | Analysis-ready data and linked demographics |
| `data/fits/` | Saved fits and wildcard weight estimates |
| `scripts/` | Fitting smoke test |
| `outputs/` | Analysis workspaces; ignored by Git |
