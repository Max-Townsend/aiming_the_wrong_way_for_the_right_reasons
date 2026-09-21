# Aiming the wrong way for the right reasons

People sometimes aim in the wrong direction after visual feedback changes. SHIFT is a Bayesian model of how people infer the relationship between their actions and visual feedback, and how those beliefs shape their aiming strategies. The name stands for Spectral Hierarchical Inference of Field Transformations.

This repository contains Python code, experimental data and saved fits. The analyses compare SHIFT with Q-HMM across three datasets: target geometry (Ding), strategic savings and target-specific perturbations (wildcard).

## Run an analysis

Use **Python 3.11** and **Git LFS**. The data and fits require approximately **744 MiB** of LFS downloads. In PowerShell:

```powershell
git lfs install
git clone https://github.com/Max-Townsend/aiming_the_wrong_way_for_the_right_reasons.git
cd aiming_the_wrong_way_for_the_right_reasons
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py run figure33
```

On macOS/Linux, create the environment with `python3.11 -m venv .venv` and activate it with `source .venv/bin/activate`.

`figure33` runs the strategic-savings analyses using the saved fits. Outputs and logs go to `outputs/reproduction/`. See the [reproduction guide](docs/REPRODUCING.md) for other figures, statistics and fitting checks. Figure numbers follow Chapter 3 of my PhD thesis. Full refitting can take days.

## Code

- [SHIFT](src/models/BayesHypothesisTesting.py) and [Q-HMM](src/models/HMM.py): model implementations.
- [Fitting](src/chapter3/fitting.py): estimation and checkpoints.
- [Figures](src/chapter3/figures.py): plotting and model comparisons.
- [reproduce.py](reproduce.py): analysis commands and output workspaces.

## Paper

Townsend, M., Warburton, M., Campagnoli, C., Mon-Williams, M., Mushtaq, F., & Morehead, J. R. *Aiming the wrong way for the right reasons.* In preparation. [Request the manuscript](mailto:max.o.b.townsend@gmail.com?subject=Request%3A%20Aiming%20the%20wrong%20way%20for%20the%20right%20reasons).

Max Townsend · [max.o.b.townsend@gmail.com](mailto:max.o.b.townsend@gmail.com)
