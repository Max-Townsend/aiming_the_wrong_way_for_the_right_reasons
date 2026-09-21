#Aiming the wrong way for the right reasons

**A generative model of how people infer changes in their environment and choose movement strategies.**

People can respond to the same visual perturbation with very different strategies, including initially aiming in the wrong direction. SHIFT (**Spectral Hierarchical Inference of Field Transformations**) models behaviour through Bayesian inference over possible transformations between an action and its visual consequences.

Research code and data from **Max Townsend's PhD in computational cognitive science**, supporting *Aiming the wrong way for the right reasons* (**manuscript in preparation**).

[Request the manuscript](mailto:max.o.b.townsend@gmail.com?subject=Request%3A%20Aiming%20the%20wrong%20way%20for%20the%20right%20reasons) · [Reproduction guide](docs/REPRODUCING.md) · [Contact](mailto:max.o.b.townsend@gmail.com)

## What is included

- **Human behavioural modelling:** trial-by-trial inference and predictive distributions for heterogeneous aiming behaviour.
- **Three experimental settings:** target geometry (Ding), strategic savings and target-specific perturbations (wildcard), with SHIFT and Q-HMM model comparisons.
- **Reproducible analysis:** raw and processed data, saved fits, figure recipes, statistical analyses, checkpointed fitting and a fitting smoke test.

**Stack:** Python · NumPy / SciPy · Numba · CMA-ES · pandas · Matplotlib / seaborn.

## Explore the code

| Start here | What to look for |
|---|---|
| [SHIFT model](src/models/BayesHypothesisTesting.py) | Bayesian inference, structural hypotheses and predictive sampling |
| [Q-HMM comparison](src/models/HMM.py) | Alternative model of trial-by-trial behaviour |
| [Fitting workflows](src/chapter3/fitting.py) | Dataset-specific estimation and saved checkpoints |
| [Figure recipes](src/chapter3/figures.py) | Human/model comparisons across experiments |
| [Reproduction entry point](reproduce.py) | Named tasks, isolated output workspaces and run logs |

## Reproduce a figure

Install **Python 3.11** and **Git LFS**. Data and saved fits require approximately **744 MiB** of LFS downloads.

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

`figure33` uses the supplied fits to reproduce the strategic-savings analyses. Outputs and run logs go to `outputs/reproduction/`. Figure numbers retain the thesis Chapter 3 numbering. See the [reproduction guide](docs/REPRODUCING.md) for all figures, statistics, reduced checks and refitting. Full refitting can take days.

## Paper and contact

Townsend, M., Warburton, M., Campagnoli, C., Mon-Williams, M., Mushtaq, F., & Morehead, J. R. **Aiming the wrong way for the right reasons.** Manuscript in preparation; available on request.

For the manuscript or questions about the model, contact [max.o.b.townsend@gmail.com](mailto:max.o.b.townsend@gmail.com).
