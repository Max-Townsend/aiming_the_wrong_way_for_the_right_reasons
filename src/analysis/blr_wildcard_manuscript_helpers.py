"""Wildcard learning metrics, parameter analyses and summary plots.

Compensatory aim is -aim * sign(rotation). Generalisation sign-flips are
relative to the participant's preceding feedback trials. Coupling fidelity
compares observed target-rotation assignments with within-participant
permutations of final-block target means.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


FREE_PARAM_NAMES = [
    "logNegLogH",
    "logVarTrans",
    "logAlpha",
    "kappa",
    "logPriorOddsStruct",
    "logVisCoeff",
    "logBeta",
]

TRANSFORMED_PARAM_NAMES = [
    "hazard_h",
    "sigma_T",
    "alpha_self",
    "alpha_other",
    "alpha_other_minus_self",
    "visCoeff",
    "beta",
]

DEFAULT_FINAL_BLOCK = 4

MODEL_PARAM_ORDER = [
    "logAlpha",
    "kappa",
    "logBeta",
    "logPriorOddsStruct",
    "logVarTrans",
    "logVisCoeff",
]

PALETTE = {
    "primary": "#355C7D",
    "secondary": "#C06C84",
    "accent": "#6C5B7B",
    "catch": "#2A9D8F",
    "trained": "#E76F51",
    "fidelity": "#1D3557",
    "permutation": "#8D99AE",
}


def set_wildcard_style() -> None:
    """Apply a manuscript-friendly plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def _ensure_dir(path: os.PathLike | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure_triplet(fig: plt.Figure, path_base: os.PathLike | str, *, dpi: int = 300) -> dict:
    """Save a figure as PNG, SVG, and PDF.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    path_base:
        Either a stem path or a path with any suffix. The suffix is ignored and
        replaced with ``.png``, ``.svg``, and ``.pdf``.
    dpi:
        Raster export DPI for the PNG.
    """
    base = Path(path_base)
    if base.suffix:
        base = base.with_suffix("")
    _ensure_dir(base.parent)
    outputs = {}
    for suffix in [".png", ".svg", ".pdf"]:
        out_path = base.with_suffix(suffix)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        outputs[suffix.lstrip(".")] = str(out_path)
    return outputs


def _parse_array_cell(text):
    if pd.isna(text):
        return []
    if isinstance(text, (list, tuple, np.ndarray)):
        return list(text)
    return json.loads(str(text).replace("nan", "null"))


def _wrap_signed_comp(rotation: float, aim: float) -> float:
    if rotation is None or np.isnan(rotation):
        return np.nan
    if aim is None or np.isnan(aim):
        return np.nan
    if float(rotation) == 0:
        return float(aim)
    return float(-float(aim) * np.sign(float(rotation)))


def _safe_sign(value: float) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = float(np.sign(value))
    return int(s)


def _to_fit_shell(obj):
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    if isinstance(obj, np.ndarray) and len(obj) == 1:
        return obj[0]
    return obj


def load_wildcard_fits(fit_path: os.PathLike | str = "BHTWildcard.npy"):
    """Load the fitted Wildcard BHT model."""
    loaded = np.load(fit_path, allow_pickle=True)
    return _to_fit_shell(loaded)


def load_wildcard_task(csv_path: os.PathLike | str = "wildCardTask.csv") -> pd.DataFrame:
    """Load the Wildcard task CSV."""
    return pd.read_csv(csv_path)


def extract_wildcard_param_table(fit) -> pd.DataFrame:
    """Return one participant-level row per fitted Wildcard subject."""
    rows = []
    for pp, xs in zip(fit.participantNums, fit.xs):
        xs = list(xs)
        if len(xs) < 9:
            continue
        row = {
            "participantNum": int(pp),
            "condition": getattr(fit, "condition", None),
        }
        for name, value in zip(FREE_PARAM_NAMES, xs[2:9]):
            row[name] = float(value)

        row["hazard_h"] = float(np.exp(-np.exp(np.clip(row["logNegLogH"], -700, 700))))
        row["sigma_T"] = float(np.exp(0.5 * np.clip(row["logVarTrans"], -700, 700)))
        row["alpha_self"] = float(
            np.log1p(np.exp(-abs(row["logAlpha"]))) + max(-row["logAlpha"], 0)
        )
        row["alpha_other"] = float(
            np.log1p(np.exp(-abs(row["logAlpha"]))) + max(row["logAlpha"], 0)
        )
        row["alpha_other_minus_self"] = row["alpha_other"] - row["alpha_self"]
        row["visCoeff"] = float(np.exp(np.clip(row["logVisCoeff"], -700, 700)))
        row["beta"] = float(np.exp(np.clip(row["logBeta"], -700, 700)))

        rows.append(row)

    return pd.DataFrame(rows)


def build_wildcard_rotation_phase_table(task_df: pd.DataFrame) -> pd.DataFrame:
    """Build a long table of rotation-phase trials from the wide Wildcard CSV."""
    rows = []
    for _, wide_row in task_df.iterrows():
        participant_num = int(wide_row["participantNum"])
        rotations = _parse_array_cell(wide_row["rotation"])
        aims = _parse_array_cell(wide_row["aim"])
        targets = _parse_array_cell(wide_row["targetPosition"])
        if len(rotations) == 0:
            continue
        nonzero = [i for i, r in enumerate(rotations) if r is not None and float(r) != 0]
        if not nonzero:
            continue
        first_idx = min(nonzero)
        last_idx = max(nonzero)
        for trial_idx in range(first_idx, last_idx + 1):
            rotation = rotations[trial_idx]
            aim = aims[trial_idx] if trial_idx < len(aims) else None
            target = targets[trial_idx] if trial_idx < len(targets) else None
            if rotation is None or aim is None or target is None:
                continue
            rotation = float(rotation)
            aim = float(aim)
            target = float(target)
            signed_comp = _wrap_signed_comp(rotation, aim)
            rows.append(
                {
                    "participantNum": participant_num,
                    "trial_idx": int(trial_idx),
                    "target": target,
                    "rotation": rotation,
                    "magnitude": int(abs(rotation)) if rotation != 0 else 0,
                    "aim": aim,
                    "signed_comp": signed_comp,
                }
            )

    return pd.DataFrame(rows)


def _participant_training_sign(history_df: pd.DataFrame) -> float | None:
    """Infer a participant's current training sign from previous feedback trials."""
    if history_df.empty:
        return None
    vals = history_df["signed_comp"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    s = _safe_sign(np.nanmedian(vals))
    if s == 0:
        s = _safe_sign(np.nanmean(vals))
    return float(s) if s != 0 else None


def _final_block_mean(series: pd.Series, final_block: int) -> float:
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals[-final_block:]))


def _exact_permutation_mae(values: np.ndarray, rotations: np.ndarray, max_exact: int = 40320):
    """Compute exact or near-exact within-participant permutation MAE."""
    values = np.asarray(values, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    n = len(values)
    if n == 0:
        return {
            "actual_mae": np.nan,
            "perm_mae_mean": np.nan,
            "perm_mae_sd": np.nan,
            "delta_mae_vs_perm_mean": np.nan,
            "perm_p_value": np.nan,
            "prop_actual_better_than_perm_mean": np.nan,
            "prop_actual_is_best_permutation": np.nan,
            "n_permutations": 0,
        }

    actual_mae = float(np.mean(np.abs(values - rotations)))
    if n > 8:
        # Bound factorial growth with reproducible sampled permutations.
        perm_iter = itertools.islice(itertools.permutations(rotations), max_exact)
    else:
        perm_iter = itertools.permutations(rotations)

    perm_maes = np.fromiter(
        (float(np.mean(np.abs(values - np.asarray(perm, dtype=float)))) for perm in perm_iter),
        dtype=float,
    )
    if len(perm_maes) == 0:
        return {
            "actual_mae": actual_mae,
            "perm_mae_mean": np.nan,
            "perm_mae_sd": np.nan,
            "delta_mae_vs_perm_mean": np.nan,
            "perm_p_value": np.nan,
            "prop_actual_better_than_perm_mean": np.nan,
            "prop_actual_is_best_permutation": np.nan,
            "n_permutations": 0,
        }

    perm_mean = float(np.mean(perm_maes))
    perm_sd = float(np.std(perm_maes, ddof=1)) if len(perm_maes) > 1 else 0.0
    delta = perm_mean - actual_mae
    p_value = float((np.sum(perm_maes <= actual_mae) + 1) / (len(perm_maes) + 1))
    prop_better = float(np.mean(actual_mae < perm_maes))
    prop_best = float(np.mean(actual_mae <= perm_maes.min()))

    return {
        "actual_mae": actual_mae,
        "perm_mae_mean": perm_mean,
        "perm_mae_sd": perm_sd,
        "delta_mae_vs_perm_mean": delta,
        "perm_p_value": p_value,
        "prop_actual_better_than_perm_mean": prop_better,
        "prop_actual_is_best_permutation": prop_best,
        "n_permutations": int(len(perm_maes)),
    }


def build_wildcard_human_metrics(
    task_df: pd.DataFrame,
    final_block: int = DEFAULT_FINAL_BLOCK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build participant-level human metrics and supporting tables.

    Returns
    -------
    participant_metrics_df:
        One row per participant with trained-target compensation, catch-target
        compensation, dose slope, and coupling-fidelity summaries.
    final_mag_df:
        One row per participant per magnitude with the final-block compensation.
    final_target_df:
        One row per participant per target with the final-block target-specific
        compensation and target-specific final rotation.
    """
    final_mag_rows = []
    final_target_rows = []
    participant_rows = []

    for _, wide_row in task_df.iterrows():
        participant_num = int(wide_row["participantNum"])
        rotations = _parse_array_cell(wide_row["rotation"])
        aims = _parse_array_cell(wide_row["aim"])
        targets = _parse_array_cell(wide_row["targetPosition"])
        if not rotations:
            continue

        nonzero = [i for i, r in enumerate(rotations) if r is not None and float(r) != 0]
        if not nonzero:
            continue
        first_idx = min(nonzero)
        last_idx = max(nonzero)

        trial_rows = []
        for trial_idx in range(first_idx, last_idx + 1):
            if trial_idx >= len(rotations):
                continue
            rotation = rotations[trial_idx]
            aim = aims[trial_idx] if trial_idx < len(aims) else None
            target = targets[trial_idx] if trial_idx < len(targets) else None
            if rotation is None or aim is None or target is None:
                continue
            rotation = float(rotation)
            aim = float(aim)
            target = float(target)
            signed_comp = _wrap_signed_comp(rotation, aim)
            trial_rows.append(
                {
                    "participantNum": participant_num,
                    "trial_idx": int(trial_idx),
                    "target": target,
                    "rotation": rotation,
                    "magnitude": int(abs(rotation)) if rotation != 0 else 0,
                    "signed_comp": signed_comp,
                }
            )

        trial_df = pd.DataFrame(trial_rows)
        if trial_df.empty:
            continue

        mag_final = {}
        for magnitude, sub in trial_df.groupby("magnitude"):
            mag_final[int(magnitude)] = _final_block_mean(
                sub.sort_values("trial_idx")["signed_comp"], final_block
            )
            final_mag_rows.append(
                {
                    "participantNum": participant_num,
                    "magnitude": int(magnitude),
                    "final_comp": mag_final[int(magnitude)],
                }
            )

        target_final = {}
        target_final_rot = {}
        for target, sub in trial_df.groupby("target"):
            sub = sub.sort_values("trial_idx")
            target_final[float(target)] = _final_block_mean(sub["signed_comp"], final_block)
            target_final_rot[float(target)] = _final_block_mean(sub["rotation"], final_block)
            final_target_rows.append(
                {
                    "participantNum": participant_num,
                    "target": float(target),
                    "final_target_comp": target_final[float(target)],
                    "rotation_final": target_final_rot[float(target)],
                }
            )

        available = sorted((m, v) for m, v in mag_final.items() if np.isfinite(v))
        if len(available) >= 2:
            mags = np.array([m for m, _ in available], dtype=float)
            comps = np.array([v for _, v in available], dtype=float)
            dose_slope = float(np.polyfit(mags, comps, 1)[0])
        else:
            dose_slope = np.nan

        trained_vals = [v for m, v in mag_final.items() if m > 0 and np.isfinite(v)]
        trained_mean = float(np.mean(trained_vals)) if trained_vals else np.nan
        catch_mean = float(mag_final.get(0, np.nan))
        trained_vs_catch_gap = (
            trained_mean - catch_mean
            if np.isfinite(trained_mean) and np.isfinite(catch_mean)
            else np.nan
        )

        coupling = _exact_permutation_mae(
            values=np.array([target_final[t] for t in sorted(target_final)], dtype=float),
            rotations=np.array(
                [target_final_rot[t] for t in sorted(target_final_rot)], dtype=float
            ),
        )

        participant_rows.append(
            {
                "participantNum": participant_num,
                "n_rotation_trials": int(len(trial_df)),
                "n_magnitudes": int(len(mag_final)),
                "n_targets": int(len(target_final)),
                "trained_mean": trained_mean,
                "catch_mean": catch_mean,
                "trained_vs_catch_gap": trained_vs_catch_gap,
                "dose_slope": dose_slope,
                "actual_mae": coupling["actual_mae"],
                "perm_mae_mean": coupling["perm_mae_mean"],
                "perm_mae_sd": coupling["perm_mae_sd"],
                "delta_mae_vs_perm_mean": coupling["delta_mae_vs_perm_mean"],
                "perm_p_value": coupling["perm_p_value"],
                "prop_actual_better_than_perm_mean": coupling["prop_actual_better_than_perm_mean"],
                "prop_actual_is_best_permutation": coupling["prop_actual_is_best_permutation"],
                "n_permutations": coupling["n_permutations"],
                "coupling_fidelity": coupling["delta_mae_vs_perm_mean"],
            }
        )

    participant_metrics_df = pd.DataFrame(participant_rows)
    final_mag_df = pd.DataFrame(final_mag_rows)
    final_target_df = pd.DataFrame(final_target_rows)
    return participant_metrics_df, final_mag_df, final_target_df


def merge_wildcard_metrics_and_params(
    participant_metrics_df: pd.DataFrame,
    param_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge participant metrics with fitted model parameters."""
    merged = participant_metrics_df.merge(param_df, on="participantNum", how="inner")
    return merged


def correlation_summary(
    merged_df: pd.DataFrame,
    metrics: Sequence[str],
    params: Sequence[str],
    *,
    method: str = "spearman",
    fdr: bool = True,
) -> pd.DataFrame:
    """Compute metric/parameter correlation summaries."""
    rows = []
    for metric in metrics:
        for param in params:
            sub = merged_df[[metric, param]].dropna()
            if len(sub) < 5:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(sub[metric], sub[param])
            else:
                r, p = stats.spearmanr(sub[metric], sub[param])
            rows.append(
                {
                    "metric": metric,
                    "param": param,
                    "method": method,
                    "r": float(r),
                    "p_unc": float(p),
                    "n": int(len(sub)),
                }
            )

    out = pd.DataFrame(rows)
    if fdr and not out.empty:
        out["p_fdr"] = multipletests(out["p_unc"], method="fdr_bh")[1]
    return (
        out.sort_values(["p_fdr", "p_unc"], na_position="last").reset_index(drop=True)
        if not out.empty
        else out
    )


def focused_regression_summary(
    merged_df: pd.DataFrame,
    outcomes: Sequence[str],
    predictors: Sequence[str] = MODEL_PARAM_ORDER,
    *,
    robust_cov: str = "HC3",
    standardize: bool = True,
) -> pd.DataFrame:
    """Fit focused OLS models for a set of outcomes.

    Parameters
    ----------
    merged_df:
        Participant-level table with both behavior metrics and fitted params.
    outcomes:
        Dependent variables to model.
    predictors:
        Model parameters to include.
    robust_cov:
        Covariance type passed to statsmodels.
    standardize:
        If True, z-score both outcomes and predictors before fitting. This makes
        coefficients directly comparable across terms.
    """
    rows = []

    def zscore(series: pd.Series) -> pd.Series:
        sd = series.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return series * np.nan
        return (series - series.mean()) / sd

    for outcome in outcomes:
        cols = [outcome] + list(predictors)
        sub = merged_df[cols].dropna().copy()
        if len(sub) < len(predictors) + 3:
            continue
        if standardize:
            for col in cols:
                sub[col] = zscore(sub[col])
            y_name = f"z_{outcome}"
        else:
            y_name = outcome

        formula = f"{outcome} ~ " + " + ".join(predictors)
        fit = smf.ols(formula, data=sub).fit(cov_type=robust_cov)
        for term, coef in fit.params.items():
            if term == "Intercept":
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": float(coef),
                    "std_err": float(fit.bse[term]),
                    "t": float(fit.tvalues[term]),
                    "p": float(fit.pvalues[term]),
                    "ci_low": float(fit.conf_int().loc[term, 0]),
                    "ci_high": float(fit.conf_int().loc[term, 1]),
                    "r_squared": float(fit.rsquared),
                    "n": int(len(sub)),
                    "standardized": bool(standardize),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
        out = out.sort_values(["outcome", "p"]).reset_index(drop=True)
    return out


def build_wildcard_analysis_bundle(
    fit_path: os.PathLike | str = "BHTWildcard.npy",
    csv_path: os.PathLike | str = "wildCardTask.csv",
    *,
    final_block: int = DEFAULT_FINAL_BLOCK,
) -> dict:
    """Load everything needed for the Wildcard manuscript analyses."""
    fit = load_wildcard_fits(fit_path)
    task_df = load_wildcard_task(csv_path)
    param_df = extract_wildcard_param_table(fit)
    rotation_df = build_wildcard_rotation_phase_table(task_df)
    participant_metrics_df, final_mag_df, final_target_df = build_wildcard_human_metrics(
        task_df,
        final_block=final_block,
    )
    merged_df = merge_wildcard_metrics_and_params(participant_metrics_df, param_df)
    corr_df = correlation_summary(
        merged_df,
        metrics=[
            "trained_mean",
            "catch_mean",
            "trained_vs_catch_gap",
            "dose_slope",
            "coupling_fidelity",
        ],
        params=[
            "logAlpha",
            "alpha_other_minus_self",
            "kappa",
            "logBeta",
            "beta",
            "logPriorOddsStruct",
            "logVarTrans",
            "sigma_T",
            "logVisCoeff",
            "visCoeff",
        ],
    )
    reg_df = focused_regression_summary(
        merged_df,
        outcomes=[
            "trained_mean",
            "catch_mean",
            "trained_vs_catch_gap",
            "dose_slope",
            "coupling_fidelity",
        ],
    )
    return {
        "fit": fit,
        "task_df": task_df,
        "rotation_df": rotation_df,
        "param_df": param_df,
        "participant_metrics_df": participant_metrics_df,
        "final_mag_df": final_mag_df,
        "final_target_df": final_target_df,
        "merged_df": merged_df,
        "correlation_df": corr_df,
        "regression_df": reg_df,
    }


def _annotate_rho(ax, x, y, rho, p, n, *, loc=(0.03, 0.97), color="black"):
    txt = f"rho={rho:.2f}\np={p:.3g}\nn={n}"
    ax.text(
        loc[0],
        loc[1],
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
    )


def plot_wildcard_behavior_summary(
    final_mag_df: pd.DataFrame,
    final_target_df: pd.DataFrame,
    participant_metrics_df: pd.DataFrame,
    *,
    title: str = "Wildcard human learning summary",
) -> plt.Figure:
    """Create a summary figure for the Wildcard human behavior."""
    set_wildcard_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))

    sns.pointplot(
        data=final_mag_df,
        x="magnitude",
        y="final_comp",
        errorbar=("ci", 95),
        color=PALETTE["primary"],
        ax=axes[0],
    )
    sns.stripplot(
        data=final_mag_df,
        x="magnitude",
        y="final_comp",
        color=PALETTE["primary"],
        alpha=0.35,
        size=4,
        ax=axes[0],
    )
    for mag in sorted(final_mag_df["magnitude"].dropna().unique()):
        axes[0].axhline(mag, ls=":", lw=0.8, color="gray", alpha=0.35)
    axes[0].axhline(0, ls="--", lw=0.8, color="black", alpha=0.5)
    axes[0].set_title("Dose response")
    axes[0].set_xlabel("Perturbation magnitude")
    axes[0].set_ylabel("Final-block compensation")

    paired = participant_metrics_df[["participantNum", "trained_mean", "catch_mean"]].dropna()
    for _, row in paired.iterrows():
        axes[1].plot(
            [0, 1], [row["catch_mean"], row["trained_mean"]], color="0.75", lw=0.8, alpha=0.6
        )
    axes[1].scatter(
        np.zeros(len(paired)),
        paired["catch_mean"],
        color=PALETTE["catch"],
        s=28,
        alpha=0.8,
        label="Catch",
    )
    axes[1].scatter(
        np.ones(len(paired)),
        paired["trained_mean"],
        color=PALETTE["trained"],
        s=28,
        alpha=0.8,
        label="Trained mean",
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Catch", "Trained"])
    axes[1].set_ylabel("Final-block compensation")
    axes[1].set_title("Catch isolation")
    axes[1].legend(frameon=False, loc="upper left")

    perm = participant_metrics_df[["actual_mae", "perm_mae_mean", "coupling_fidelity"]].dropna()
    sns.scatterplot(
        data=perm,
        x="actual_mae",
        y="perm_mae_mean",
        color=PALETTE["fidelity"],
        s=45,
        ax=axes[2],
    )
    lo = float(np.nanmin([perm["actual_mae"].min(), perm["perm_mae_mean"].min()]))
    hi = float(np.nanmax([perm["actual_mae"].max(), perm["perm_mae_mean"].max()]))
    axes[2].plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
    axes[2].set_xlabel("Actual MAE")
    axes[2].set_ylabel("Mean shuffled MAE")
    axes[2].set_title("Coupling fidelity")
    axes[2].text(
        0.03,
        0.97,
        f"mean delta={perm['coupling_fidelity'].mean():.2f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
    )

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_wildcard_parameter_map(
    merged_df: pd.DataFrame,
    *,
    title: str = "Wildcard learning ability mapped onto fitted parameters",
) -> plt.Figure:
    """Create a parameter-mapping figure for the Wildcard individual differences."""
    set_wildcard_style()
    specs = [
        ("trained_mean", "sigma_T", "Trained compensation vs translation scale"),
        ("trained_vs_catch_gap", "alpha_other_minus_self", "Catch gap vs other-self bias"),
        ("coupling_fidelity", "alpha_other_minus_self", "Coupling fidelity vs other-self bias"),
        ("dose_slope", "logVisCoeff", "Dose slope vs visual noise"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    axes = axes.flatten()

    for ax, (metric, param, subtitle) in zip(axes, specs):
        sub = merged_df[[metric, param]].dropna()
        sns.regplot(
            data=sub,
            x=param,
            y=metric,
            ax=ax,
            scatter_kws={"s": 42, "alpha": 0.75, "color": PALETTE["primary"]},
            line_kws={"color": PALETTE["secondary"], "lw": 2},
        )
        if len(sub) >= 5:
            rho, p = stats.spearmanr(sub[param], sub[metric])
            _annotate_rho(ax, sub[param], sub[metric], rho, p, len(sub), color="black")
        ax.set_title(subtitle)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_wildcard_regression_heatmap(
    corr_df: pd.DataFrame, *, title: str = "Wildcard correlation map"
) -> plt.Figure:
    """Create a compact heatmap of the correlation table."""
    set_wildcard_style()
    if corr_df.empty:
        raise ValueError("corr_df is empty")
    pivot = corr_df.pivot_table(index="metric", columns="param", values="r", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12.0, max(3.5, 0.45 * len(pivot.index))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Spearman rho"},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def build_wildcard_manuscript_figures(
    bundle: dict,
    output_dir: os.PathLike | str,
    *,
    prefix: str = "wildcard_manuscript",
) -> dict:
    """Build and save the main Wildcard manuscript figures."""
    out_dir = _ensure_dir(output_dir)
    outputs = {}

    fig1 = plot_wildcard_behavior_summary(
        bundle["final_mag_df"],
        bundle["final_target_df"],
        bundle["participant_metrics_df"],
    )
    outputs["behavior"] = save_figure_triplet(fig1, out_dir / f"{prefix}_behavior_summary")
    plt.close(fig1)

    fig2 = plot_wildcard_parameter_map(bundle["merged_df"])
    outputs["parameter_map"] = save_figure_triplet(fig2, out_dir / f"{prefix}_parameter_map")
    plt.close(fig2)

    fig3 = plot_wildcard_regression_heatmap(bundle["correlation_df"])
    outputs["correlation_map"] = save_figure_triplet(fig3, out_dir / f"{prefix}_correlation_map")
    plt.close(fig3)

    return outputs


__all__ = [
    "DEFAULT_FINAL_BLOCK",
    "FREE_PARAM_NAMES",
    "MODEL_PARAM_ORDER",
    "TRANSFORMED_PARAM_NAMES",
    "build_wildcard_analysis_bundle",
    "build_wildcard_human_metrics",
    "build_wildcard_manuscript_figures",
    "build_wildcard_rotation_phase_table",
    "correlation_summary",
    "extract_wildcard_param_table",
    "focused_regression_summary",
    "load_wildcard_fits",
    "load_wildcard_task",
    "merge_wildcard_metrics_and_params",
    "plot_wildcard_behavior_summary",
    "plot_wildcard_parameter_map",
    "plot_wildcard_regression_heatmap",
    "save_figure_triplet",
    "set_wildcard_style",
]
