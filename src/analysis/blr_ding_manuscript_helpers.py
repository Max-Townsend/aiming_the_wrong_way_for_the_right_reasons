"""Ding parameter, structural-belief and generalisation analyses.

Fits are keyed by conditions such as Inner_2T_90 or Outer_8T_315.
Generalisation trials are marked in fit.trialStatuses. Human compensation
uses aim_signed when available, otherwise aim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from scipy.special import logsumexp
from statsmodels.formula.api import ols
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


def _set_default_plot_style() -> None:
    """Apply a manuscript-friendly plotting style."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.setdefault("figure.dpi", 120)
    plt.rcParams.setdefault("savefig.dpi", 300)
    plt.rcParams.setdefault("axes.spines.top", False)
    plt.rcParams.setdefault("axes.spines.right", False)
    plt.rcParams.setdefault("font.size", 11)


def load_ding_individual_fits(path: str | Path = "dingBLRDing.npy"):
    """Load the saved Ding individual fits as a condition -> FitShell dict."""

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    elif (
        isinstance(loaded, np.ndarray)
        and loaded.size == 1
        and hasattr(loaded[0], "participantNums")
    ):
        loaded = loaded[0]
    return loaded


def _free_param_record(xs: Sequence[float]) -> Dict[str, float]:
    xs = list(xs)
    if len(xs) < 9:
        raise ValueError(f"Expected 9 fitted values, got {len(xs)}")
    return {name: float(value) for name, value in zip(FREE_PARAM_NAMES, xs[2:9])}


def build_ding_param_table(ding_fit_dict) -> pd.DataFrame:
    """Build a participant-level table of the seven free BHT parameters."""

    rows: List[Dict[str, object]] = []
    for condition, fit in ding_fit_dict.items():
        for participant_num, xs in zip(fit.participantNums, fit.xs):
            info = fit.participantInfo[participant_num]
            condition_name = info.get("condition", condition)
            rows.append(
                {
                    "participantNum": participant_num,
                    "condition": condition_name,
                    "geometry": info.get("inner_outer"),
                    "targets": f"{int(info.get('numTargets'))}T",
                    "gen_target": condition_name.split("_")[2],
                    "rotation_group": condition_name.split("_")[2],
                    "scale_S0": float(xs[0]),
                    "nu_S0": float(xs[1]),
                    **_free_param_record(xs),
                }
            )
    return pd.DataFrame(rows)


def test_log_prior_geometry_by_target_count(
    param_df: pd.DataFrame,
    param_name: str = "logPriorOddsStruct",
    correction: str = "holm",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test geometry effects separately within 2T and 8T.

    Returns
    -------
    summary_df:
        One row per target-count simple effect.
    nuisance_df:
        A combined table from a small nuisance model ``param ~ geometry + rotation_group``
        within each target count. This is useful as a robustness check.
    """

    summary_rows: List[Dict[str, object]] = []
    nuisance_rows: List[pd.DataFrame] = []

    for targets in ["2T", "8T"]:
        sub = param_df.loc[param_df["targets"] == targets].copy()
        inner = sub.loc[sub["geometry"] == "Inner", param_name].astype(float)
        outer = sub.loc[sub["geometry"] == "Outer", param_name].astype(float)
        t_res = stats.ttest_ind(inner, outer, equal_var=True)

        summary_rows.append(
            {
                "targets": targets,
                "n_inner": int(len(inner)),
                "n_outer": int(len(outer)),
                "mean_inner": float(inner.mean()),
                "mean_outer": float(outer.mean()),
                "diff_inner_minus_outer": float(inner.mean() - outer.mean()),
                "t_stat": float(t_res.statistic),
                "p_unc": float(t_res.pvalue),
            }
        )

        sub["rotation_group"] = sub["rotation_group"].astype(str)
        nuisance_model = ols(f"{param_name} ~ C(geometry) + C(rotation_group)", data=sub).fit()
        nuisance_anova = (
            sm.stats.anova_lm(nuisance_model, typ=2)
            .reset_index()
            .rename(columns={"index": "effect"})
        )
        nuisance_anova.insert(0, "targets", targets)
        nuisance_anova.insert(1, "parameter", param_name)
        nuisance_rows.append(nuisance_anova)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["p_" + correction] = multipletests(summary_df["p_unc"], method=correction)[1]
    nuisance_df = pd.concat(nuisance_rows, ignore_index=True)
    return summary_df, nuisance_df


def _extract_rotation_phase_probabilities(
    fit, participant_num: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return trialwise p(rotation), p(translation), trial status, and df for a participant."""

    pred_state = fit.predState[participant_num]
    log_weights = np.asarray(pred_state["predStructLogW"], dtype=float)
    probs = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    p_trans = probs[:, 0]
    p_rot = probs[:, 1]

    df_pp = (
        fit.df.loc[fit.df["participantNum"] == participant_num]
        .sort_values("trial_number")
        .reset_index(drop=True)
    )
    statuses = np.asarray(fit.trialStatuses[participant_num])
    if len(df_pp) > len(statuses):
        df_pp = df_pp.iloc[: len(statuses)].copy()
    if len(df_pp) != len(statuses):
        raise ValueError(
            f"Participant {participant_num} has {len(df_pp)} fitted rows but {len(statuses)} trial statuses"
        )
    df_pp["trialStatus"] = statuses
    return p_rot, p_trans, statuses, df_pp


def build_ding_structure_phase_table(ding_fit_dict, first_n: int = 4) -> pd.DataFrame:
    """Compute participant-level model structure summaries for the early rotation phase."""

    rows: List[Dict[str, object]] = []
    for condition, fit in ding_fit_dict.items():
        geometry, targets, _rotation_group = condition.split("_")
        for participant_num in fit.participantNums:
            p_rot, p_trans, _statuses, df_pp = _extract_rotation_phase_probabilities(
                fit, participant_num
            )
            rot_idx = np.flatnonzero(df_pp["rotation"].to_numpy(dtype=float) != 0)
            if len(rot_idx) < first_n:
                continue
            first = rot_idx[:first_n]
            rows.append(
                {
                    "participantNum": participant_num,
                    "condition": condition,
                    "geometry": geometry,
                    "targets": targets,
                    "p_rotation_first4": float(np.nanmean(p_rot[first])),
                    "p_translation_first4": float(np.nanmean(p_trans[first])),
                }
            )
    return pd.DataFrame(rows)


def build_ding_generalisation_alignment_table(ding_fit_dict) -> pd.DataFrame:
    """Align model P(trans) and human sign-flip proxy on the same generalisation trials.

    The human sign-flip proxy is defined using only preceding feedback training trials.
    For each no-feedback generalisation trial, we infer the current trained sign from
    the median sign of all prior feedback trials in the rotation phase.
    """

    rows: List[Dict[str, object]] = []
    for condition, fit in ding_fit_dict.items():
        geometry, targets, gen_target = condition.split("_")
        for participant_num in fit.participantNums:
            p_rot, p_trans, _statuses, df_pp = _extract_rotation_phase_probabilities(
                fit, participant_num
            )
            aim_col = "aim_signed" if "aim_signed" in df_pp.columns else "aim"
            gen_mask = (df_pp["rotation"] != 0) & (df_pp["trialStatus"] == "generalisation")
            gen_indices = np.flatnonzero(gen_mask.to_numpy())
            if len(gen_indices) == 0:
                continue

            for gen_order, trial_idx in enumerate(gen_indices, start=1):
                prev = df_pp.iloc[:trial_idx]
                prev_train = prev[
                    (prev["rotation"] != 0)
                    & (prev["trialStatus"] != "generalisation")
                    & (prev["Cursor FB"] != "no_fb")
                ]
                if prev_train.empty:
                    continue

                prev_sign = float(np.sign(np.nanmedian(prev_train[aim_col].to_numpy(dtype=float))))
                if prev_sign == 0:
                    prev_sign = float(
                        np.sign(np.nanmean(prev_train[aim_col].to_numpy(dtype=float)))
                    )
                if prev_sign == 0:
                    continue

                human_sign = float(np.sign(float(df_pp.iloc[trial_idx][aim_col])))
                if human_sign == 0:
                    human_sign_flip = np.nan
                else:
                    human_sign_flip = float(human_sign != prev_sign)

                rows.append(
                    {
                        "participantNum": participant_num,
                        "condition": condition,
                        "geometry": geometry,
                        "targets": targets,
                        "gen_target": gen_target,
                        "gen_trial_index": gen_order,
                        "trial_idx": trial_idx,
                        "p_trans_model": float(p_trans[trial_idx]),
                        "p_rot_model": float(p_rot[trial_idx]),
                        "human_sign_flip": human_sign_flip,
                        "human_rot_proxy": np.nan
                        if np.isnan(human_sign_flip)
                        else float(1.0 - human_sign_flip),
                        "train_sign": prev_sign,
                        "human_sign": human_sign,
                        "aim_value": float(df_pp.iloc[trial_idx][aim_col]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_ding_generalisation_alignment(
    alignment_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the alignment table at the participant and exposure-by-condition levels."""

    participant_rows: List[Dict[str, object]] = []
    for (participant_num, geometry, targets), sub in alignment_df.groupby(
        ["participantNum", "geometry", "targets"]
    ):
        sub = sub.sort_values("gen_trial_index")
        p_trans = sub["p_trans_model"].to_numpy(dtype=float)
        human = sub["human_sign_flip"].to_numpy(dtype=float)
        if len(p_trans) < 4:
            continue
        thirds = np.array_split(np.arange(len(p_trans)), 3)
        participant_rows.append(
            {
                "participantNum": participant_num,
                "geometry": geometry,
                "targets": targets,
                "model_ptrans_first4": float(np.nanmean(p_trans[:4])),
                "human_flip_first4": float(np.nanmean(human[:4])),
                "model_ptrans_mid": float(np.nanmean(p_trans[thirds[1]])),
                "human_flip_mid": float(np.nanmean(human[thirds[1]])),
                "model_ptrans_late": float(np.nanmean(p_trans[thirds[2]])),
                "human_flip_late": float(np.nanmean(human[thirds[2]])),
            }
        )

    participant_df = pd.DataFrame(participant_rows)
    exposure_df = (
        alignment_df.groupby(["geometry", "targets", "gen_trial_index"], as_index=False)
        .agg(
            mean_p_trans_model=("p_trans_model", "mean"),
            mean_p_rot_model=("p_rot_model", "mean"),
            mean_human_sign_flip=("human_sign_flip", "mean"),
            mean_human_rot_proxy=("human_rot_proxy", "mean"),
            n=("participantNum", "nunique"),
        )
        .sort_values(["geometry", "targets", "gen_trial_index"])
        .reset_index(drop=True)
    )
    return participant_df, exposure_df


def figure_ding_structure_and_alignment(
    structure_df: pd.DataFrame,
    alignment_exposure_df: pd.DataFrame,
    save_dir: Optional[str | Path] = None,
):
    """Create a two-panel manuscript figure linking model P(trans) to human sign-flip."""

    _set_default_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    plot_df = structure_df.melt(
        id_vars=["participantNum", "geometry", "targets"],
        value_vars=["p_rotation_first4", "p_translation_first4"],
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "p_rotation_first4": "P(rotation) first 4",
            "p_translation_first4": "P(translation) first 4",
        }
    )
    sns.pointplot(
        data=plot_df,
        x="targets",
        y="value",
        hue="geometry",
        dodge=True,
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_title("Early structure evidence")
    axes[0].set_xlabel("Target count")
    axes[0].set_ylabel("Model probability")

    sns.lineplot(
        data=alignment_exposure_df,
        x="gen_trial_index",
        y="mean_p_trans_model",
        hue="geometry",
        style="targets",
        markers=True,
        dashes=False,
        ax=axes[1],
    )
    sns.lineplot(
        data=alignment_exposure_df,
        x="gen_trial_index",
        y="mean_human_sign_flip",
        hue="geometry",
        style="targets",
        markers=True,
        dashes=True,
        ax=axes[1],
        legend=False,
        alpha=0.9,
    )
    axes[1].set_title("Aligned model-human generalisation")
    axes[1].set_xlabel("Generalisation trial index")
    axes[1].set_ylabel("Mean probability / sign-flip rate")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if save_dir is not None:
        save_figure_bundle(fig, Path(save_dir) / "ding_structure_alignment")
    return fig, axes


def figure_ding_logprior_simple_effects(
    simple_effect_df: pd.DataFrame, save_dir: Optional[str | Path] = None
):
    """Create a compact figure for the geometry effect on logPriorOddsStruct."""

    _set_default_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    sns.barplot(
        data=simple_effect_df,
        x="targets",
        y="diff_inner_minus_outer",
        color="#4c78a8",
        ax=ax,
    )
    sns.stripplot(
        data=simple_effect_df,
        x="targets",
        y="diff_inner_minus_outer",
        color="black",
        size=7,
        ax=ax,
    )
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title("Geometry simple effects on structure prior odds")
    ax.set_xlabel("Target count")
    ax.set_ylabel("Inner minus Outer")
    fig.tight_layout()

    if save_dir is not None:
        save_figure_bundle(fig, Path(save_dir) / "ding_logprior_simple_effects")
    return fig, ax


def save_figure_bundle(fig, base_path: str | Path) -> Dict[str, str]:
    """Save a figure as PNG, SVG, and PDF using the same base path."""

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in [".png", ".svg", ".pdf"]:
        out_path = base.with_suffix(ext)
        fig.savefig(out_path, bbox_inches="tight")
        outputs[ext.lstrip(".")] = str(out_path)
    return outputs


def run_ding_manuscript_pipeline(
    fits_path: str | Path = "dingBLRDing.npy",
    save_dir: Optional[str | Path] = None,
):
    """Convenience wrapper that computes all Ding manuscript tables and figures."""

    ding_fits = load_ding_individual_fits(fits_path)
    param_df = build_ding_param_table(ding_fits)
    simple_effect_df, nuisance_df = test_log_prior_geometry_by_target_count(param_df)
    structure_df = build_ding_structure_phase_table(ding_fits)
    alignment_df = build_ding_generalisation_alignment_table(ding_fits)
    participant_alignment_df, exposure_alignment_df = summarize_ding_generalisation_alignment(
        alignment_df
    )

    figures = {}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1, _ = figure_ding_logprior_simple_effects(simple_effect_df, save_dir=save_dir)
        fig2, _ = figure_ding_structure_and_alignment(
            structure_df, exposure_alignment_df, save_dir=save_dir
        )
        figures = {
            "logprior": fig1,
            "alignment": fig2,
        }

    return {
        "fits": ding_fits,
        "param_df": param_df,
        "simple_effect_df": simple_effect_df,
        "nuisance_df": nuisance_df,
        "structure_df": structure_df,
        "alignment_df": alignment_df,
        "participant_alignment_df": participant_alignment_df,
        "exposure_alignment_df": exposure_alignment_df,
        "figures": figures,
    }


if __name__ == "__main__":
    out = run_ding_manuscript_pipeline(save_dir=None)
    print(out["simple_effect_df"].to_string(index=False))
