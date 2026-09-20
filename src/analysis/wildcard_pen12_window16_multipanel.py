from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.special import logsumexp

from recover_true_self_other_weights import extract_fit_shells


WINDOW_CYCLES = 16
PEN = 1.2
ROOT = Path(__file__).resolve().parent
FIT_PATH = ROOT / "BHTWildcard.npy"
RECOVERED_PATH = ROOT / "BHTWildcard_true_self_other.pkl"
OUT_FIG = ROOT / "tempFigures" / "wildcard_pen12_window16_multipanel.png"
OUT_CSV = ROOT / "tempFigures" / "wildcard_pen12_window16_multipanel_points.csv"


def first_interior_breakpoint(signal: Iterable[float], pen: float) -> Optional[int]:
    arr = np.asarray(list(signal), dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return None
    arr = arr[valid].reshape(-1, 1)
    original_idx = np.arange(len(signal))[valid]
    if np.std(arr) == 0:
        return None
    algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(arr)
    bkps = algo.predict(pen=pen)
    interiors = [original_idx[b] for b in bkps[:-1]] if bkps[:-1] else []
    return int(interiors[0]) if interiors else None


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = nan"
    if p < 1e-4:
        return "p < 1e-4"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def load_fit_and_recovery() -> Tuple[object, Dict]:
    loaded = np.load(FIT_PATH, allow_pickle=True)
    fit = extract_fit_shells(loaded)[0]
    with RECOVERED_PATH.open("rb") as fh:
        recovered = pickle.load(fh)[0]
    return fit, recovered


def compute_trialwise_model_mean(
    pred_state_pp: Dict[str, np.ndarray],
    arr_idx: np.ndarray,
) -> np.ndarray:
    pred_logp = np.asarray(pred_state_pp["predCompLogProb"])
    pred_mean = np.asarray(pred_state_pp["predCompMean"])
    pred_n = np.asarray(pred_state_pp["predNComp"]).astype(int)

    out = np.full(len(arr_idx), np.nan, dtype=float)
    for ii, t in enumerate(arr_idx):
        n_comp = int(pred_n[t])
        if n_comp <= 0:
            out[ii] = 0.0
            continue
        lp = pred_logp[t, :n_comp]
        mu = pred_mean[t, :n_comp]
        w = np.exp(lp - logsumexp(lp))
        out[ii] = float(np.sum(w * mu))
    return out


def wrap_angle_deg(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def compute_trialwise_model_abs_comp(
    pred_state_pp: Dict[str, np.ndarray],
    arr_idx: np.ndarray,
    n_draws: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    pred_logp = np.asarray(pred_state_pp["predCompLogProb"])
    pred_mean = np.asarray(pred_state_pp["predCompMean"])
    pred_scale = np.asarray(pred_state_pp["predCompScale"])
    pred_nu = np.asarray(pred_state_pp["predCompNu"])
    pred_n = np.asarray(pred_state_pp["predNComp"]).astype(int)

    out = np.full(len(arr_idx), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for ii, t in enumerate(arr_idx):
        n_comp = int(pred_n[t])
        if n_comp <= 0:
            out[ii] = 0.0
            continue
        lp = pred_logp[t, :n_comp]
        w = np.exp(lp - logsumexp(lp))
        comp_idx = rng.choice(n_comp, size=n_draws, p=w)
        mu = pred_mean[t, :n_comp][comp_idx]
        scale = np.clip(pred_scale[t, :n_comp][comp_idx], 1e-9, None)
        nu = np.clip(pred_nu[t, :n_comp][comp_idx], 1e-6, None)
        draws = mu + scale * rng.standard_t(df=nu, size=n_draws)
        out[ii] = float(np.mean(np.abs(wrap_angle_deg(draws))))
    return out


def fit_line_with_ci(
    x: np.ndarray, y: np.ndarray, grid: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    if grid is None:
        grid = np.linspace(x.min(), x.max(), 200)
    grid_X = sm.add_constant(grid)
    pred = model.get_prediction(grid_X).summary_frame(alpha=0.05)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    return {
        "grid": grid,
        "mean": pred["mean"].to_numpy(dtype=float),
        "ci_lo": pred["mean_ci_lower"].to_numpy(dtype=float),
        "ci_hi": pred["mean_ci_upper"].to_numpy(dtype=float),
        "r": pearson_r,
        "p": pearson_p,
        "rho": spearman_rho,
        "rho_p": spearman_p,
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "model": model,
    }


def build_analysis_table(
    fit: object, recovered: Dict, window_cycles: int = WINDOW_CYCLES
) -> pd.DataFrame:
    all_df = fit.df.copy().reset_index(drop=True)
    rows = []

    for pp, part in recovered["participants"].items():
        pp = int(pp)
        pdf = all_df[all_df["participantNum"] == pp].copy().reset_index(drop=True)
        phase = np.asarray(part["phase"])
        if len(pdf) < len(phase):
            rows.append({"pp": pp, "status": "length_mismatch"})
            continue
        if len(pdf) != len(phase):
            pdf = pdf.iloc[: len(phase)].copy().reset_index(drop=True)

        pdf["phase_rec"] = phase
        pdf["rotation_rec"] = np.asarray(part["rotation"], dtype=float)
        pdf["other_true"] = np.asarray(part["m1_other_weight"], dtype=float)

        baseline = pdf[pdf["phase_rec"].astype(str).str.lower() == "baseline"].copy()
        rotation = (
            pdf[pdf["phase_rec"].astype(str).str.lower() == "rotation"]
            .copy()
            .reset_index(drop=True)
        )
        if rotation.empty:
            rows.append({"pp": pp, "status": "no_rotation"})
            continue

        rotation["rot_trial_idx"] = np.arange(len(rotation))
        rotation["cycle"] = rotation["rot_trial_idx"] // 4
        onset_signal = (
            rotation.loc[rotation["rotation_rec"] != 0]
            .groupby("cycle")["aim"]
            .apply(lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float)))))
            .sort_index()
        )
        onset = first_interior_breakpoint(onset_signal.to_numpy(dtype=float), PEN)
        if onset is None:
            rows.append({"pp": pp, "status": "no_onset"})
            continue

        selected = rotation[
            (rotation["cycle"] >= onset) & (rotation["cycle"] < onset + window_cycles)
        ].copy()
        nonzero = selected[selected["rotation_rec"] != 0].copy()
        if nonzero.empty:
            rows.append({"pp": pp, "status": "no_nonzero", "onset_cycle": onset})
            continue

        mean_abs_comp = float(np.mean(np.abs(nonzero["aim"].to_numpy(dtype=float))))
        status = "included" if mean_abs_comp >= 5.0 else "excluded_low_comp"

        by_rot = (
            selected.groupby("rotation_rec", as_index=False)["aim"]
            .mean()
            .sort_values("rotation_rec")
        )
        if by_rot["rotation_rec"].nunique() < 2:
            rows.append({"pp": pp, "status": "insufficient_levels", "onset_cycle": onset})
            continue
        discrim = float(
            -np.polyfit(
                by_rot["rotation_rec"].to_numpy(dtype=float),
                by_rot["aim"].to_numpy(dtype=float),
                1,
            )[0]
        )

        rotation_positions = np.where(np.array([str(x).lower() == "rotation" for x in phase]))[0]
        arr_idx = rotation_positions[selected["rot_trial_idx"].to_numpy(dtype=int)]
        model_abs_comp = compute_trialwise_model_abs_comp(
            fit.predState[pp],
            arr_idx,
            seed=1_000 + int(pp),
        )

        mean_abs_rot = float(np.mean(np.abs(nonzero["rotation_rec"].to_numpy(dtype=float))))
        true_other = float(np.nanmean(selected["other_true"].to_numpy(dtype=float)))
        human_abs_comp = float(np.mean(np.abs(nonzero["aim"].to_numpy(dtype=float))))
        model_abs_comp_nonzero = float(
            np.mean(model_abs_comp[selected["rotation_rec"].to_numpy(dtype=float) != 0])
        )
        base_abs = (
            float(np.nanmean(np.abs(baseline["aim"].to_numpy(dtype=float))))
            if len(baseline)
            else np.nan
        )
        delta_abs_norm = (
            (human_abs_comp - base_abs) / mean_abs_rot
            if np.isfinite(base_abs) and mean_abs_rot != 0
            else np.nan
        )
        human_comp_norm = human_abs_comp / mean_abs_rot if mean_abs_rot != 0 else np.nan
        model_comp_norm = model_abs_comp_nonzero / mean_abs_rot if mean_abs_rot != 0 else np.nan

        rot_means = selected.groupby("rotation_rec")["aim"].mean().to_dict()
        residuals = selected["aim"].to_numpy(dtype=float) - selected["rotation_rec"].map(
            rot_means
        ).to_numpy(dtype=float)
        resid_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan
        resid_sd_norm = (
            resid_sd / mean_abs_rot if np.isfinite(resid_sd) and mean_abs_rot != 0 else np.nan
        )

        rows.append(
            {
                "pp": pp,
                "status": status,
                "onset_cycle": onset,
                "true_other": true_other,
                "disc": discrim,
                "mean_abs_rot": mean_abs_rot,
                "mean_abs_comp": mean_abs_comp,
                "human_comp_norm": human_comp_norm,
                "model_comp_norm": model_comp_norm,
                "base_abs": base_abs,
                "delta_abs_norm": delta_abs_norm,
                "resid_sd_norm": resid_sd_norm,
                "n_rotation_trials": len(selected),
                "n_nonzero_trials": len(nonzero),
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_panel_a_table(fit: object, recovered: Dict) -> pd.DataFrame:
    all_df = fit.df.copy().reset_index(drop=True)
    rows = []

    for pp in fit.participantNums:
        pp = int(pp)
        pdat = all_df[all_df["participantNum"] == pp].copy().reset_index(drop=True)
        if pdat.empty:
            continue

        aims = pdat["aim"].to_numpy(dtype=float)
        rots = pdat["rotation"].to_numpy(dtype=float)
        rot_mask = rots != 0
        if int(rot_mask.sum()) < 20:
            continue

        ra = aims[rot_mask]
        rr = rots[rot_mask]
        vf = np.isfinite(ra)
        if int(vf.sum()) < 5:
            continue

        maa = float(np.nanmean(np.abs(ra[vf])))
        sc = float(np.nanmean(-ra[vf] / rr[vf]))
        learner = bool(maa >= 10.0)

        pred = compute_trialwise_model_mean(fit.predState[pp], np.arange(len(aims)))
        ms = float(np.nanmean(-pred[rot_mask][vf] / rr[vf]))

        mean_other_rot = np.nan
        if pp in recovered["participants"]:
            part = recovered["participants"][pp]
            phase = np.asarray(part["phase"])
            other = np.asarray(part["m1_other_weight"], dtype=float)
            n = min(len(phase), len(other), len(rots))
            if n > 0:
                rot_phase_mask = np.array([str(x).lower() == "rotation" for x in phase[:n]]) & (
                    rots[:n] != 0
                )
                if np.any(rot_phase_mask):
                    mean_other_rot = float(np.nanmean(other[:n][rot_phase_mask]))

        rows.append(
            {
                "pp": pp,
                "maa": maa,
                "sc": sc,
                "ms": ms,
                "learner": learner,
                "other_rot_mean": mean_other_rot,
            }
        )

    return pd.DataFrame(rows)


def build_profile_table(
    fit: object,
    recovered: Dict,
    learner_pp: Iterable[int],
    window_cycles: int = WINDOW_CYCLES,
) -> pd.DataFrame:
    all_df = fit.df.copy().reset_index(drop=True)
    rows = []

    for pp in learner_pp:
        pp = int(pp)
        part = recovered["participants"][pp]
        pdf = all_df[all_df["participantNum"] == pp].copy().reset_index(drop=True)
        phase = np.asarray(part["phase"])
        if len(pdf) != len(phase):
            pdf = pdf.iloc[: len(phase)].copy().reset_index(drop=True)
        pdf["phase_rec"] = phase
        pdf["rotation_rec"] = np.asarray(part["rotation"], dtype=float)

        baseline = pdf[pdf["phase_rec"].astype(str).str.lower() == "baseline"].copy()
        base_abs = (
            float(np.nanmean(np.abs(baseline["aim"].to_numpy(dtype=float))))
            if len(baseline)
            else np.nan
        )

        rotation = (
            pdf[pdf["phase_rec"].astype(str).str.lower() == "rotation"]
            .copy()
            .reset_index(drop=True)
        )
        if rotation.empty:
            continue
        rotation["rot_trial_idx"] = np.arange(len(rotation))
        rotation["cycle"] = rotation["rot_trial_idx"] // 4
        onset_signal = (
            rotation.loc[rotation["rotation_rec"] != 0]
            .groupby("cycle")["aim"]
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .sort_index()
        )
        onset = first_interior_breakpoint(onset_signal.to_numpy(dtype=float), PEN)
        if onset is None:
            continue

        selected = rotation[
            (rotation["cycle"] >= onset) & (rotation["cycle"] < onset + window_cycles)
        ].copy()
        if selected.empty:
            continue
        selected["mag"] = np.abs(selected["rotation_rec"])
        by_mag = (
            selected.groupby("mag")["aim"]
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .reset_index(name="mean_abs_aim")
        )
        for _, row in by_mag.iterrows():
            rows.append(
                {
                    "pp": pp,
                    "mag": float(row["mag"]),
                    "mean_abs_aim": float(row["mean_abs_aim"]),
                    "base_abs": base_abs,
                }
            )
    return pd.DataFrame(rows)


def build_model_profile_table(
    fit: object,
    recovered: Dict,
    learner_pp: Iterable[int],
    window_cycles: int = WINDOW_CYCLES,
) -> pd.DataFrame:
    all_df = fit.df.copy().reset_index(drop=True)
    rows = []

    for pp in learner_pp:
        pp = int(pp)
        part = recovered["participants"][pp]
        pdf = all_df[all_df["participantNum"] == pp].copy().reset_index(drop=True)
        phase = np.asarray(part["phase"])
        if len(pdf) != len(phase):
            pdf = pdf.iloc[: len(phase)].copy().reset_index(drop=True)
        pdf["phase_rec"] = phase
        pdf["rotation_rec"] = np.asarray(part["rotation"], dtype=float)

        model_abs = compute_trialwise_model_abs_comp(
            fit.predState[pp],
            np.arange(len(pdf), dtype=int),
            seed=10_000 + int(pp),
        )

        baseline_mask = pdf["phase_rec"].astype(str).str.lower() == "baseline"
        base_abs = (
            float(np.nanmean(model_abs[baseline_mask.to_numpy()]))
            if baseline_mask.any()
            else np.nan
        )

        rotation = (
            pdf[pdf["phase_rec"].astype(str).str.lower() == "rotation"]
            .copy()
            .reset_index(drop=True)
        )
        if rotation.empty:
            continue
        rotation["rot_trial_idx"] = np.arange(len(rotation))
        rotation["cycle"] = rotation["rot_trial_idx"] // 4
        onset_signal = (
            rotation.loc[rotation["rotation_rec"] != 0]
            .groupby("cycle")["aim"]
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .sort_index()
        )
        onset = first_interior_breakpoint(onset_signal.to_numpy(dtype=float), PEN)
        if onset is None:
            continue

        selected = rotation[
            (rotation["cycle"] >= onset) & (rotation["cycle"] < onset + window_cycles)
        ].copy()
        if selected.empty:
            continue
        selected["mag"] = np.abs(selected["rotation_rec"])
        rotation_positions = np.where(np.array([str(x).lower() == "rotation" for x in phase]))[0]
        arr_idx = rotation_positions[selected["rot_trial_idx"].to_numpy(dtype=int)]
        model_sel = model_abs[arr_idx]
        selected["model_abs_aim"] = model_sel
        by_mag = (
            selected.groupby("mag")["model_abs_aim"]
            .apply(lambda s: float(np.nanmean(s.to_numpy(dtype=float))))
            .reset_index(name="mean_abs_model_aim")
        )
        for _, row in by_mag.iterrows():
            rows.append(
                {
                    "pp": pp,
                    "mag": float(row["mag"]),
                    "mean_abs_model_aim": float(row["mean_abs_model_aim"]),
                    "base_abs_model": base_abs,
                }
            )

    return pd.DataFrame(rows)


def assign_discrimination_groups(
    disc_df: pd.DataFrame,
    scheme: str = "quartiles",
) -> Tuple[pd.DataFrame, list[str], Dict[str, str]]:
    vals = disc_df["disc"].to_numpy(dtype=float)
    if scheme == "tertiles":
        q_low, q_high = np.quantile(vals, [1 / 3, 2 / 3])
        labels = np.where(
            vals <= q_low,
            "Low discrimination",
            np.where(vals >= q_high, "High discrimination", "Mid discrimination"),
        )
        order = ["Low discrimination", "Mid discrimination", "High discrimination"]
        colors = {
            "Low discrimination": "#d95f02",
            "Mid discrimination": "#6f6f6f",
            "High discrimination": "#1b6ca8",
        }
    elif scheme == "quartiles":
        q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        labels = np.select(
            [
                vals <= q1,
                (vals > q1) & (vals <= q2),
                (vals > q2) & (vals <= q3),
                vals > q3,
            ],
            [
                "Q1 lowest",
                "Q2",
                "Q3",
                "Q4 highest",
            ],
            default="Q2",
        )
        order = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
        cmap_vals = ["#a7dba0", "#74c476", "#31a354", "#006d2c"]
        colors = {
            "Q1 lowest": cmap_vals[0],
            "Q2": cmap_vals[1],
            "Q3": cmap_vals[2],
            "Q4 highest": cmap_vals[3],
        }
    else:
        raise ValueError(f"Unknown grouping scheme: {scheme}")

    out = disc_df[["pp", "disc"]].copy()
    out["disc_group"] = labels
    return out, order, colors


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def plot_scatter_with_excluded(
    ax: plt.Axes,
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    include_identity: bool = False,
    color_by_other: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    annotation_extra: Optional[str] = None,
) -> None:
    ax.set_facecolor("white")

    if len(excluded):
        ax.scatter(
            excluded[x_col],
            excluded[y_col],
            s=34,
            facecolors="none",
            edgecolors="#c5c5c5",
            linewidths=0.95,
            alpha=0.9,
            zorder=2,
        )

    if color_by_other:
        sc = ax.scatter(
            included[x_col],
            included[y_col],
            s=40,
            c=included["true_other"],
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.97,
            zorder=3,
        )
    else:
        sc = ax.scatter(
            included[x_col],
            included[y_col],
            s=40,
            color="#24557a",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.97,
            zorder=3,
        )

    fit = fit_line_with_ci(
        included[x_col].to_numpy(dtype=float), included[y_col].to_numpy(dtype=float)
    )
    ax.fill_between(fit["grid"], fit["ci_lo"], fit["ci_hi"], color="#b24a3a", alpha=0.16, zorder=1)
    ax.plot(fit["grid"], fit["mean"], color="#b24a3a", lw=1.9, zorder=4)

    if include_identity:
        lo = float(
            min(
                np.nanmin(included[[x_col, y_col]].to_numpy()),
                np.nanmin(excluded[[x_col, y_col]].to_numpy()) if len(excluded) else np.inf,
            )
        )
        hi = float(
            max(
                np.nanmax(included[[x_col, y_col]].to_numpy()),
                np.nanmax(excluded[[x_col, y_col]].to_numpy()) if len(excluded) else -np.inf,
            )
        )
        grid = np.linspace(lo, hi, 100)
        ax.plot(grid, grid, color="#7f7a6d", lw=1.2, ls="--", zorder=1)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    lines = [
        f"learners n = {len(included)}",
        f"non-learners n = {len(excluded)}",
        f"r = {fit['r']:.2f}",
        f"$\\rho$ = {fit['rho']:.2f}",
        fmt_p(fit["p"]),
    ]
    if annotation_extra:
        lines.append(annotation_extra)
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d6cfbf", alpha=0.95),
    )

    return sc


def make_multipanel_figure(
    df: pd.DataFrame,
    panel_a_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    model_profile_df: Optional[pd.DataFrame] = None,
    window_cycles: int = WINDOW_CYCLES,
    out_fig: Path = OUT_FIG,
    out_csv: Path = OUT_CSV,
    show_profile_spaghetti: bool = True,
    panel_d_mode: str = "coef",
    show_panel_b: bool = True,
) -> None:
    df = df.merge(panel_a_df[["pp", "learner"]], on="pp", how="left")
    included = df[df["learner"] == True].copy()
    excluded = df[df["learner"] == False].copy()
    out_fig.parent.mkdir(exist_ok=True, parents=True)
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_csv, index=False)

    adj_df = included[np.isfinite(included["delta_abs_norm"])].copy()

    disc_group, group_order, group_colors = assign_discrimination_groups(
        included[["pp", "disc"]], scheme="quartiles"
    )
    profile_df = profile_df.merge(disc_group[["pp", "disc_group"]], on="pp", how="left")
    quartile_other = (
        included.merge(disc_group[["pp", "disc_group"]], on="pp", how="left")
        .groupby("disc_group")["true_other"]
        .mean()
    )

    model_rows = []
    model_specs = [
        ("Unadjusted", ["true_other"]),
        ("+ normalized aim shift", ["true_other", "delta_abs_norm"]),
    ]
    for label, cols in model_specs:
        model_df = included[["disc"] + cols].dropna().copy()
        X = sm.add_constant(model_df[cols])
        res = sm.OLS(model_df["disc"], X).fit()
        ci_lo, ci_hi = res.conf_int().loc["true_other"]
        model_rows.append(
            {
                "label": label,
                "n": len(model_df),
                "coef": float(res.params["true_other"]),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "p": float(res.pvalues["true_other"]),
                "r2": float(res.rsquared),
            }
        )
    model_df_plot = pd.DataFrame(model_rows)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
        }
    )
    fig = plt.figure(figsize=(13.4, 10.4) if show_panel_b else (18.0, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    if show_panel_b:
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 3, wspace=0.28)
        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[0, 2])
    ax1.set_facecolor("white")
    pa_nl = panel_a_df[~panel_a_df["learner"]].copy()
    pa_l = panel_a_df[panel_a_df["learner"]].copy()
    if len(pa_nl):
        ax1.scatter(
            pa_nl["ms"],
            pa_nl["sc"],
            s=34,
            facecolors="none",
            edgecolors="#c5c5c5",
            linewidths=0.95,
            alpha=0.9,
            zorder=2,
        )
    ax1.scatter(
        pa_l["ms"],
        pa_l["sc"],
        s=40,
        color="#24557a",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.97,
        zorder=3,
    )
    fit1 = fit_line_with_ci(pa_l["ms"].to_numpy(dtype=float), pa_l["sc"].to_numpy(dtype=float))
    ax1.fill_between(
        fit1["grid"], fit1["ci_lo"], fit1["ci_hi"], color="#b24a3a", alpha=0.16, zorder=1
    )
    ax1.plot(fit1["grid"], fit1["mean"], color="#b24a3a", lw=1.9, zorder=4)
    lo1 = float(min(panel_a_df[["ms", "sc"]].min().min(), -0.3))
    hi1 = float(max(panel_a_df[["ms", "sc"]].max().max(), 1.3))
    ident = np.linspace(lo1, hi1, 100)
    ax1.plot(ident, ident, color="#7f7a6d", lw=1.2, ls="--", zorder=1)
    ax1.set_title("Model vs Human Compensation", fontsize=9.5, pad=4)
    ax1.set_xlabel("Model-predicted compensation")
    ax1.set_ylabel(r"Human compensation ($-$aim / rot)")
    ax1.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"learners n = {len(pa_l)}",
                f"non-learners n = {len(pa_nl)}",
                f"r = {fit1['r']:.2f}",
                f"$\\rho$ = {fit1['rho']:.2f}",
                fmt_p(fit1["p"]),
            ]
        ),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d6cfbf", alpha=0.95),
    )
    add_panel_label(ax1, "A")

    if show_panel_b:
        ax2 = fig.add_subplot(gs[0, 1])
        plot_scatter_with_excluded(
            ax2,
            included,
            excluded,
            "true_other",
            "disc",
            title="True Other vs Discrimination",
            xlabel="Mean true other weight",
            ylabel=r"Discrimination: $-\,\mathrm{slope}(\overline{aim}\sim rotation)$",
            xlim=(0, 1.0),
            annotation_extra=f"{window_cycles} cycles, pen = {PEN:.1f}",
        )
        ax2.axhline(0, color="#8a8578", lw=1.0, ls="--", zorder=1)
        add_panel_label(ax2, "B")
        label_c = "C"
        label_d = "D"
    else:
        label_c = "B"
        label_d = "C"

    ax3.set_facecolor("white")
    if show_profile_spaghetti:
        for pp, pp_df in profile_df.groupby("pp"):
            pp_df = pp_df.sort_values("mag")
            if len(pp_df) < 2:
                continue
            ax3.plot(
                pp_df["mag"],
                pp_df["mean_abs_aim"],
                color="#cfcfcf",
                lw=0.8,
                alpha=0.5,
                zorder=1,
            )
    pooled_base = included["base_abs"].dropna().to_numpy(dtype=float)
    base_mu = np.nan
    base_ci = np.nan
    if len(pooled_base):
        base_mu = float(np.mean(pooled_base))
        base_ci = (
            1.96 * float(np.std(pooled_base, ddof=1)) / np.sqrt(len(pooled_base))
            if len(pooled_base) > 1
            else 0.0
        )
        ax3.axhspan(base_mu - base_ci, base_mu + base_ci, color="#bdbdbd", alpha=0.22, zorder=0)

    mags = sorted(profile_df["mag"].dropna().unique())
    for grp in group_order:
        gp = profile_df[profile_df["disc_group"] == grp].copy()
        if gp.empty:
            continue
        summary = (
            gp.groupby("mag")["mean_abs_aim"]
            .agg(["mean", "count", "std"])
            .reset_index()
            .sort_values("mag")
        )
        se95 = 1.96 * summary["std"] / np.sqrt(summary["count"])
        ax3.errorbar(
            summary["mag"],
            summary["mean"],
            yerr=se95,
            fmt="o-",
            color=group_colors[grp],
            lw=2.4,
            ms=6.5,
            capsize=3.5,
            zorder=4,
            label=f"{grp} (n = {gp['pp'].nunique()})",
        )

    if mags:
        perfect_x = np.array(mags, dtype=float)
        ax3.plot(perfect_x, perfect_x, "--", color="#8a8578", lw=1.0, zorder=0)
    ax3.set_title("Human Re-Aiming Profiles", fontsize=9.5, pad=4)
    ax3.set_xlabel("Rotation magnitude (°)")
    ax3.set_ylabel("Mean |aim| in onset-aligned window (°)")
    ax3.set_xticks(mags)
    ax3.set_xticklabels([f"{int(m)}°" for m in mags])
    ax3.set_ylim(0, 45)
    ax3.set_yticks([0, 15, 30, 45])
    grp_counts = disc_group["disc_group"].value_counts()
    ax3.text(
        0.98,
        0.97,
        "\n".join(
            [
                f"learners n = {included['pp'].nunique()}",
                "summary overlays by discrimination quartile",
            ]
        ),
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d6cfbf", alpha=0.95),
    )
    legend_handles = [
        Patch(facecolor="#bdbdbd", alpha=0.22, edgecolor="none", label="Baseline |aim| ± 95% CI"),
        Line2D([0], [0], color="#8a8578", lw=1.0, ls="--", label="Perfect discrimination"),
    ]
    if show_profile_spaghetti:
        legend_handles.insert(
            0,
            Line2D(
                [0],
                [0],
                color="#cfcfcf",
                lw=1.2,
                label=f"Individual learners (n = {included['pp'].nunique()})",
            ),
        )
    for grp in group_order:
        legend_handles.insert(
            (1 if show_profile_spaghetti else 0) + group_order.index(grp),
            Line2D(
                [0],
                [0],
                color=group_colors[grp],
                marker="o",
                lw=2.4,
                label=f"{grp} (n = {int(grp_counts.get(grp, 0))})",
            ),
        )
    ax3.legend(handles=legend_handles, loc="lower right", fontsize=7.2, frameon=False)
    add_panel_label(ax3, label_c)

    ax4.set_facecolor("white")
    if panel_d_mode == "coef":
        x = np.arange(len(model_df_plot))
        ax4.errorbar(
            x,
            model_df_plot["coef"],
            yerr=[
                model_df_plot["coef"] - model_df_plot["ci_lo"],
                model_df_plot["ci_hi"] - model_df_plot["coef"],
            ],
            fmt="o",
            color="#24557a",
            ecolor="#24557a",
            elinewidth=2.0,
            capsize=4,
            markersize=7,
            zorder=3,
        )
        ax4.axhline(0, color="#8a8578", lw=1.0, ls="--", zorder=1)
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_df_plot["label"], rotation=12, ha="right")
        ax4.set_ylabel("Coefficient for true other weight\npredicting discrimination")
        ax4.set_title("Other Effect After Aim-Shift Control", fontsize=9.5, pad=4)
        y_span = float((model_df_plot["ci_hi"].max() - model_df_plot["ci_lo"].min()) or 1.0)
        for xi, (_, row) in zip(x, model_df_plot.iterrows()):
            ax4.text(
                xi,
                row["ci_hi"] + 0.06 * y_span,
                f"n = {int(row['n'])}\n{fmt_p(row['p'])}\n95% CI [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]",
                va="bottom",
                ha="center",
                fontsize=7.6,
            )
        ax4.set_ylim(
            model_df_plot["ci_lo"].min() - 0.18 * y_span,
            model_df_plot["ci_hi"].max() + 0.24 * y_span,
        )
    elif panel_d_mode == "model_profile":
        if model_profile_df is None:
            raise ValueError("model_profile_df is required when panel_d_mode='model_profile'")
        model_profile_df = model_profile_df.merge(
            disc_group[["pp", "disc_group"]], on="pp", how="left"
        )
        pooled_model_base = model_profile_df["base_abs_model"].dropna().to_numpy(dtype=float)
        if len(pooled_model_base):
            base_mu_model = float(np.mean(pooled_model_base))
            base_ci_model = (
                1.96 * float(np.std(pooled_model_base, ddof=1)) / np.sqrt(len(pooled_model_base))
                if len(pooled_model_base) > 1
                else 0.0
            )
            ax4.axhspan(
                base_mu_model - base_ci_model,
                base_mu_model + base_ci_model,
                color="#bdbdbd",
                alpha=0.22,
                zorder=0,
            )
        model_mags = sorted(model_profile_df["mag"].dropna().unique())
        for grp in group_order:
            gp = model_profile_df[model_profile_df["disc_group"] == grp].copy()
            if gp.empty:
                continue
            summary = (
                gp.groupby("mag")["mean_abs_model_aim"]
                .agg(["mean", "count", "std"])
                .reset_index()
                .sort_values("mag")
            )
            se95 = 1.96 * summary["std"] / np.sqrt(summary["count"])
            ax4.errorbar(
                summary["mag"],
                summary["mean"],
                yerr=se95,
                fmt="o-",
                color=group_colors[grp],
                lw=2.4,
                ms=6.5,
                capsize=3.5,
                zorder=4,
            )
        if model_mags:
            perfect_x = np.array(model_mags, dtype=float)
            ax4.plot(perfect_x, perfect_x, "--", color="#8a8578", lw=1.0, zorder=0)
            ax4.set_xticks(model_mags)
            ax4.set_xticklabels([f"{int(m)}°" for m in model_mags])
        ax4.set_title("Model Re-Aiming Profiles", fontsize=9.5, pad=4)
        ax4.set_xlabel("Rotation magnitude (°)")
        ax4.set_ylabel("Mean |model-predicted aim| in onset-aligned window (°)")
        ax4.set_ylim(0, 45)
        ax4.set_yticks([0, 15, 30, 45])
        ax4.text(
            0.98,
            0.97,
            "\n".join(
                [
                    "Mean true other weight:",
                    *[f"{grp}: {quartile_other.get(grp, np.nan):.2f}" for grp in group_order],
                ]
            ),
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            bbox=dict(
                boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d6cfbf", alpha=0.95
            ),
        )
        legend_handles_d = [
            Patch(
                facecolor="#bdbdbd",
                alpha=0.22,
                edgecolor="none",
                label="Model baseline |aim| ± 95% CI",
            ),
            Line2D([0], [0], color="#8a8578", lw=1.0, ls="--", label="Perfect discrimination"),
        ]
        for grp in group_order:
            legend_handles_d.insert(
                group_order.index(grp),
                Line2D(
                    [0],
                    [0],
                    color=group_colors[grp],
                    marker="o",
                    lw=2.4,
                    label=f"{grp} (n = {int(grp_counts.get(grp, 0))})",
                ),
            )
        ax4.legend(handles=legend_handles_d, loc="lower right", fontsize=7.2, frameon=False)
    else:
        raise ValueError(f"Unknown panel_d_mode: {panel_d_mode}")
    add_panel_label(ax4, label_d)

    fig.savefig(out_fig, dpi=240, bbox_inches="tight")


def generate_wildcard_trueother_multipanel(
    window_cycles: int = WINDOW_CYCLES,
    out_fig: Path = OUT_FIG,
    out_csv: Path = OUT_CSV,
    show_profile_spaghetti: bool = True,
    panel_d_mode: str = "coef",
    show_panel_b: bool = True,
) -> Tuple[Path, Path]:
    fit, recovered = load_fit_and_recovery()
    df = build_analysis_table(fit, recovered, window_cycles=window_cycles)
    panel_a_df = build_panel_a_table(fit, recovered)
    learner_pp = panel_a_df.loc[panel_a_df["learner"] == True, "pp"].tolist()
    profile_df = build_profile_table(
        fit, recovered, learner_pp=learner_pp, window_cycles=window_cycles
    )
    model_profile_df = None
    if panel_d_mode == "model_profile":
        model_profile_df = build_model_profile_table(
            fit, recovered, learner_pp=learner_pp, window_cycles=window_cycles
        )
    make_multipanel_figure(
        df,
        panel_a_df,
        profile_df,
        model_profile_df=model_profile_df,
        window_cycles=window_cycles,
        out_fig=Path(out_fig),
        out_csv=Path(out_csv),
        show_profile_spaghetti=show_profile_spaghetti,
        panel_d_mode=panel_d_mode,
        show_panel_b=show_panel_b,
    )
    return Path(out_fig), Path(out_csv)


def generate_wildcard_pen12_window16_multipanel(
    out_fig: Path = OUT_FIG,
    out_csv: Path = OUT_CSV,
) -> Tuple[Path, Path]:
    return generate_wildcard_trueother_multipanel(
        window_cycles=WINDOW_CYCLES,
        out_fig=out_fig,
        out_csv=out_csv,
    )


def main() -> None:
    out_fig, out_csv = generate_wildcard_pen12_window16_multipanel()
    print(f"Saved {out_fig}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
