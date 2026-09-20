from __future__ import annotations

import hashlib
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t


ACTION_ANGLES = np.arange(361, dtype=float) - 180.0
DEFAULT_HMM_ACTION_NU = 5.0
MIN_BASELINE_NU = 10.0
MAX_BASELINE_NU = 1000.0
GAUSSIAN_APPROX_NU = 1000.0
MAX_BASELINE_LOG_NU_SPAN = math.log(MAX_BASELINE_NU - MIN_BASELINE_NU)
MIN_SCALE = 1.0
RNG_SEED = 20260404
DEFAULT_MAX_WORKERS = max(1, min(6, os.cpu_count() or 1))

BHT_COLOR = "#1b9e77"
HMM_COLOR = "#6f6f6f"
POINT_COLOR = "#555555"
MEAN_COLOR = "#d62728"


def _unwrap_numpy_object(obj):
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        if obj.dtype == object and obj.size == 1:
            return obj.flat[0]
    return obj


def _participant_key(value):
    return str(value)


def signed_angular_dist(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def angular_energy_score(sample_a, sample_b, observed_aim):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    obs_term = np.mean(np.abs(signed_angular_dist(sample_a - observed_aim)))
    spread_term = 0.5 * np.mean(np.abs(signed_angular_dist(sample_a - sample_b)))
    return float(obs_term - spread_term), float(obs_term)


def _stable_seed(text):
    # Unlike hash(), this is stable across worker processes and interpreter sessions.
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _process_pool_available():
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "")
    if not main_file:
        return True
    return "<stdin>" not in str(main_file)


def _participant_df(df, participant):
    mask = df["participantNum"] == participant
    if not np.any(mask):
        mask = df["participantNum"].astype(str) == str(participant)
    df_pp = df.loc[mask].copy()

    for trial_col in ("trial_number", "trialNum", "trial"):
        if trial_col in df_pp.columns:
            df_pp = df_pp.sort_values(trial_col)
            break
    else:
        df_pp = df_pp.sort_index()
    return df_pp.reset_index(drop=True)


def _dict_get(mapping, key):
    if mapping is None:
        return None
    if key in mapping:
        return mapping[key]
    key_str = str(key)
    if key_str in mapping:
        return mapping[key_str]
    for candidate_key, value in mapping.items():
        if str(candidate_key) == key_str:
            return value
    return None


def trim_predstate(predstate, n_trials):
    trimmed = {}
    for key, value in predstate.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= n_trials:
            trimmed[key] = value[:n_trials].copy()
        else:
            trimmed[key] = value
    return trimmed


def _infer_rotation_mask(df_pp, q0s=None):
    if "phase" in df_pp.columns:
        phase = df_pp["phase"].astype(str).str.lower().to_numpy()
        rotation_mask = phase == "rotation"
        if np.any(rotation_mask):
            return rotation_mask

    if "rotation" in df_pp.columns:
        rotations = df_pp["rotation"].to_numpy(dtype=float)
        nz = np.flatnonzero(np.abs(rotations) > 1e-9)
        if len(nz) > 0:
            rotation_mask = np.zeros(len(df_pp), dtype=bool)
            rotation_mask[nz[0] : nz[-1] + 1] = True
            return rotation_mask

    if q0s is not None:
        q0s = np.asarray(q0s, dtype=float)
        below = np.flatnonzero(q0s < 0.95)
        if len(below) > 0:
            rotation_mask = np.zeros(len(df_pp), dtype=bool)
            rotation_mask[below[0] :] = True
            return rotation_mask

    return np.zeros(len(df_pp), dtype=bool)


def _early_rotation_mask(rotation_mask, early_rotation_trials):
    early_mask = np.zeros_like(rotation_mask, dtype=bool)
    rotation_idx = np.flatnonzero(rotation_mask)
    if len(rotation_idx) > 0:
        early_mask[rotation_idx[:early_rotation_trials]] = True
    return early_mask


def fit_baseline_student_t(bl_aims):
    bl = np.asarray(bl_aims, dtype=np.float64)
    bl = bl[~np.isnan(bl)]
    if len(bl) < 5:
        if len(bl) > 1:
            sc = max(1.4826 * np.median(np.abs(bl - np.median(bl))), MIN_SCALE)
        else:
            sc = 3.0
        return float(sc), float(MIN_BASELINE_NU)

    mad_scale = max(1.4826 * np.median(np.abs(bl - np.median(bl))), MIN_SCALE)

    def negll(params):
        log_scale, log_nu_minus_min = params
        scale = max(np.exp(np.clip(log_scale, -700.0, 700.0)), MIN_SCALE)
        nu = MIN_BASELINE_NU + np.exp(np.clip(log_nu_minus_min, -700.0, MAX_BASELINE_LOG_NU_SPAN))
        z = bl / scale
        if nu >= GAUSSIAN_APPROX_NU:
            ll = -np.log(scale * np.sqrt(2.0 * np.pi)) - 0.5 * z * z
        else:
            ll = (
                math.lgamma(0.5 * (nu + 1.0))
                - math.lgamma(0.5 * nu)
                - 0.5 * np.log(nu * np.pi)
                - np.log(scale)
                - 0.5 * (nu + 1.0) * np.log(1.0 + z * z / nu)
            )
        return -np.sum(ll)

    best_val = np.inf
    best_params = (mad_scale, float(MIN_BASELINE_NU))
    for nu_init in (MIN_BASELINE_NU + 0.1, 30.0, 100.0):
        x0 = [
            np.log(mad_scale),
            np.log(
                min(
                    max(nu_init - MIN_BASELINE_NU, 1e-6),
                    MAX_BASELINE_NU - MIN_BASELINE_NU,
                )
            ),
        ]
        try:
            res = minimize(
                negll,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6},
            )
            if res.fun < best_val:
                best_val = res.fun
                sc = max(np.exp(np.clip(res.x[0], -700.0, 700.0)), MIN_SCALE)
                nu = MIN_BASELINE_NU + np.exp(np.clip(res.x[1], -700.0, MAX_BASELINE_LOG_NU_SPAN))
                best_params = (sc, nu)
        except Exception:
            continue
    return float(best_params[0]), float(best_params[1])


def _baseline_from_df(df_pp):
    if len(df_pp) == 0 or "aim" not in df_pp.columns:
        return 3.0, float(MIN_BASELINE_NU)

    if "phase" in df_pp.columns:
        phase = df_pp["phase"].astype(str).str.lower().to_numpy()
        baseline_mask = phase == "baseline"
    elif "rotation" in df_pp.columns:
        rotations = df_pp["rotation"].to_numpy(dtype=float)
        nz = np.flatnonzero(np.abs(rotations) > 1e-9)
        first_nonzero = int(nz[0]) if len(nz) > 0 else len(rotations)
        baseline_mask = np.arange(len(rotations)) < first_nonzero
    else:
        baseline_mask = np.zeros(len(df_pp), dtype=bool)

    bl = df_pp["aim"].to_numpy(dtype=float)[baseline_mask]
    return fit_baseline_student_t(bl)


def _hmm_baseline_params(hmm_fit, hmm_idx, df_pp):
    baseline_scale = getattr(hmm_fit, "baselineScale", None)
    baseline_nu = getattr(hmm_fit, "baselineNu", None)

    if baseline_scale is not None and baseline_nu is not None:
        try:
            scale = float(np.asarray(baseline_scale)[hmm_idx])
            nu = float(np.asarray(baseline_nu)[hmm_idx])
            if np.isfinite(scale) and np.isfinite(nu) and scale > 0 and nu > 0:
                return scale, nu
        except Exception:
            pass

    return _baseline_from_df(df_pp)


def _get_hmm_pi_preds(hmm_fit, hmm_idx):
    pi_preds = getattr(hmm_fit, "pi_preds", None)
    if pi_preds is not None:
        return np.asarray(pi_preds[hmm_idx], dtype=float)
    pi_preds = getattr(hmm_fit, "model_predictive_pis", None)
    if pi_preds is not None:
        return np.asarray(pi_preds[hmm_idx], dtype=float)
    raise AttributeError("HMM fit does not expose predictive state probabilities.")


def sample_bht_predictive(record, trial_idx, rng, n_samples):
    predstate = record["predstate"]
    n_comp = int(predstate["predNComp"][trial_idx])
    if n_comp <= 0:
        return np.full(n_samples, np.nan)

    log_w = predstate["predCompLogProb"][trial_idx, :n_comp]
    weights = np.exp(log_w - logsumexp(log_w))
    comp_idx = rng.choice(n_comp, size=n_samples, p=weights)
    loc = predstate["predCompMean"][trial_idx, comp_idx]
    scale = predstate["predCompScale"][trial_idx, comp_idx]
    nu = predstate["predCompNu"][trial_idx, comp_idx]
    samples = student_t.rvs(df=nu, loc=loc, scale=scale, size=n_samples, random_state=rng)
    return signed_angular_dist(samples)


def sample_hmm_predictive(record, trial_idx, rng, n_samples):
    samples = np.empty(n_samples, dtype=float)
    pi_preds = np.asarray(record["hmm_pi_preds"][trial_idx], dtype=float)
    policies = np.asarray(record["hmm_policies"][trial_idx], dtype=float)

    state1 = rng.random(n_samples) >= pi_preds[0]
    n_baseline = int(np.sum(~state1))
    n_move = int(np.sum(state1))

    if n_baseline > 0:
        samples[~state1] = student_t.rvs(
            df=record["hmm_baseline_nu"],
            loc=0.0,
            scale=record["hmm_baseline_scale"],
            size=n_baseline,
            random_state=rng,
        )

    if n_move > 0:
        policy_sum = np.sum(policies)
        if policy_sum <= 0:
            samples[state1] = student_t.rvs(
                df=record["hmm_baseline_nu"],
                loc=0.0,
                scale=record["hmm_baseline_scale"],
                size=n_move,
                random_state=rng,
            )
        else:
            norm_policy = policies / policy_sum
            action_idx = rng.choice(len(ACTION_ANGLES), size=n_move, p=norm_policy)
            loc = ACTION_ANGLES[action_idx]
            samples[state1] = student_t.rvs(
                df=DEFAULT_HMM_ACTION_NU,
                loc=loc,
                scale=record["hmm_noise"],
                size=n_move,
                random_state=rng,
            )

    return signed_angular_dist(samples)


def compute_energy_metrics_for_record(record, n_samples):
    rng = np.random.default_rng(RNG_SEED + _stable_seed(f"{record['group']}::{record['pp']}"))
    aims = record["aims"]
    rotation_idx = np.flatnonzero(record["rotation_mask"] & ~np.isnan(aims))
    early_idx = set(np.flatnonzero(record["early_rotation_mask"] & ~np.isnan(aims)))

    bht_scores = []
    hmm_scores = []
    bht_early_scores = []
    hmm_early_scores = []
    bht_obs_terms = []
    hmm_obs_terms = []
    bht_early_obs_terms = []
    hmm_early_obs_terms = []

    # Independent draws estimate the pairwise-distance term without an O(n²) matrix.
    for trial_idx in rotation_idx:
        bht_a = sample_bht_predictive(record, trial_idx, rng, n_samples)
        bht_b = sample_bht_predictive(record, trial_idx, rng, n_samples)
        hmm_a = sample_hmm_predictive(record, trial_idx, rng, n_samples)
        hmm_b = sample_hmm_predictive(record, trial_idx, rng, n_samples)

        bht_score, bht_obs = angular_energy_score(bht_a, bht_b, aims[trial_idx])
        hmm_score, hmm_obs = angular_energy_score(hmm_a, hmm_b, aims[trial_idx])
        bht_scores.append(bht_score)
        hmm_scores.append(hmm_score)
        bht_obs_terms.append(bht_obs)
        hmm_obs_terms.append(hmm_obs)
        if trial_idx in early_idx:
            bht_early_scores.append(bht_score)
            hmm_early_scores.append(hmm_score)
            bht_early_obs_terms.append(bht_obs)
            hmm_early_obs_terms.append(hmm_obs)

    bht_rotation_energy = float(np.nanmean(bht_scores)) if bht_scores else np.nan
    hmm_rotation_energy = float(np.nanmean(hmm_scores)) if hmm_scores else np.nan
    bht_early_rotation_energy = float(np.nanmean(bht_early_scores)) if bht_early_scores else np.nan
    hmm_early_rotation_energy = float(np.nanmean(hmm_early_scores)) if hmm_early_scores else np.nan
    bht_rotation_obs = float(np.nanmean(bht_obs_terms)) if bht_obs_terms else np.nan
    hmm_rotation_obs = float(np.nanmean(hmm_obs_terms)) if hmm_obs_terms else np.nan
    bht_early_rotation_obs = (
        float(np.nanmean(bht_early_obs_terms)) if bht_early_obs_terms else np.nan
    )
    hmm_early_rotation_obs = (
        float(np.nanmean(hmm_early_obs_terms)) if hmm_early_obs_terms else np.nan
    )

    return {
        "pp": record["pp"],
        "condition": record["condition"],
        "group": record["group"],
        "bht_rotation_energy": bht_rotation_energy,
        "hmm_rotation_energy": hmm_rotation_energy,
        "delta_rotation_energy": hmm_rotation_energy - bht_rotation_energy,
        "bht_early_rotation_energy": bht_early_rotation_energy,
        "hmm_early_rotation_energy": hmm_early_rotation_energy,
        "delta_early_rotation_energy": hmm_early_rotation_energy - bht_early_rotation_energy,
        "bht_rotation_obs": bht_rotation_obs,
        "hmm_rotation_obs": hmm_rotation_obs,
        "delta_rotation_obs": hmm_rotation_obs - bht_rotation_obs,
        "bht_early_rotation_obs": bht_early_rotation_obs,
        "hmm_early_rotation_obs": hmm_early_rotation_obs,
        "delta_early_rotation_obs": hmm_early_rotation_obs - bht_early_rotation_obs,
        "n_rotation_trials": int(np.sum(record["rotation_mask"] & ~np.isnan(aims))),
        "n_early_rotation_trials": int(np.sum(record["early_rotation_mask"] & ~np.isnan(aims))),
    }


def _build_records_for_fit_pair(
    condition_label,
    group_label,
    bht_fit,
    hmm_fit,
    early_rotation_trials,
):
    bht_fit = _unwrap_numpy_object(bht_fit)
    hmm_fit = _unwrap_numpy_object(hmm_fit)

    if not hasattr(bht_fit, "predState"):
        raise ValueError(f"{condition_label} does not expose per-participant predState.")

    hmm_lookup = {_participant_key(pp): i for i, pp in enumerate(hmm_fit.participantNums)}
    records = []

    for pp_idx, pp in enumerate(bht_fit.participantNums):
        predstate = _dict_get(bht_fit.predState, pp)
        if predstate is None:
            continue

        hmm_idx = hmm_lookup.get(_participant_key(pp))
        if hmm_idx is None:
            continue

        df_pp = _participant_df(bht_fit.df, pp)
        trial_statuses = _dict_get(getattr(bht_fit, "trialStatuses", None), pp)
        if trial_statuses is not None:
            trial_statuses = np.asarray(trial_statuses, dtype=object)
            if len(df_pp) > len(trial_statuses):
                df_pp = df_pp.iloc[: len(trial_statuses)].copy()

        aims = np.asarray(bht_fit.allAims[pp_idx], dtype=float)
        hmm_pi_preds = _get_hmm_pi_preds(hmm_fit, hmm_idx)
        hmm_policies = np.asarray(hmm_fit.model_predictive_policies[hmm_idx], dtype=float)

        n_trials = min(
            len(aims),
            len(df_pp),
            len(predstate["predNComp"]),
            len(hmm_pi_preds),
            len(hmm_policies),
        )
        if n_trials < 5:
            continue

        predstate = trim_predstate(predstate, n_trials)
        aims = aims[:n_trials]
        df_pp = df_pp.iloc[:n_trials].copy()
        rotation_mask = _infer_rotation_mask(df_pp, q0s=predstate.get("q0s"))
        if not np.any(rotation_mask):
            continue

        early_mask = _early_rotation_mask(rotation_mask, early_rotation_trials)
        hmm_baseline_scale, hmm_baseline_nu = _hmm_baseline_params(hmm_fit, hmm_idx, df_pp)
        hmm_noise = float(np.asarray(hmm_fit.xs[hmm_idx], dtype=float)[0])

        records.append(
            {
                "pp": _participant_key(pp),
                "condition": condition_label,
                "group": group_label,
                "aims": aims,
                "predstate": predstate,
                "rotation_mask": rotation_mask,
                "early_rotation_mask": early_mask,
                "hmm_pi_preds": hmm_pi_preds[:n_trials],
                "hmm_policies": hmm_policies[:n_trials],
                "hmm_noise": hmm_noise,
                "hmm_baseline_scale": float(hmm_baseline_scale),
                "hmm_baseline_nu": float(hmm_baseline_nu),
            }
        )

    return records


def compute_energy_metric_table(records, n_samples, parallel=True, max_workers=DEFAULT_MAX_WORKERS):
    if len(records) == 0:
        return pd.DataFrame()

    rows = []
    use_parallel = parallel and len(records) > 1 and _process_pool_available()
    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                iterator = executor.map(
                    compute_energy_metrics_for_record,
                    records,
                    repeat(n_samples),
                )
                rows = list(iterator)
        except Exception as exc:
            print(f"Parallel energy-score computation failed ({exc}); falling back to serial.")
            rows = [compute_energy_metrics_for_record(record, n_samples) for record in records]
    else:
        rows = [compute_energy_metrics_for_record(record, n_samples) for record in records]

    return pd.DataFrame(rows)


def save_figure_bundle(fig, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in (".png", ".svg"):
        out_path = base_path.with_suffix(ext)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        outputs[ext.lstrip(".")] = str(out_path)
    plt.close(fig)
    return outputs


def _force_all_ticks(ax):
    """Ensure all tick labels are visible on both axes."""
    ax.tick_params(
        axis="both", which="both", labelbottom=True, labelleft=True, bottom=True, left=True
    )
    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    for label in ax.get_xticklabels():
        label.set_visible(True)
    for label in ax.get_yticklabels():
        label.set_visible(True)


def _plot_delta_panel(ax, participant_df, group_order, value_col, ylabel, title, ylim=None):
    positions = np.arange(len(group_order))
    labels = []

    ax.axhline(0.0, color="black", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(-0.5, len(group_order) - 0.5)

    for ci, group in enumerate(group_order):
        sub = (
            participant_df.loc[participant_df["group"] == group, value_col]
            .dropna()
            .to_numpy(dtype=float)
        )
        labels.append(f"{group}\n(n={len(sub)})")
        if len(sub) == 0:
            continue
        jitter = np.random.default_rng(100 + ci).uniform(-0.2, 0.2, len(sub))
        ax.scatter(
            ci + jitter,
            sub,
            s=20,
            alpha=0.35,
            color=POINT_COLOR,
            edgecolors="none",
            zorder=2,
        )
        mean_val = float(np.mean(sub))
        sem_val = float(np.std(sub, ddof=0) / np.sqrt(len(sub)))
        ax.errorbar(
            ci,
            mean_val,
            yerr=1.96 * sem_val,
            fmt="o",
            color=MEAN_COLOR,
            markersize=8,
            capsize=5,
            capthick=1.5,
            lw=1.5,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    ymin, ymax = ax.get_ylim()
    ax.fill_between(
        [-0.5, len(group_order) - 0.5],
        0,
        ymax,
        color="#d4edda",
        alpha=0.15,
        zorder=0,
    )
    ax.fill_between(
        [-0.5, len(group_order) - 0.5],
        ymin,
        0,
        color="#f8d7da",
        alpha=0.15,
        zorder=0,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        len(group_order) - 0.55,
        ymax - 0.06 * (ymax - ymin),
        "BHT better",
        ha="right",
        va="top",
        fontsize=9,
        color="#155724",
        style="italic",
    )
    ax.text(
        len(group_order) - 0.55,
        ymin + 0.06 * (ymax - ymin),
        "HMM better",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#721c24",
        style="italic",
    )
    sns.despine(ax=ax)
    _force_all_ticks(ax)


def _plot_absolute_panel(
    ax, participant_df, group_order, bht_col, hmm_col, ylabel, title, ylim=None
):
    width = 0.16
    labels = []

    for ci, group in enumerate(group_order):
        group_df = (
            participant_df.loc[participant_df["group"] == group].copy().reset_index(drop=True)
        )
        paired_df = group_df.loc[group_df[bht_col].notna() & group_df[hmm_col].notna()].copy()
        labels.append(
            f"{group}\n(n={max(int(group_df[bht_col].notna().sum()), int(group_df[hmm_col].notna().sum()))})"
        )

        if len(paired_df) > 0:
            pair_jitter = np.random.default_rng(300 + ci).uniform(-0.06, 0.06, len(paired_df))
            paired_df["bht_x"] = ci - width + pair_jitter
            paired_df["hmm_x"] = ci + width + pair_jitter

            for _, row in paired_df.iterrows():
                ax.plot(
                    [float(row["bht_x"]), float(row["hmm_x"])],
                    [float(row[bht_col]), float(row[hmm_col])],
                    color="#9a9a9a",
                    alpha=0.18,
                    lw=0.5,
                    zorder=1.5,
                )

            ax.scatter(
                paired_df["bht_x"].to_numpy(dtype=float),
                paired_df[bht_col].to_numpy(dtype=float),
                s=18,
                alpha=0.30,
                color=BHT_COLOR,
                edgecolors="none",
                zorder=2.1,
            )
            ax.scatter(
                paired_df["hmm_x"].to_numpy(dtype=float),
                paired_df[hmm_col].to_numpy(dtype=float),
                s=18,
                alpha=0.30,
                color=HMM_COLOR,
                edgecolors="none",
                zorder=2.1,
            )

        for x_pos, color, value_col in (
            (ci - width, BHT_COLOR, bht_col),
            (ci + width, HMM_COLOR, hmm_col),
        ):
            values = group_df[value_col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean_val = float(np.mean(values))
            sem_val = float(np.std(values, ddof=0) / np.sqrt(len(values)))
            ax.errorbar(
                x_pos,
                mean_val,
                yerr=1.96 * sem_val,
                fmt="o",
                color=color,
                markersize=8,
                capsize=5,
                capthick=1.5,
                lw=1.5,
                zorder=3,
            )

    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.99,
        0.98,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
        style="italic",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    sns.despine(ax=ax)
    _force_all_ticks(ax)


def _save_metric_tables(participant_df, group_order, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    participant_path = save_dir / "participant_energy_metrics.csv"
    participant_df.to_csv(participant_path, index=False)

    summary_df = (
        participant_df.groupby("group", as_index=False)
        .agg(
            n=("pp", "nunique"),
            bht_rotation_energy_mean=("bht_rotation_energy", "mean"),
            hmm_rotation_energy_mean=("hmm_rotation_energy", "mean"),
            delta_rotation_energy_mean=("delta_rotation_energy", "mean"),
            bht_early_rotation_energy_mean=("bht_early_rotation_energy", "mean"),
            hmm_early_rotation_energy_mean=("hmm_early_rotation_energy", "mean"),
            delta_early_rotation_energy_mean=("delta_early_rotation_energy", "mean"),
            bht_rotation_obs_mean=("bht_rotation_obs", "mean"),
            hmm_rotation_obs_mean=("hmm_rotation_obs", "mean"),
            delta_rotation_obs_mean=("delta_rotation_obs", "mean"),
            bht_early_rotation_obs_mean=("bht_early_rotation_obs", "mean"),
            hmm_early_rotation_obs_mean=("hmm_early_rotation_obs", "mean"),
            delta_early_rotation_obs_mean=("delta_early_rotation_obs", "mean"),
        )
        .set_index("group")
        .reindex(group_order)
        .reset_index()
    )
    summary_path = save_dir / "group_energy_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "participant_metrics_csv": str(participant_path),
        "group_summary_csv": str(summary_path),
    }


def _save_energy_figures(
    dataset_label,
    participant_df,
    group_order,
    save_dir,
    early_rotation_trials,
    ylim=None,
    obs_ylim=None,
):
    delta_fig, delta_axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_delta_panel(
        delta_axes[0],
        participant_df,
        group_order,
        "delta_rotation_energy",
        "Delta angular energy score\n(HMM - BHT)",
        "Rotation phase",
        ylim=ylim,
    )
    _plot_delta_panel(
        delta_axes[1],
        participant_df,
        group_order,
        "delta_early_rotation_energy",
        f"Delta angular energy score\n(HMM - BHT), first {early_rotation_trials} rotation trials",
        "Early rotation",
        ylim=ylim,
    )
    delta_fig.suptitle(f"{dataset_label}: delta energy score", fontsize=12, y=0.98)

    absolute_fig, absolute_axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_absolute_panel(
        absolute_axes[0],
        participant_df,
        group_order,
        "bht_rotation_energy",
        "hmm_rotation_energy",
        "Angular energy score",
        "Rotation phase",
    )
    _plot_absolute_panel(
        absolute_axes[1],
        participant_df,
        group_order,
        "bht_early_rotation_energy",
        "hmm_early_rotation_energy",
        f"Angular energy score\nfirst {early_rotation_trials} rotation trials",
        "Early rotation",
    )
    absolute_fig.suptitle(f"{dataset_label}: absolute energy score", fontsize=12, y=0.98)
    absolute_fig.legend(
        [
            plt.Line2D([0], [0], marker="o", color=BHT_COLOR, lw=0, markersize=8, label="BHT"),
            plt.Line2D([0], [0], marker="o", color=HMM_COLOR, lw=0, markersize=8, label="HMM"),
        ],
        ["BHT", "HMM"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    obs_delta_fig, obs_delta_axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_delta_panel(
        obs_delta_axes[0],
        participant_df,
        group_order,
        "delta_rotation_obs",
        "Delta mean sample-obs distance (°)\n(HMM - BHT)",
        "Rotation phase",
        ylim=obs_ylim,
    )
    _plot_delta_panel(
        obs_delta_axes[1],
        participant_df,
        group_order,
        "delta_early_rotation_obs",
        f"Delta mean sample-obs distance (°)\n(HMM - BHT), first {early_rotation_trials} rotation trials",
        "Early rotation",
        ylim=obs_ylim,
    )
    obs_delta_fig.suptitle(
        f"{dataset_label}: delta mean sample-observation distance", fontsize=12, y=0.98
    )

    obs_absolute_fig, obs_absolute_axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_absolute_panel(
        obs_absolute_axes[0],
        participant_df,
        group_order,
        "bht_rotation_obs",
        "hmm_rotation_obs",
        "Mean sample-obs distance (°)",
        "Rotation phase",
    )
    _plot_absolute_panel(
        obs_absolute_axes[1],
        participant_df,
        group_order,
        "bht_early_rotation_obs",
        "hmm_early_rotation_obs",
        f"Mean sample-obs distance (°)\nfirst {early_rotation_trials} rotation trials",
        "Early rotation",
    )
    obs_absolute_fig.suptitle(
        f"{dataset_label}: absolute mean sample-observation distance", fontsize=12, y=0.98
    )
    obs_absolute_fig.legend(
        [
            plt.Line2D([0], [0], marker="o", color=BHT_COLOR, lw=0, markersize=8, label="BHT"),
            plt.Line2D([0], [0], marker="o", color=HMM_COLOR, lw=0, markersize=8, label="HMM"),
        ],
        ["BHT", "HMM"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    return {
        "energy_delta": save_figure_bundle(delta_fig, Path(save_dir) / "energy_delta"),
        "energy_absolute": save_figure_bundle(absolute_fig, Path(save_dir) / "energy_absolute"),
        "obs_delta": save_figure_bundle(obs_delta_fig, Path(save_dir) / "obs_delta"),
        "obs_absolute": save_figure_bundle(obs_absolute_fig, Path(save_dir) / "obs_absolute"),
    }


def _run_energy_analysis(
    dataset_label,
    records,
    group_order,
    save_dir,
    n_samples,
    early_rotation_trials,
    parallel,
    max_workers,
    ylim=None,
    obs_ylim=None,
):
    sns.set_theme(style="white", context="notebook")
    participant_df = compute_energy_metric_table(
        records,
        n_samples=n_samples,
        parallel=parallel,
        max_workers=max_workers,
    )
    table_paths = _save_metric_tables(participant_df, group_order, save_dir)
    figure_paths = _save_energy_figures(
        dataset_label,
        participant_df,
        group_order,
        save_dir,
        early_rotation_trials=early_rotation_trials,
        ylim=ylim,
        obs_ylim=obs_ylim,
    )
    return {
        "participant_df": participant_df,
        "figure_paths": figure_paths,
        "table_paths": table_paths,
        "group_order": group_order,
    }


def run_ding_energy_analysis(
    bht_fit_dict,
    hmm_fit,
    save_dir,
    n_samples=10000,
    early_rotation_trials=24,
    parallel=True,
    max_workers=DEFAULT_MAX_WORKERS,
    ylim=None,
    obs_ylim=None,
):
    bht_fit_dict = {
        condition: _unwrap_numpy_object(fit)
        for condition, fit in _unwrap_numpy_object(bht_fit_dict).items()
    }
    hmm_fit = _unwrap_numpy_object(hmm_fit)
    group_order = ["Inner_2T", "Outer_2T", "Inner_8T", "Outer_8T"]

    records = []
    for condition in sorted(bht_fit_dict):
        group = condition.rsplit("_", 1)[0]
        if group not in group_order:
            group_order.append(group)
        records.extend(
            _build_records_for_fit_pair(
                condition_label=condition,
                group_label=group,
                bht_fit=bht_fit_dict[condition],
                hmm_fit=hmm_fit,
                early_rotation_trials=early_rotation_trials,
            )
        )

    return _run_energy_analysis(
        dataset_label="Ding",
        records=records,
        group_order=group_order,
        save_dir=save_dir,
        n_samples=n_samples,
        early_rotation_trials=early_rotation_trials,
        parallel=parallel,
        max_workers=max_workers,
        ylim=ylim,
        obs_ylim=obs_ylim,
    )


def run_single_group_energy_analysis(
    dataset_label,
    bht_fit,
    hmm_fit,
    save_dir,
    group_label=None,
    n_samples=10000,
    early_rotation_trials=24,
    parallel=True,
    max_workers=DEFAULT_MAX_WORKERS,
    ylim=None,
    obs_ylim=None,
):
    group_label = group_label or dataset_label
    bht_fit = _unwrap_numpy_object(bht_fit)
    hmm_fit = _unwrap_numpy_object(hmm_fit)
    records = _build_records_for_fit_pair(
        condition_label=group_label,
        group_label=group_label,
        bht_fit=bht_fit,
        hmm_fit=hmm_fit,
        early_rotation_trials=early_rotation_trials,
    )
    return _run_energy_analysis(
        dataset_label=dataset_label,
        records=records,
        group_order=[group_label],
        save_dir=save_dir,
        n_samples=n_samples,
        early_rotation_trials=early_rotation_trials,
        parallel=parallel,
        max_workers=max_workers,
        ylim=ylim,
        obs_ylim=obs_ylim,
    )


def run_array_energy_analysis(
    dataset_label,
    bht_fits,
    hmm_fits,
    save_dir,
    group_labels=None,
    n_samples=10000,
    early_rotation_trials=24,
    parallel=True,
    max_workers=DEFAULT_MAX_WORKERS,
    ylim=None,
    obs_ylim=None,
):
    bht_fits = list(np.asarray(bht_fits, dtype=object).flat)
    hmm_fits = list(np.asarray(hmm_fits, dtype=object).flat)
    if len(bht_fits) != len(hmm_fits):
        raise ValueError(f"{dataset_label}: BHT/HMM array lengths do not match.")

    if group_labels is None:
        group_labels = []
        for hmm_fit in hmm_fits:
            con_val = getattr(hmm_fit, "conVal", None)
            if con_val is None:
                group_labels.append(str(len(group_labels)))
            else:
                group_labels.append(str(int(abs(float(con_val)))))
    if len(group_labels) != len(bht_fits):
        raise ValueError(f"{dataset_label}: group_labels length does not match fit arrays.")

    records = []
    for group_label, bht_fit, hmm_fit in zip(group_labels, bht_fits, hmm_fits):
        records.extend(
            _build_records_for_fit_pair(
                condition_label=group_label,
                group_label=group_label,
                bht_fit=bht_fit,
                hmm_fit=hmm_fit,
                early_rotation_trials=early_rotation_trials,
            )
        )

    return _run_energy_analysis(
        dataset_label=dataset_label,
        records=records,
        group_order=list(group_labels),
        save_dir=save_dir,
        n_samples=n_samples,
        early_rotation_trials=early_rotation_trials,
        parallel=parallel,
        max_workers=max_workers,
        ylim=ylim,
        obs_ylim=obs_ylim,
    )
