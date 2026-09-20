"""Compare SHIFT and Q-HMM predictions on the Ding dataset.

Metrics are circular energy scores, generalisation sign-flip Brier scores,
and rotation-phase R-squared from predictive means.
"""

from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp
from scipy.stats import t as student_t


COND_ORDER = ["Inner_2T", "Outer_2T", "Inner_8T", "Outer_8T"]
COND_COLORS = {
    "Inner_2T": "#1f77b4",
    "Outer_2T": "#ff7f0e",
    "Inner_8T": "#2ca02c",
    "Outer_8T": "#d62728",
}
BHT_COLOR = "#1b9e77"
HMM_COLOR = "#6f6f6f"
HUMAN_COLOR = "#111111"

ACTION_ANGLES = np.arange(361, dtype=float) - 180.0
DEFAULT_HMM_ACTION_NU = 5.0
EARLY_ROTATION_TRIALS = 24
EARLY_GENERALISATION_TRIALS = 8
ENERGY_SCORE_SAMPLES = 192
RNG_SEED = 20260404
DEFAULT_MAX_WORKERS = max(1, min(6, os.cpu_count() or 1))

FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 8


def signed_angular_dist(x):
    return (x + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles, weights=None):
    angles = np.asarray(angles, dtype=float)
    valid = ~np.isnan(angles)
    if weights is None:
        weights = np.ones(np.sum(valid), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)[valid]
    angles = angles[valid]
    if len(angles) == 0 or np.sum(weights) <= 0:
        return np.nan
    radians = np.deg2rad(angles)
    sin_term = np.sum(weights * np.sin(radians))
    cos_term = np.sum(weights * np.cos(radians))
    return np.rad2deg(np.arctan2(sin_term, cos_term))


def compute_bht_predictive_mean(predstate, n_trials):
    pred_means = np.full(n_trials, np.nan)
    for t in range(n_trials):
        n_comp = int(predstate["predNComp"][t])
        if n_comp <= 0:
            continue
        log_w = predstate["predCompLogProb"][t, :n_comp]
        weights = np.exp(log_w - logsumexp(log_w))
        means = predstate["predCompMean"][t, :n_comp]
        pred_means[t] = circular_mean_deg(means, weights=weights)
    return pred_means


def compute_angular_r2(aims, predictions):
    aims = np.asarray(aims, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    valid = ~np.isnan(aims) & ~np.isnan(predictions)
    if np.sum(valid) < 2:
        return np.nan
    y = aims[valid]
    y_hat = predictions[valid]
    ss_res = np.sum(signed_angular_dist(y - y_hat) ** 2)
    center = circular_mean_deg(y)
    if np.isnan(center):
        return np.nan
    ss_tot = np.sum(signed_angular_dist(y - center) ** 2)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_scalar_r2(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(valid) < 2:
        return np.nan
    y = observed[valid]
    y_hat = predicted[valid]
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_rmse(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(valid) < 1:
        return np.nan
    y = observed[valid]
    y_hat = predicted[valid]
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def detect_rotation_onset(df_pp, q0s=None):
    rotations = df_pp["rotation"].to_numpy(dtype=float)
    rot_idx = np.flatnonzero(rotations != 0)
    if len(rot_idx) > 0:
        return int(rot_idx[0])
    if q0s is not None:
        q0s = np.asarray(q0s, dtype=float)
        below = np.flatnonzero(q0s < 0.95)
        if len(below) > 0:
            return int(below[0])
    return min(24, len(df_pp))


def trim_predstate(predstate, n_trials):
    trimmed = {}
    for key, value in predstate.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= n_trials:
            trimmed[key] = value[:n_trials].copy()
        else:
            trimmed[key] = value
    return trimmed


def infer_sign_multiplier(rotations):
    """Return multiplier that makes correct compensation positive."""
    rotations = np.asarray(rotations, dtype=float)
    nz = rotations[np.abs(rotations) > 1e-9]
    if len(nz) == 0:
        return 1.0
    sign_val = float(np.sign(nz[0]))
    if sign_val == 0:
        return 1.0
    return -sign_val


def load_bht_fit_dict(path="dingBLRDing.npy"):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    return loaded.item()


def load_hmm_fit(path="HMMDing.npy"):
    loaded = np.load(path, allow_pickle=True)
    return loaded.item() if getattr(loaded, "ndim", None) == 0 else loaded[0]


def extract_hmm_predictive_mean(hmm_obj, idx, n_trials):
    expected = np.asarray(hmm_obj.expected_aims[idx], dtype=float)
    if len(expected) >= n_trials:
        return expected[:n_trials].copy()
    all_explicits = getattr(hmm_obj, "all_model_explicits", None)
    if all_explicits is not None:
        arr = np.asarray(all_explicits[idx], dtype=float)
        if len(arr) >= n_trials:
            return arr[:n_trials].copy()
    out = np.full(n_trials, np.nan)
    out[: min(len(expected), n_trials)] = expected[: min(len(expected), n_trials)]
    return out


def build_participant_records():
    print("Loading BHT fits...")
    bht_fit_dict = load_bht_fit_dict()

    print("Loading HMM fits...")
    hmm_obj = load_hmm_fit()
    hmm_lookup = {pp: i for i, pp in enumerate(hmm_obj.participantNums)}

    grouped_records = {group: [] for group in COND_ORDER}

    for cond_name, fit in sorted(bht_fit_dict.items()):
        group = cond_name.rsplit("_", 1)[0]
        if group not in grouped_records:
            grouped_records[group] = []

        for pp_idx, pp in enumerate(fit.participantNums):
            hmm_idx = hmm_lookup.get(pp)
            if hmm_idx is None:
                continue

            predstate = fit.predState[pp]

            df_pp = (
                fit.df.loc[fit.df["participantNum"] == pp]
                .sort_values("trial_number")
                .reset_index(drop=True)
            )
            trial_statuses = np.asarray(fit.trialStatuses[pp])
            if len(df_pp) > len(trial_statuses):
                df_pp = df_pp.iloc[: len(trial_statuses)].copy()
            if len(df_pp) != len(trial_statuses):
                continue
            df_pp["trialStatus"] = trial_statuses

            aims = np.asarray(fit.allAims[pp_idx], dtype=float)
            n_trials = min(
                len(aims),
                len(df_pp),
                len(predstate["predNComp"]),
                len(hmm_obj.pi_preds[hmm_idx]),
                len(hmm_obj.model_predictive_policies[hmm_idx]),
            )
            if n_trials < 5:
                continue

            df_pp = df_pp.iloc[:n_trials].copy()
            aims = aims[:n_trials]
            predstate = trim_predstate(predstate, n_trials)
            sign_multiplier = infer_sign_multiplier(df_pp["rotation"].to_numpy(dtype=float))
            if "aim" in df_pp.columns:
                df_pp["aim_signed"] = df_pp["aim"].to_numpy(dtype=float) * sign_multiplier
            rot_onset = detect_rotation_onset(df_pp, q0s=predstate.get("q0s"))
            trial_index = np.arange(n_trials)
            rotation_mask = trial_index >= rot_onset
            early_rotation_mask = rotation_mask & (trial_index < rot_onset + EARLY_ROTATION_TRIALS)

            grouped_records[group].append(
                {
                    "pp": pp,
                    "condition": cond_name,
                    "group": group,
                    "aims": aims,
                    "df_pp": df_pp,
                    "predstate": predstate,
                    "rot_onset": rot_onset,
                    "rotation_mask": rotation_mask,
                    "early_rotation_mask": early_rotation_mask,
                    "sign_multiplier": sign_multiplier,
                    "hmm_idx": hmm_idx,
                    "hmm_pi_preds": np.asarray(hmm_obj.pi_preds[hmm_idx], dtype=float)[:n_trials],
                    "hmm_policies": np.asarray(
                        hmm_obj.model_predictive_policies[hmm_idx], dtype=float
                    )[:n_trials],
                    "hmm_noise": float(hmm_obj.xs[hmm_idx][0]),
                    "hmm_baseline_scale": float(hmm_obj.baselineScale[hmm_idx]),
                    "hmm_baseline_nu": float(hmm_obj.baselineNu[hmm_idx]),
                    "hmm_pred_mean": extract_hmm_predictive_mean(hmm_obj, hmm_idx, n_trials),
                }
            )

    for group in COND_ORDER:
        print(f"  {group}: {len(grouped_records.get(group, []))} participants")

    return grouped_records


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
    pi_preds = record["hmm_pi_preds"][trial_idx]
    policies = record["hmm_policies"][trial_idx]

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


def angular_energy_score(sample_a, sample_b, observed_aim):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    obs_term = np.mean(np.abs(signed_angular_dist(sample_a - observed_aim)))
    spread_term = 0.5 * np.mean(np.abs(signed_angular_dist(sample_a - sample_b)))
    return obs_term - spread_term


def _stable_seed(text):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _process_pool_available():
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "")
    if not main_file:
        return True
    return "<stdin>" not in str(main_file)


def compute_continuous_metrics_for_record(record, n_samples):
    rng = np.random.default_rng(RNG_SEED + _stable_seed(record["pp"]))
    aims = record["aims"]
    rotation_idx = np.flatnonzero(record["rotation_mask"] & ~np.isnan(aims))
    early_idx = set(np.flatnonzero(record["early_rotation_mask"] & ~np.isnan(aims)))

    bht_rot_scores = []
    hmm_rot_scores = []
    bht_early_scores = []
    hmm_early_scores = []

    for trial_idx in rotation_idx:
        bht_a = sample_bht_predictive(record, trial_idx, rng, n_samples)
        bht_b = sample_bht_predictive(record, trial_idx, rng, n_samples)
        hmm_a = sample_hmm_predictive(record, trial_idx, rng, n_samples)
        hmm_b = sample_hmm_predictive(record, trial_idx, rng, n_samples)

        bht_score = angular_energy_score(bht_a, bht_b, aims[trial_idx])
        hmm_score = angular_energy_score(hmm_a, hmm_b, aims[trial_idx])

        bht_rot_scores.append(bht_score)
        hmm_rot_scores.append(hmm_score)
        if trial_idx in early_idx:
            bht_early_scores.append(bht_score)
            hmm_early_scores.append(hmm_score)

    bht_pred_mean = compute_bht_predictive_mean(record["predstate"], len(aims))
    bht_rotation_r2 = compute_angular_r2(
        aims[record["rotation_mask"]],
        bht_pred_mean[record["rotation_mask"]],
    )
    hmm_rotation_r2 = compute_angular_r2(
        aims[record["rotation_mask"]],
        record["hmm_pred_mean"][record["rotation_mask"]],
    )

    bht_rotation_energy = float(np.nanmean(bht_rot_scores)) if bht_rot_scores else np.nan
    hmm_rotation_energy = float(np.nanmean(hmm_rot_scores)) if hmm_rot_scores else np.nan
    bht_early_rotation_energy = float(np.nanmean(bht_early_scores)) if bht_early_scores else np.nan
    hmm_early_rotation_energy = float(np.nanmean(hmm_early_scores)) if hmm_early_scores else np.nan

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
        "bht_rotation_r2": bht_rotation_r2,
        "hmm_rotation_r2": hmm_rotation_r2,
        "delta_rotation_r2": bht_rotation_r2 - hmm_rotation_r2,
        "n_rotation_trials": int(np.sum(record["rotation_mask"] & ~np.isnan(aims))),
        "n_early_rotation_trials": int(np.sum(record["early_rotation_mask"] & ~np.isnan(aims))),
    }


def compute_continuous_metric_table(
    grouped_records,
    n_samples=ENERGY_SCORE_SAMPLES,
    parallel=True,
    max_workers=DEFAULT_MAX_WORKERS,
):
    all_records = []
    for group in COND_ORDER:
        all_records.extend(grouped_records.get(group, []))

    print(
        f"Computing angular energy scores with {n_samples} samples/trial "
        f"using up to {max_workers if parallel else 1} worker(s)..."
    )
    rows = []
    use_parallel = parallel and len(all_records) > 1 and _process_pool_available()
    if parallel and not use_parallel:
        print(
            "  Process-pool parallelism is unavailable in this launch mode; using serial fallback."
        )
    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                iterator = executor.map(
                    compute_continuous_metrics_for_record,
                    all_records,
                    repeat(n_samples),
                )
                for idx, row in enumerate(iterator, start=1):
                    rows.append(row)
                    if idx % 50 == 0 or idx == len(all_records):
                        print(f"  energy-score progress: {idx}/{len(all_records)} participants")
        except Exception as exc:
            print(f"Parallel energy-score computation failed ({exc}); falling back to serial.")
            rows = [
                compute_continuous_metrics_for_record(record, n_samples) for record in all_records
            ]
    else:
        rows = [compute_continuous_metrics_for_record(record, n_samples) for record in all_records]

    return pd.DataFrame(rows)


def bht_sign_flip_probability(record, trial_idx):
    """P(normalised model aim < 0), after aligning rotation sign convention."""
    predstate = record["predstate"]
    sign_multiplier = float(record.get("sign_multiplier", 1.0))
    n_comp = int(predstate["predNComp"][trial_idx])
    if n_comp <= 0:
        return np.nan
    log_w = predstate["predCompLogProb"][trial_idx, :n_comp]
    weights = np.exp(log_w - logsumexp(log_w))
    means = sign_multiplier * predstate["predCompMean"][trial_idx, :n_comp]
    scale = predstate["predCompScale"][trial_idx, :n_comp]
    nu = predstate["predCompNu"][trial_idx, :n_comp]
    return float(np.sum(weights * student_t.cdf(-means / scale, df=nu)))


def hmm_sign_flip_probability(record, trial_idx):
    """P(normalised HMM aim < 0), after aligning rotation sign convention."""
    sign_multiplier = float(record.get("sign_multiplier", 1.0))
    pi_preds = record["hmm_pi_preds"][trial_idx]
    policies = record["hmm_policies"][trial_idx]
    policy_sum = np.sum(policies)
    if policy_sum <= 0:
        norm_policy = np.zeros_like(policies)
    else:
        norm_policy = policies / policy_sum

    baseline_prob = float(
        student_t.cdf(
            0.0,
            df=record["hmm_baseline_nu"],
            loc=0.0,
            scale=record["hmm_baseline_scale"],
        )
    )
    move_prob = student_t.cdf(
        0.0,
        df=DEFAULT_HMM_ACTION_NU,
        loc=sign_multiplier * ACTION_ANGLES,
        scale=record["hmm_noise"],
    )
    return float(pi_preds[0] * baseline_prob + pi_preds[1] * np.sum(norm_policy * move_prob))


def build_flip_trial_table(
    grouped_records,
    target_mode,
    parallel=True,
    max_workers=DEFAULT_MAX_WORKERS,
):
    all_records = []
    for group in COND_ORDER:
        all_records.extend(grouped_records.get(group, []))

    label = "generalisation" if target_mode == "generalisation" else "training"
    print(f"Computing mechanistic {label} sign-flip rows across {len(all_records)} participants...")
    rows = []
    use_parallel = parallel and len(all_records) > 1 and _process_pool_available()
    if parallel and not use_parallel:
        print(
            "  Process-pool parallelism is unavailable in this launch mode; using serial fallback."
        )
    if use_parallel:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                iterator = executor.map(
                    build_flip_rows_for_record,
                    all_records,
                    repeat(target_mode),
                )
                for idx, participant_rows in enumerate(iterator, start=1):
                    rows.extend(participant_rows)
                    if idx % 100 == 0 or idx == len(all_records):
                        print(f"  sign-flip progress: {idx}/{len(all_records)} participants")
        except Exception as exc:
            print(f"Parallel sign-flip computation failed ({exc}); falling back to serial.")
            for record in all_records:
                rows.extend(build_flip_rows_for_record(record, target_mode))
    else:
        for record in all_records:
            rows.extend(build_flip_rows_for_record(record, target_mode))

    return pd.DataFrame(rows)


def build_flip_rows_for_record(record, target_mode):
    """Build sign-flip rows for one participant.

    A participant-specific sign multiplier is applied first so correct
    compensation is positive regardless of raw rotation direction. A sign-flip
    error is then simply normalised aim < 0.
    """
    rows = []
    df_pp = record["df_pp"]
    sign_multiplier = float(record.get("sign_multiplier", 1.0))
    aim_col = "aim_signed" if "aim_signed" in df_pp.columns else "aim"
    if target_mode == "generalisation":
        target_mask = (df_pp["rotation"] != 0) & (df_pp["trialStatus"] == "generalisation")
    elif target_mode == "training":
        target_mask = (
            (df_pp["rotation"] != 0)
            & (df_pp["trialStatus"] != "generalisation")
            & (df_pp["Cursor FB"] != "no_fb")
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    target_indices = np.flatnonzero(target_mask.to_numpy())

    for target_order, trial_idx in enumerate(target_indices, start=1):
        aim_val = float(df_pp.iloc[trial_idx][aim_col])
        if aim_col != "aim_signed":
            aim_val *= sign_multiplier
        if np.isnan(aim_val) or aim_val == 0:
            continue

        human_flip = float(aim_val < 0)
        bht_flip_p = bht_sign_flip_probability(record, trial_idx)
        hmm_flip_p = hmm_sign_flip_probability(record, trial_idx)
        if np.isnan(bht_flip_p) or np.isnan(hmm_flip_p):
            continue

        bht_brier = float((bht_flip_p - human_flip) ** 2)
        hmm_brier = float((hmm_flip_p - human_flip) ** 2)

        rows.append(
            {
                "pp": record["pp"],
                "condition": record["condition"],
                "group": record["group"],
                "trial_idx": int(trial_idx),
                "target_mode": target_mode,
                "target_trial_index": int(target_order),
                "human_flip": human_flip,
                "bht_flip_p": float(bht_flip_p),
                "hmm_flip_p": float(hmm_flip_p),
                "bht_brier": bht_brier,
                "hmm_brier": hmm_brier,
                "delta_brier": hmm_brier - bht_brier,
            }
        )

    return rows


def summarize_sign_flip_metrics(flip_trial_df, early_cutoff=EARLY_GENERALISATION_TRIALS):
    rows = []

    for (group, pp), sub in flip_trial_df.groupby(["group", "pp"]):
        sub = sub.sort_values("target_trial_index").reset_index(drop=True)
        early = sub.loc[sub["target_trial_index"] <= early_cutoff]
        bht_flip_brier_all = float(sub["bht_brier"].mean())
        hmm_flip_brier_all = float(sub["hmm_brier"].mean())
        bht_flip_brier_early = float(early["bht_brier"].mean()) if len(early) > 0 else np.nan
        hmm_flip_brier_early = float(early["hmm_brier"].mean()) if len(early) > 0 else np.nan
        rows.append(
            {
                "pp": pp,
                "condition": sub["condition"].iloc[0],
                "group": group,
                "bht_flip_brier_all": bht_flip_brier_all,
                "hmm_flip_brier_all": hmm_flip_brier_all,
                "delta_flip_brier_all": hmm_flip_brier_all - bht_flip_brier_all,
                "bht_flip_brier_early": bht_flip_brier_early,
                "hmm_flip_brier_early": hmm_flip_brier_early,
                "delta_flip_brier_early": (
                    hmm_flip_brier_early - bht_flip_brier_early
                    if not np.isnan(bht_flip_brier_early) and not np.isnan(hmm_flip_brier_early)
                    else np.nan
                ),
                "n_gen_trials": int(len(sub)),
                "n_early_gen_trials": int(np.sum(sub["target_trial_index"] <= early_cutoff)),
            }
        )

    return pd.DataFrame(rows)


def build_combined_participant_table(continuous_df, flip_df):
    return continuous_df.merge(
        flip_df,
        on=["pp", "condition", "group"],
        how="left",
    )


def save_figure_bundle(fig, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in (".png", ".svg"):
        out_path = base_path.with_suffix(ext)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        outputs[ext.lstrip(".")] = str(out_path)
    return outputs


def _plot_bic_style_panel(ax, participant_df, value_col, ylabel, title):
    positions = np.arange(len(COND_ORDER))
    labels = []

    ax.axhline(0.0, color="black", ls="--", lw=1.1, zorder=1)
    ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)

    for ci, group in enumerate(COND_ORDER):
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
            s=18,
            alpha=0.35,
            color="#555555",
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
            color="#d62728",
            markersize=8,
            capsize=5,
            capthick=1.5,
            lw=1.5,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)

    ymin, ymax = ax.get_ylim()
    ax.fill_between(
        [-0.5, len(COND_ORDER) - 0.5],
        0,
        ymax,
        color="#d4edda",
        alpha=0.15,
        zorder=0,
    )
    ax.fill_between(
        [-0.5, len(COND_ORDER) - 0.5],
        ymin,
        0,
        color="#f8d7da",
        alpha=0.15,
        zorder=0,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        len(COND_ORDER) - 0.55,
        ymax - 0.06 * (ymax - ymin),
        "BHT better",
        ha="right",
        va="top",
        fontsize=FS_LEGEND,
        color="#155724",
        style="italic",
    )
    ax.text(
        len(COND_ORDER) - 0.55,
        ymin + 0.06 * (ymax - ymin),
        "HMM better",
        ha="right",
        va="bottom",
        fontsize=FS_LEGEND,
        color="#721c24",
        style="italic",
    )
    ax.tick_params(labelsize=FS_TICK)
    sns.despine(ax=ax)


def _plot_absolute_metric_panel(
    ax, participant_df, bht_col, hmm_col, ylabel, title, lower_is_better
):
    width = 0.16
    group_centers = np.arange(len(COND_ORDER))
    labels = []

    for ci, group in enumerate(COND_ORDER):
        group_df = (
            participant_df.loc[participant_df["group"] == group].copy().reset_index(drop=True)
        )
        bht_only_df = group_df.loc[group_df[bht_col].notna() & group_df[hmm_col].isna()].copy()
        hmm_only_df = group_df.loc[group_df[bht_col].isna() & group_df[hmm_col].notna()].copy()
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

        if len(paired_df) > 0:
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

        if len(bht_only_df) > 0:
            jitter = np.random.default_rng(400 + ci).uniform(-0.06, 0.06, len(bht_only_df))
            x_pos = ci - width
            ax.scatter(
                x_pos + jitter,
                bht_only_df[bht_col].to_numpy(dtype=float),
                s=18,
                alpha=0.30,
                color=BHT_COLOR,
                edgecolors="none",
                zorder=2.1,
            )

        bht_values = group_df[bht_col].dropna().to_numpy(dtype=float)
        if len(bht_values) > 0:
            mean_val = float(np.mean(bht_values))
            sem_val = float(np.std(bht_values, ddof=0) / np.sqrt(len(bht_values)))
            ax.errorbar(
                ci - width,
                mean_val,
                yerr=1.96 * sem_val,
                fmt="o",
                color=BHT_COLOR,
                markersize=8,
                markerfacecolor=BHT_COLOR,
                markeredgecolor="black",
                markeredgewidth=1.0,
                capsize=5,
                capthick=1.5,
                lw=1.5,
                zorder=3,
            )

        if len(hmm_only_df) > 0:
            jitter = np.random.default_rng(500 + ci).uniform(-0.06, 0.06, len(hmm_only_df))
            x_pos = ci + width
            ax.scatter(
                x_pos + jitter,
                hmm_only_df[hmm_col].to_numpy(dtype=float),
                s=18,
                alpha=0.30,
                color=HMM_COLOR,
                edgecolors="none",
                zorder=2.1,
            )

        hmm_values = group_df[hmm_col].dropna().to_numpy(dtype=float)
        if len(hmm_values) > 0:
            mean_val = float(np.mean(hmm_values))
            sem_val = float(np.std(hmm_values, ddof=0) / np.sqrt(len(hmm_values)))
            ax.errorbar(
                ci + width,
                mean_val,
                yerr=1.96 * sem_val,
                fmt="o",
                color="#4d4d4d",
                markersize=8,
                markerfacecolor=HMM_COLOR,
                markeredgecolor="black",
                markeredgewidth=1.0,
                capsize=5,
                capthick=1.5,
                lw=1.5,
                zorder=3,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)
    hint = "Lower is better" if lower_is_better else "Higher is better"
    ax.text(
        0.99,
        0.98,
        hint,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS_LEGEND,
        color="#444444",
        style="italic",
    )
    sns.despine(ax=ax)


def figure_main_candidate(participant_df, save_dir):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _plot_bic_style_panel(
        ax1,
        participant_df,
        "delta_rotation_energy",
        "Delta angular energy score\n(HMM - BHT)",
        "A. Full predictive score: rotation phase",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_bic_style_panel(
        ax2,
        participant_df,
        "delta_early_rotation_energy",
        f"Delta angular energy score\n(HMM - BHT), first {EARLY_ROTATION_TRIALS} rotation trials",
        "B. Full predictive score: early rotation",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_bic_style_panel(
        ax3,
        participant_df,
        "delta_flip_brier_all",
        "Delta sign-flip Brier score\n(HMM - BHT)",
        "C. Mechanistic score: generalisation sign flips",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    _plot_bic_style_panel(
        ax4,
        participant_df,
        "delta_rotation_r2",
        "Delta R^2 (BHT - HMM)\nrotation phase only",
        "D. Point-prediction summary: rotation phase",
    )

    fig.suptitle(
        "Ding: mechanistic comparison without BIC or NegLL",
        fontsize=12,
        y=0.98,
    )

    return save_figure_bundle(fig, Path(save_dir) / "01_main_candidate")


def figure_flip_brier_early(participant_df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _plot_bic_style_panel(
        axes[0],
        participant_df,
        "delta_flip_brier_early",
        f"Delta sign-flip Brier score\n(HMM - BHT), first {EARLY_GENERALISATION_TRIALS} generalisation trials",
        "A. Early mechanistic sign-flip score",
    )
    _plot_absolute_metric_panel(
        axes[1],
        participant_df,
        "bht_flip_brier_early",
        "hmm_flip_brier_early",
        f"Sign-flip Brier score\nfirst {EARLY_GENERALISATION_TRIALS} generalisation trials",
        "B. Early mechanistic sign-flip score, absolute",
        lower_is_better=True,
    )
    return save_figure_bundle(fig, Path(save_dir) / "05_flip_brier_early")


def figure_absolute_metrics(participant_df, save_dir):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _plot_absolute_metric_panel(
        ax1,
        participant_df,
        "bht_rotation_energy",
        "hmm_rotation_energy",
        "Angular energy score",
        "A. Rotation-phase energy score",
        lower_is_better=True,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_absolute_metric_panel(
        ax2,
        participant_df,
        "bht_early_rotation_energy",
        "hmm_early_rotation_energy",
        f"Angular energy score\nfirst {EARLY_ROTATION_TRIALS} rotation trials",
        "B. Early rotation energy score",
        lower_is_better=True,
    )

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_absolute_metric_panel(
        ax3,
        participant_df,
        "bht_flip_brier_all",
        "hmm_flip_brier_all",
        "Sign-flip Brier score",
        "C. Generalisation sign-flip Brier score",
        lower_is_better=True,
    )

    ax4 = fig.add_subplot(gs[1, 1])
    _plot_absolute_metric_panel(
        ax4,
        participant_df,
        "bht_rotation_r2",
        "hmm_rotation_r2",
        "Rotation-phase R^2",
        "D. Rotation-phase R^2",
        lower_is_better=False,
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color=BHT_COLOR, lw=0, markersize=8, label="BHT"),
        plt.Line2D([0], [0], marker="o", color="#4d4d4d", lw=0, markersize=8, label="HMM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=FS_LEGEND)
    fig.suptitle("Absolute metric panels for BHT and HMM", fontsize=12, y=0.98)
    return save_figure_bundle(fig, Path(save_dir) / "02_absolute_metric_panels")


def figure_r2_rotation(participant_df, save_dir):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _plot_bic_style_panel(
        ax,
        participant_df,
        "delta_rotation_r2",
        "Delta R^2 (BHT - HMM)\nrotation phase only",
        "Rotation-phase R^2",
    )
    return save_figure_bundle(fig, Path(save_dir) / "03_rotation_r2")


def figure_flip_trajectories(flip_trial_df, save_dir, target_mode, basename, suptitle):
    summary = (
        flip_trial_df.groupby(["group", "target_trial_index"], as_index=False)
        .agg(
            mean_human_flip=("human_flip", "mean"),
            mean_bht_flip_p=("bht_flip_p", "mean"),
            mean_hmm_flip_p=("hmm_flip_p", "mean"),
            n=("pp", "nunique"),
        )
        .sort_values(["group", "target_trial_index"])
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, group in zip(axes, COND_ORDER):
        sub = summary.loc[summary["group"] == group].copy()
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        ax.plot(
            sub["target_trial_index"],
            sub["mean_human_flip"],
            color=HUMAN_COLOR,
            lw=0.0,
            ls="None",
            marker="o",
            alpha=0.72,
            label="Human flip rate",
        )
        ax.plot(
            sub["target_trial_index"],
            sub["mean_bht_flip_p"],
            color=BHT_COLOR,
            lw=0.0,
            ls="None",
            marker="o",
            alpha=0.72,
            label="BHT predicted flip p",
        )
        ax.plot(
            sub["target_trial_index"],
            sub["mean_hmm_flip_p"],
            color=HMM_COLOR,
            lw=0.0,
            ls="None",
            marker="s",
            alpha=0.72,
            label="HMM predicted flip p",
        )

        bht_r2 = compute_scalar_r2(
            sub["mean_human_flip"],
            sub["mean_bht_flip_p"],
        )
        hmm_r2 = compute_scalar_r2(
            sub["mean_human_flip"],
            sub["mean_hmm_flip_p"],
        )
        bht_rmse = compute_rmse(
            sub["mean_human_flip"],
            sub["mean_bht_flip_p"],
        )
        hmm_rmse = compute_rmse(
            sub["mean_human_flip"],
            sub["mean_hmm_flip_p"],
        )
        stat_lines = []
        stat_lines.append("BHT $R^2$ = " + ("nan" if np.isnan(bht_r2) else f"{bht_r2:.3f}"))
        stat_lines.append("BHT RMSE = " + ("nan" if np.isnan(bht_rmse) else f"{bht_rmse:.3f}"))
        stat_lines.append("HMM $R^2$ = " + ("nan" if np.isnan(hmm_r2) else f"{hmm_r2:.3f}"))
        stat_lines.append("HMM RMSE = " + ("nan" if np.isnan(hmm_rmse) else f"{hmm_rmse:.3f}"))
        ax.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_LEGEND,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9),
        )

        ax.set_title(group, fontsize=FS_TITLE)
        xlabel = (
            "Generalisation trial index"
            if target_mode == "generalisation"
            else "Training trial index"
        )
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
        ax.set_ylabel("Probability / rate", fontsize=FS_LABEL)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=FS_TICK,
            bottom=True,
            left=True,
            length=4,
            width=0.9,
            direction="out",
        )
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=FS_LEGEND)
    fig.suptitle(suptitle, fontsize=12, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure_bundle(fig, Path(save_dir) / basename)


def figure_generalisation_trajectories(flip_trial_df, save_dir):
    sub = flip_trial_df.loc[flip_trial_df["target_mode"] == "generalisation"].copy()
    return figure_flip_trajectories(
        sub,
        save_dir,
        target_mode="generalisation",
        basename="04_generalisation_trajectories",
        suptitle="Mechanistic option: predicted vs observed sign flips on generalisation trials",
    )


def figure_training_trajectories(flip_trial_df, save_dir):
    sub = flip_trial_df.loc[flip_trial_df["target_mode"] == "training"].copy()
    return figure_flip_trajectories(
        sub,
        save_dir,
        target_mode="training",
        basename="06_training_target_trajectories",
        suptitle="Mechanistic option: predicted vs observed sign flips on training targets",
    )


def save_tables(participant_df, flip_trial_df, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    participant_path = save_dir / "participant_metrics.csv"
    flip_path = save_dir / "flip_trials.csv"
    participant_df.to_csv(participant_path, index=False)
    flip_trial_df.to_csv(flip_path, index=False)
    return {
        "participant_metrics_csv": str(participant_path),
        "flip_trials_csv": str(flip_path),
    }


def main(
    save_dir="BLRFigures/DingFigures/mechanistic_comparison",
    n_samples=ENERGY_SCORE_SAMPLES,
    max_workers=DEFAULT_MAX_WORKERS,
    parallel=True,
):
    sns.set_theme(style="white", context="notebook")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grouped_records = build_participant_records()
    continuous_df = compute_continuous_metric_table(
        grouped_records,
        n_samples=n_samples,
        parallel=parallel,
        max_workers=max_workers,
    )
    generalisation_flip_trial_df = build_flip_trial_table(
        grouped_records,
        target_mode="generalisation",
        parallel=parallel,
        max_workers=max_workers,
    )
    training_flip_trial_df = build_flip_trial_table(
        grouped_records,
        target_mode="training",
        parallel=parallel,
        max_workers=max_workers,
    )
    flip_trial_df = pd.concat(
        [generalisation_flip_trial_df, training_flip_trial_df],
        ignore_index=True,
    )
    flip_metric_df = summarize_sign_flip_metrics(generalisation_flip_trial_df)
    participant_df = build_combined_participant_table(continuous_df, flip_metric_df)

    table_paths = save_tables(participant_df, flip_trial_df, save_dir)
    figure_paths = {
        "main_candidate": figure_main_candidate(participant_df, save_dir),
        "absolute_metrics": figure_absolute_metrics(participant_df, save_dir),
        "rotation_r2": figure_r2_rotation(participant_df, save_dir),
        "generalisation_trajectories": figure_generalisation_trajectories(flip_trial_df, save_dir),
        "flip_brier_early": figure_flip_brier_early(participant_df, save_dir),
        "training_trajectories": figure_training_trajectories(flip_trial_df, save_dir),
    }

    print("\nSaved Ding outputs to:", save_dir)
    for key, path_map in figure_paths.items():
        print(f"  {key}: {path_map.get('png', '')}")
    for key, path_str in table_paths.items():
        print(f"  {key}: {path_str}")

    return {
        "grouped_records": grouped_records,
        "participant_df": participant_df,
        "continuous_df": continuous_df,
        "flip_trial_df": flip_trial_df,
        "generalisation_flip_trial_df": generalisation_flip_trial_df,
        "training_flip_trial_df": training_flip_trial_df,
        "flip_metric_df": flip_metric_df,
        "figure_paths": figure_paths,
        "table_paths": table_paths,
    }


if __name__ == "__main__":
    DING_RESULTS = main()
