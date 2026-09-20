from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from scipy.special import logsumexp

import BayesHypothesisTesting as BHT


def extract_fit_shells(loaded: Any) -> List[Any]:
    if isinstance(loaded, np.ndarray):
        loaded = loaded.item() if loaded.ndim == 0 else list(loaded)
    if isinstance(loaded, dict):
        out: List[Any] = []
        for value in loaded.values():
            if isinstance(value, list):
                out.extend(value)
            else:
                out.append(value)
        return out
    if isinstance(loaded, list):
        out = []
        for value in loaded:
            if isinstance(value, list):
                out.extend(value)
            else:
                out.append(value)
        return out
    return [loaded]


def _fit_model_config(fit: Any) -> Dict[str, Any]:
    cfg = getattr(fit, "modelConfig", None)
    if cfg:
        return {
            "n_harm": int(cfg["n_harm"]),
            "a_hz": float(cfg["a_hz"]),
            "b_hz": float(cfg["b_hz"]),
            "maxrun": int(cfg["maxrun"]),
            "n_struct": int(cfg["n_struct"]),
        }
    overrides = dict(getattr(fit, "_hyper_overrides", {}) or {})
    return {k: v for k, v in overrides.items() if v is not None}


def _participant_index_map(fit: Any) -> Dict[int, int]:
    return {int(pp): idx for idx, pp in enumerate(fit.participantNums)}


def _participant_forward_inputs(fit: Any, pp: int) -> Dict[str, Any]:
    idx_map = _participant_index_map(fit)
    if int(pp) not in idx_map:
        raise KeyError(f"Participant {pp} not found in fit shell.")
    p_idx = idx_map[int(pp)]
    xs = fit.xs[p_idx]

    pdat_full = fit.df[fit.df["participantNum"] == pp].copy().reset_index(drop=True)
    phases_full = BHT.derivePhase(pdat_full)
    include_mask = (
        np.ones(len(phases_full), dtype=bool)
        if getattr(fit, "fitWashout", False)
        else (phases_full != "washout")
    )
    pdat = pdat_full[include_mask].reset_index(drop=True)
    phases = phases_full[include_mask]

    gen_targets = BHT.identifyGeneralisationTargets(pdat_full)
    train_targets = BHT.identifyTrainingTargets(pdat_full, gen_targets)
    dest_targets = sorted(set(train_targets) | gen_targets)

    targets = pdat["targetPosition"].to_numpy(dtype=np.float64)
    rotations = pdat["rotation"].to_numpy(dtype=np.float64)
    trials = np.arange(len(pdat), dtype=np.int64)
    has_feedback = BHT.buildHasFeedback(pdat)
    unique_targets = np.unique(targets[np.isfinite(targets)])

    return {
        "params": np.asarray(xs[2:], dtype=np.float64),
        "scale_S0": float(xs[0]),
        "nu_S0": float(xs[1]),
        "trials": trials,
        "targets": targets,
        "rotations": rotations,
        "has_feedback": has_feedback,
        "dest_targets": np.asarray(dest_targets, dtype=np.float64),
        "unique_targets": np.asarray(unique_targets, dtype=np.float64),
        "phases": np.asarray(phases),
        "participant_num": int(pp),
    }


def _run_limit_from_diag(log_r_row: np.ndarray) -> int:
    run_limit = 1
    for k in range(1, len(log_r_row)):
        if log_r_row[k] > -700.0 or log_r_row[k - 1] > -700.0:
            run_limit = k + 1
    return run_limit


def _recover_component_masses(result: Any, n_struct: int) -> Dict[str, np.ndarray]:
    (
        _negll,
        q0s,
        run_limits,
        p_k_arrs,
        p_changes,
        pred_entropy,
        post_log_q0,
        post_log_r,
        a_hz_store,
        b_hz_store,
        pre_log_q0,
        pre_log_r,
        pred_comp_log_prob,
        pred_comp_mean,
        pred_comp_scale,
        pred_comp_nu,
        pred_n_comp,
        pred_self_other_w_legacy,
        pred_c_pred,
        pred_d_pred,
        pred_self_count,
        pred_cross_dot,
        pred_s_vals,
        pred_gibbs_w,
        pred_obs_var,
        pred_tgt_idx,
        pred_struct_log_w,
        diag_d_h,
        diag_c_h,
        diag_log_r_h,
        diag_log_q0_h,
        diag_log_marg_per_hyp,
        diag_log_marg_bocpd,
        pred_log_zm1,
        pred_log_pm1,
    ) = result

    num_trials = pred_comp_log_prob.shape[0]
    m1_self = np.full(num_trials, np.nan)
    m1_other = np.full(num_trials, np.nan)
    m0_prob = np.full(num_trials, np.nan)
    m1_prob = np.full(num_trials, np.nan)
    legacy_other = np.full(num_trials, np.nan)
    legacy_self = np.full(num_trials, np.nan)

    legacy_run_probs = np.exp(pre_log_r - logsumexp(pre_log_r, axis=1, keepdims=True))
    legacy_self = np.sum(legacy_run_probs * pred_self_other_w_legacy[:, :, 0], axis=1)
    legacy_other = np.sum(legacy_run_probs * pred_self_other_w_legacy[:, :, 1], axis=1)

    for t in range(num_trials):
        comp_count = int(pred_n_comp[t])
        if comp_count <= 0:
            continue

        all_log_w = pred_comp_log_prob[t, :comp_count]
        log_all = logsumexp(all_log_w)

        log_m0_terms: List[float] = []
        log_m1_terms: List[float] = []
        log_self_terms: List[float] = []
        log_other_terms: List[float] = []

        # Each hypothesis stores M0, then self/other pairs per run (one prior for unknown targets).
        comp_idx = 0
        has_valid_target = int(pred_tgt_idx[t]) >= 0
        for m in range(n_struct):
            log_m0_terms.append(pred_comp_log_prob[t, comp_idx])
            comp_idx += 1

            run_limit = _run_limit_from_diag(diag_log_r_h[t, m])
            if has_valid_target:
                for _k in range(run_limit):
                    log_self = pred_comp_log_prob[t, comp_idx]
                    log_other = pred_comp_log_prob[t, comp_idx + 1]
                    log_self_terms.append(log_self)
                    log_other_terms.append(log_other)
                    log_m1_terms.extend((log_self, log_other))
                    comp_idx += 2
            else:
                for _k in range(run_limit):
                    log_m1_terms.append(pred_comp_log_prob[t, comp_idx])
                    comp_idx += 1

        if comp_idx != comp_count:
            raise ValueError(
                f"Predictive component parse mismatch at trial {t}: parsed {comp_idx}, expected {comp_count}."
            )

        log_m0 = logsumexp(np.asarray(log_m0_terms, dtype=np.float64))
        m0_prob[t] = float(np.exp(log_m0 - log_all))

        if log_m1_terms:
            log_m1 = logsumexp(np.asarray(log_m1_terms, dtype=np.float64))
            m1_prob[t] = float(np.exp(log_m1 - log_all))
            # Self/other weights are conditional on M1, excluding baseline mass.
            if has_valid_target and log_self_terms and log_other_terms:
                m1_self[t] = float(
                    np.exp(logsumexp(np.asarray(log_self_terms, dtype=np.float64)) - log_m1)
                )
                m1_other[t] = float(
                    np.exp(logsumexp(np.asarray(log_other_terms, dtype=np.float64)) - log_m1)
                )

    return {
        "m1_self_weight": m1_self,
        "m1_other_weight": m1_other,
        "m0_prob": m0_prob,
        "m1_prob": m1_prob,
        "legacy_h0_self_weight": legacy_self,
        "legacy_h0_other_weight": legacy_other,
        "q0s": np.asarray(q0s),
        "run_limits": np.asarray(run_limits),
        "p_k_arrs": np.asarray(p_k_arrs),
        "p_changes": np.asarray(p_changes),
        "pred_struct_log_w": np.asarray(pred_struct_log_w),
        "pred_tgt_idx": np.asarray(pred_tgt_idx),
    }


def recover_true_self_other_for_participant(fit: Any, pp: int) -> Dict[str, Any]:
    cfg = _fit_model_config(fit)
    if cfg:
        BHT._configure_globals(**cfg)

    inputs = _participant_forward_inputs(fit, pp)
    num_trials = len(inputs["trials"])
    is_rotation = np.asarray(
        [phase.lower() == "rotation" for phase in inputs["phases"]], dtype=np.bool_
    )

    result = BHT.computeNegLl(
        inputs["params"],
        np.zeros(num_trials, dtype=np.float64),
        np.zeros(num_trials, dtype=np.bool_),
        inputs["has_feedback"],
        inputs["trials"],
        is_rotation,
        inputs["rotations"],
        inputs["targets"],
        inputs["unique_targets"],
        inputs["scale_S0"],
        inputs["nu_S0"],
        inputs["dest_targets"],
        len(inputs["dest_targets"]),
        computeEntropy=False,
    )

    recovered = _recover_component_masses(result, int(cfg.get("n_struct", BHT.N_STRUCT)))
    recovered.update(
        {
            "participant_num": int(pp),
            "trial_index": inputs["trials"].copy(),
            "phase": inputs["phases"].copy(),
            "rotation": inputs["rotations"].copy(),
            "target_position": inputs["targets"].copy(),
            "dest_targets": inputs["dest_targets"].copy(),
            "scale_S0": inputs["scale_S0"],
            "nu_S0": inputs["nu_S0"],
            "params": inputs["params"].copy(),
        }
    )
    return recovered


def recover_true_self_other_for_fit(fit: Any) -> Dict[str, Any]:
    cfg = _fit_model_config(fit)
    if cfg:
        BHT._configure_globals(**cfg)

    out: Dict[str, Any] = {
        "dataset_name": getattr(fit, "datasetName", "dataset"),
        "model_config": cfg,
        "participants": {},
    }

    for pp in fit.participantNums:
        out["participants"][int(pp)] = recover_true_self_other_for_participant(fit, int(pp))
    return out


def recover_true_self_other_for_loaded(loaded: Any) -> List[Dict[str, Any]]:
    return [recover_true_self_other_for_fit(fit) for fit in extract_fit_shells(loaded)]


def save_recovered_weights(input_path: Path, output_path: Path) -> Path:
    loaded = np.load(input_path, allow_pickle=True)
    recovered = recover_true_self_other_for_loaded(loaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(recovered, handle)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover hypothesis-marginalized self/other weights from saved BHT fits."
    )
    parser.add_argument(
        "fit_paths",
        nargs="+",
        help="One or more .npy fit files produced by BayesHypothesisTesting.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output pickle files. Defaults to the input file directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None

    for raw_path in args.fit_paths:
        input_path = Path(raw_path).resolve()
        if out_dir is None:
            output_path = input_path.with_name(f"{input_path.stem}_true_self_other.pkl")
        else:
            output_path = out_dir / f"{input_path.stem}_true_self_other.pkl"
        saved_to = save_recovered_weights(input_path, output_path)
        print(f"Saved {saved_to}")


if __name__ == "__main__":
    main()
