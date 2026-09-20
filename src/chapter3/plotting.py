"""Plot predictive distributions, learning curves and model diagnostics."""

import BayesHypothesisTesting as BHT

from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import inspect
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import os

try:
    import seaborn as sns
except ImportError:
    pass

importlib.reload(BHT)


def _infer_expected_nparams():
    _sp_src = inspect.getsource(BHT.samplePredictive.py_func)
    for _line in _sp_src.splitlines():
        _line = _line.strip()
        if '= params' in _line and ',' in _line:
            return _line.split('= params')[0].count(',') + 1
    return None


_EXPECTED_NPARAMS = _infer_expected_nparams()
_BASE_BHT_MODEL_CONFIG = dict(
    BHT.DEFAULT_MODEL_CONFIG
    if hasattr(BHT, 'DEFAULT_MODEL_CONFIG')
    else BHT._current_model_config()
    if hasattr(BHT, '_current_model_config')
    else {
        'n_harm': int(getattr(BHT, 'N_HARM', 1)),
        'a_hz': float(getattr(BHT, 'A_HZ', 1.0)),
        'b_hz': float(getattr(BHT, 'B_HZ', 100.0)),
        'maxrun': int(getattr(BHT, 'MAXRUN', 3)),
        'n_struct': int(getattr(BHT, 'N_STRUCT', 2)),
    }
)
print(
    f"BHT.samplePredictive expects {_EXPECTED_NPARAMS} params "
    f"(xs should have {_EXPECTED_NPARAMS + 2} elements: scale_S0 + nu_S0 + params)"
)

# Notebook reloads must not wrap savefig repeatedly.
if not getattr(mpl_figure.Figure.savefig, '_dual_png_svg_patch', False):
    _ORIG_FIG_SAVEFIG = mpl_figure.Figure.savefig

    def _savefig_png_and_svg(self, fname, *args, **kwargs):
        """Save every figure to both .png and .svg siblings."""
        if getattr(self, '_dual_save_in_progress', False):
            return _ORIG_FIG_SAVEFIG(self, fname, *args, **kwargs)
        if not isinstance(fname, (str, os.PathLike)):
            return _ORIG_FIG_SAVEFIG(self, fname, *args, **kwargs)
        path = Path(fname)
        ext = path.suffix.lower()
        if ext:
            base = path.with_suffix('')
            requested = path
        else:
            base = path
            requested = base.with_suffix('.png')
        targets = [requested]
        for suffix in ('.png', '.svg'):
            alt = base.with_suffix(suffix)
            if alt not in targets:
                targets.append(alt)
        result = None
        self._dual_save_in_progress = True
        try:
            for idx, target in enumerate(targets):
                out = _ORIG_FIG_SAVEFIG(self, target, *args, **kwargs)
                if idx == 0:
                    result = out
            return result
        finally:
            self._dual_save_in_progress = False

    _savefig_png_and_svg._dual_png_svg_patch = True
    mpl_figure.Figure.savefig = _savefig_png_and_svg


def extract_fit_shells(loaded):
    if isinstance(loaded, np.ndarray):
        loaded = loaded.item() if loaded.ndim == 0 else list(loaded)
    if isinstance(loaded, dict):
        out = []
        for v in loaded.values():
            out.extend(v) if isinstance(v, list) else out.append(v)
        return out
    if isinstance(loaded, list):
        out = []
        for v in loaded:
            out.extend(v) if isinstance(v, list) else out.append(v)
        return out
    return [loaded]


def _is_shared_fit(obj):
    return hasattr(obj, 'bestParams') and not hasattr(obj, 'genTargets')


def _group_condition_key(datasetName):
    """Map a datasetName like 'EqMeans_Inner_2T_90' -> 'Inner_2T'."""
    parts = datasetName.split('_')
    core = [p for p in parts if p not in ('Ding', 'EqMeans')]
    if len(core) >= 2:
        return '_'.join(core[:-1])
    return datasetName


def _aggregate_obj_groups(obj_list):
    """Group FitShell objects by condition type (e.g. Inner_2T)."""
    groups = {}
    for obj in obj_list:
        name = getattr(obj, 'datasetName', '')
        gk = _group_condition_key(name)
        groups.setdefault(gk, []).append(obj)
    return list(groups.items())


def _default_bht_model_config():
    return dict(_BASE_BHT_MODEL_CONFIG)


_LAST_BHT_MODEL_CONFIG = None


def _infer_fit_maxrun(obj):
    ps_map = getattr(obj, 'predState', None)
    if isinstance(ps_map, dict) and len(ps_map) > 0:
        ps = next(iter(ps_map.values()))
        for key in ('predSelfOtherW', 'predCPred', 'postLogR'):
            val = ps.get(key) if isinstance(ps, dict) else None
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.ndim >= 2 and arr.shape[1] > 0:
                return int(arr.shape[1])
        run_limits = ps.get('runLimits') if isinstance(ps, dict) else None
        if run_limits is not None:
            arr = np.asarray(run_limits)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                return int(np.max(finite))
    return None


def _infer_fit_n_struct(obj):
    ps_map = getattr(obj, 'predState', None)
    if isinstance(ps_map, dict) and len(ps_map) > 0:
        ps = next(iter(ps_map.values()))
        for key in ('predStructLogW', 'diagLogR_h'):
            val = ps.get(key) if isinstance(ps, dict) else None
            if val is None:
                continue
            arr = np.asarray(val)
            if key == 'predStructLogW' and arr.ndim >= 2 and arr.shape[1] > 0:
                return int(arr.shape[1])
            if key == 'diagLogR_h' and arr.ndim >= 3 and arr.shape[1] > 0:
                return int(arr.shape[1])
    return None


def _get_fit_model_config(obj):
    cfg = _default_bht_model_config()
    explicit = set()
    model_cfg = getattr(obj, 'modelConfig', None)
    if model_cfg:
        cfg.update({k: v for k, v in model_cfg.items() if v is not None})
        explicit.update(k for k, v in model_cfg.items() if v is not None)
    overrides = getattr(obj, '_hyper_overrides', None)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        explicit.update(k for k, v in overrides.items() if v is not None)
    inferred_maxrun = _infer_fit_maxrun(obj)
    if inferred_maxrun is not None and 'maxrun' not in explicit:
        cfg['maxrun'] = inferred_maxrun
    inferred_n_struct = _infer_fit_n_struct(obj)
    if inferred_n_struct is not None and 'n_struct' not in explicit:
        cfg['n_struct'] = inferred_n_struct
    return cfg


def _apply_fit_model_config(obj, debug=False):
    global _LAST_BHT_MODEL_CONFIG
    cfg = _get_fit_model_config(obj)
    if hasattr(BHT, '_configure_globals') and cfg != _LAST_BHT_MODEL_CONFIG:
        BHT._configure_globals(**cfg)
        _LAST_BHT_MODEL_CONFIG = dict(cfg)
        if debug:
            print(f"Configured BHT for {getattr(obj, 'datasetName', 'dataset')}: {cfg}")
    return cfg


class _MergedObj:
    """Lightweight merged FitShell for grouped plotting."""

    pass


def _merge_obj_group(group_key, group_objs):
    """Merge multiple FitShell objects into one for grouped plotting."""
    if len(group_objs) == 1:
        return group_objs[0]
    merged = _MergedObj()
    merged.datasetName = group_key
    merged.conVal = 'grouped'
    merged.fitWashout = getattr(group_objs[0], 'fitWashout', True)
    merged.df = pd.concat([o.df for o in group_objs], ignore_index=True)
    merged.participantNums = []
    merged.xs = []
    merged.bics = []
    merged.predState = {}
    merged.genTargets = {}
    merged.destTargets = {}
    merged.trainTargets = {}
    merged.trialStatuses = {}
    for _cfg_attr in ('modelConfig', '_hyper_overrides'):
        for _src in group_objs:
            _cfg_val = getattr(_src, _cfg_attr, None)
            if _cfg_val:
                setattr(merged, _cfg_attr, dict(_cfg_val))
                break
    for o in group_objs:
        for i, pp in enumerate(o.participantNums):
            if pp not in merged.predState:
                merged.participantNums.append(pp)
                merged.xs.append(o.xs[i])
                if hasattr(o, 'bics') and o.bics is not None:
                    merged.bics.append(o.bics[i])
                if hasattr(o, 'predState') and o.predState is not None:
                    ps = o.predState.get(pp)
                    if ps is not None:
                        merged.predState[pp] = ps
                for attr in ('genTargets', 'destTargets', 'trainTargets', 'trialStatuses'):
                    d = getattr(o, attr, None)
                    if d is not None and pp in d:
                        getattr(merged, attr)[pp] = d[pp]
    merged.participantNums = np.array(merged.participantNums)
    merged.xs = np.array(merged.xs, dtype=object)
    merged.bics = np.array(merged.bics) if len(merged.bics) > 0 else None
    return merged


def _make_transparent_cmap(name, rgb):
    cdict = {c: [(0, v, v), (1, v, v)] for c, v in zip(['red', 'green', 'blue'], rgb)}
    cdict['alpha'] = [(0, 0.0, 0.0), (1, 1.0, 1.0)]
    return LinearSegmentedColormap(name, cdict)


CMAP_TRAIN = _make_transparent_cmap('purple', (0.7, 0.1, 0.9))
CMAP_GEN = _make_transparent_cmap('hot', (1.0, 0.3, 0.0))
MULTI_ROT_PALETTE = [
    ('rot_teal', (0.0, 0.7, 0.7)),
    ('rot_orange', (1.0, 0.5, 0.0)),
    ('rot_magenta', (0.85, 0.15, 0.55)),
    ('rot_green', (0.2, 0.7, 0.2)),
    ('rot_blue', (0.2, 0.3, 0.9)),
    ('rot_gold', (0.85, 0.75, 0.0)),
]


def _extract_params(xs_i):
    """Extract (scale_S0, nu_S0, params) from xs array.

    Accepts [scale_S0, params...] and [scale_S0, nu_S0, params...].
    Parameter count is inferred from BHT.samplePredictive.
    """
    if _EXPECTED_NPARAMS is not None:
        expected_total = _EXPECTED_NPARAMS + 2  # scale_S0 + nu_S0 + params
        if len(xs_i) == expected_total:
            # With fitted baseline degrees of freedom: [scale_S0, nu_S0, params...]
            scale_S0 = float(xs_i[0])
            nu_S0 = float(xs_i[1])
            raw_params = np.array(xs_i[2:], dtype=np.float64)
            return scale_S0, nu_S0, raw_params
        elif len(xs_i) == expected_total - 1:
            # Without baseline degrees of freedom: [scale_S0, params...] (no nu_S0)
            scale_S0 = float(xs_i[0])
            nu_S0 = 10.0  # default baseline degrees of freedom
            raw_params = np.array(xs_i[1:], dtype=np.float64)
            if len(raw_params) != _EXPECTED_NPARAMS:
                raw_params = np.array(xs_i[-_EXPECTED_NPARAMS:], dtype=np.float64)
            return scale_S0, nu_S0, raw_params
        else:
            raw_params = np.array(xs_i[-_EXPECTED_NPARAMS:], dtype=np.float64)
            remaining = xs_i[: len(xs_i) - _EXPECTED_NPARAMS]
            if len(remaining) >= 2:
                scale_S0 = float(remaining[0])
                nu_S0 = float(remaining[1])
            elif len(remaining) == 1:
                scale_S0 = float(remaining[0])
                nu_S0 = 10.0
            else:
                scale_S0 = 1.0
                nu_S0 = 10.0
            return scale_S0, nu_S0, raw_params
    else:
        # Infer baseline fields from the array length.
        scale_S0 = float(xs_i[0])
        nu_S0 = float(xs_i[1]) if len(xs_i) > 8 else 10.0
        start = 2 if len(xs_i) > 8 else 1
        raw_params = np.array(xs_i[start:], dtype=np.float64)
        return scale_S0, nu_S0, raw_params


def reconstruct_participant_data(bht, pIdx, exclude_washout=False):
    _apply_fit_model_config(bht)
    pp = bht.participantNums[pIdx]
    pDat_full = bht.df[bht.df['participantNum'] == pp].copy()
    phases_full = BHT.derivePhase(pDat_full)
    fitWashout = getattr(bht, 'fitWashout', True)
    include_washout = fitWashout and (not exclude_washout)
    nwM = np.ones(len(phases_full), dtype=bool) if include_washout else (phases_full != 'washout')
    pDat = pDat_full[nwM].reset_index(drop=True)
    phases = phases_full[nwM]
    allAims = pDat['aim'].values
    hasFeedback = BHT.buildHasFeedback(pDat)
    trials = np.arange(len(pDat))
    compMags = pDat['rotation'].values
    pDat['targetPosition'] = pDat['targetPosition'].fillna(0)
    targets = pDat['targetPosition'].values
    isRotation = np.array([phases[t].lower() == 'rotation' for t in trials], dtype=bool)
    scale_S0, nu_S0, params = _extract_params(bht.xs[pIdx])
    destTargets = bht.destTargets[pp]
    destTargArr = np.array([x for x in destTargets if not np.isnan(x)], dtype=np.float64)
    nDest = len(destTargArr)
    genTargets = getattr(bht, 'genTargets', {}).get(pp, set())
    trialStatus = BHT.buildTrialStatus(pDat, genTargets) if genTargets else ['normal'] * len(trials)
    return (
        params,
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        scale_S0,
        nu_S0,
        destTargArr,
        nDest,
        trialStatus,
        phases,
        allAims,
    )


def reconstruct_shared_participant_data(obj, pIdx, exclude_washout=False):
    _apply_fit_model_config(obj)
    pp = obj.participantNums[pIdx]
    pDat_full = obj.df[obj.df['participantNum'] == pp].copy()
    phases_full = BHT.derivePhase(pDat_full)
    fitWashout = getattr(obj, 'fitWashout', True)
    include_washout = fitWashout and (not exclude_washout)
    nwM = np.ones(len(phases_full), dtype=bool) if include_washout else (phases_full != 'washout')
    pDat = pDat_full[nwM].reset_index(drop=True)
    phases = phases_full[nwM]
    allAims = pDat['aim'].values
    hasFeedback = BHT.buildHasFeedback(pDat)
    trials = np.arange(len(pDat))
    compMags = pDat['rotation'].values
    pDat['targetPosition'] = pDat['targetPosition'].fillna(0)
    targets = pDat['targetPosition'].values
    isRotation = np.array([phases[t].lower() == 'rotation' for t in trials], dtype=bool)
    scale_S0, nu_S0, params = _extract_params(obj.xs[pIdx])
    destTargets = obj.destTargets[pp]
    destTargArr = np.array([x for x in destTargets if not np.isnan(x)], dtype=np.float64)
    nDest = len(destTargArr)
    trialStatus = ['normal'] * len(trials)
    return (
        params,
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        scale_S0,
        nu_S0,
        destTargArr,
        nDest,
        trialStatus,
        phases,
        allAims,
    )


def _compute_participant_heatmap(
    recon_fn,
    obj,
    pIdx,
    num_samples,
    abs_rot,
    flip_sign,
    rot_groups=None,
    include_implicit=False,
    per_pp_flip=False,
    exclude_washout=False,
):
    (
        params,
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        scale_S0,
        nu_S0,
        destTargArr,
        nDest,
        trialStatus,
        phases,
        allAims,
    ) = recon_fn(obj, pIdx)
    if per_pp_flip:
        flip_sign = _pp_flip_sign(compMags, phases)
    numTrials = len(trials)
    samples = BHT.samplePredictive(
        params,
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        scale_S0,
        nu_S0,
        destTargArr,
        nDest,
        num_samples,
        rngSeed=42,
    )
    baseline_trials = int(np.sum(phases == 'baseline'))
    washout_trials = int(np.sum(phases == 'washout'))
    shift = baseline_trials
    samples_plot = -samples if flip_sign else samples.copy()
    human_signed = np.array(
        [BHT.wrapAngle(float(a)) if not np.isnan(a) else np.nan for a in allAims]
    )
    if flip_sign:
        human_signed = np.array(
            [BHT.wrapAngle(-a) if not np.isnan(a) else np.nan for a in human_signed]
        )
    genMask = np.array([s == 'generalisation' for s in trialStatus], dtype=bool)
    bin_width = 4.0  # fixed bin width (degrees) for consistent heatmaps across datasets
    num_angle_bins = int(360 / bin_width) + 1
    angle_bin_edges = np.linspace(-180 - bin_width / 2, 180 + bin_width / 2, num_angle_bins + 1)
    if rot_groups is not None:
        sign_mode = rot_groups.get('__sign_mode__', False)
        group_labels = sorted(set(v for k, v in rot_groups.items() if k != '__sign_mode__'))
        count_by_group = {g: np.zeros((numTrials, num_angle_bins)) for g in group_labels}
        for t in range(numTrials):
            bin_idx = np.clip(
                np.digitize(samples_plot[t], angle_bin_edges) - 1, 0, num_angle_bins - 1
            )
            c = np.bincount(bin_idx, minlength=num_angle_bins).astype(float)
            rv = compMags[t]
            if sign_mode:
                g = 'Positive' if rv > 0.5 else ('Negative' if rv < -0.5 else 'baseline')
            else:
                g = rot_groups.get(rv, rot_groups.get(round(float(rv), 1), 'baseline'))
            if g in count_by_group:
                count_by_group[g][t] = c
            elif 'baseline' in count_by_group:
                count_by_group['baseline'][t] = c
    else:
        count_by_group = {}
        count_train = np.zeros((numTrials, num_angle_bins))
        count_gen = np.zeros((numTrials, num_angle_bins))
        for t in range(numTrials):
            bin_idx = np.clip(
                np.digitize(samples_plot[t], angle_bin_edges) - 1, 0, num_angle_bins - 1
            )
            c = np.bincount(bin_idx, minlength=num_angle_bins).astype(float)
            if genMask[t]:
                count_gen[t] = c
            else:
                count_train[t] = c
        count_by_group['train'] = count_train
        if np.any(genMask):
            count_by_group['gen'] = count_gen
    pp_nonzero = compMags[compMags != 0]
    pp_unique_rots = sorted(np.unique(np.round(pp_nonzero, 1))) if len(pp_nonzero) > 0 else []
    implicit_vals = None
    probe_trials = None
    probe_hand = None
    if include_implicit:
        pp = obj.participantNums[pIdx]
        pDat_impl = obj.df[obj.df['participantNum'] == pp].copy()
        if 'implicit' in pDat_impl.columns:
            phases_impl = BHT.derivePhase(pDat_impl)
            fitWashout_impl = getattr(obj, 'fitWashout', True)
            include_washout_impl = fitWashout_impl and (not exclude_washout)
            nwM_impl = (
                np.ones(len(phases_impl), dtype=bool)
                if include_washout_impl
                else (phases_impl != 'washout')
            )
            pDat_filt = pDat_impl[nwM_impl].reset_index(drop=True)
            implicit_raw = pDat_filt['implicit'].values.copy()
            if flip_sign:
                implicit_raw = -implicit_raw
            implicit_vals = implicit_raw
            if 'hand_theta' in pDat_filt.columns:
                fbi_col = (
                    pDat_filt['fbi'].values
                    if 'fbi' in pDat_filt.columns
                    else np.zeros(len(pDat_filt))
                )
                if 'phase' in pDat_filt.columns:
                    washout_mask = np.array(
                        [str(p).lower() == 'washout' for p in pDat_filt['phase'].values]
                    )
                else:
                    phases_filt = BHT.derivePhase(pDat_filt)
                    washout_mask = np.array([p.lower() == 'washout' for p in phases_filt])
                implicit_measure_mask = (fbi_col == 0) | washout_mask
                probe_trials = np.where(implicit_measure_mask)[0]
                probe_hand_raw = pDat_filt['hand_theta'].values[implicit_measure_mask].copy()
                if flip_sign:
                    probe_hand_raw = -probe_hand_raw
                probe_hand = probe_hand_raw
    return {
        'count_by_group': count_by_group,
        'trials': trials,
        'shift': shift,
        'washout_trials': washout_trials,
        'abs_rot': abs_rot,
        'flip_sign': flip_sign,
        'human_signed': human_signed,
        'genMask': genMask,
        'num_angle_bins': num_angle_bins,
        'numTrials': numTrials,
        'compMags': compMags,
        'pp_unique_rots': pp_unique_rots,
        'implicit': implicit_vals,
        'probe_trials': probe_trials,
        'probe_hand': probe_hand,
    }


def _detect_rotation_properties(obj):
    """Scan up to 3 participants. Use MAX abs rotation for scale reference."""
    all_nonzero = []
    for pp in obj.participantNums[:3]:
        pDat = obj.df[obj.df['participantNum'] == pp]
        rots = pDat['rotation'].values
        all_nonzero.append(rots[rots != 0])
    nonzero_rots = np.concatenate(all_nonzero) if all_nonzero else np.array([])
    rotation_label = getattr(obj, 'conVal', '')
    has_both_signs = len(nonzero_rots) > 0 and nonzero_rots.min() < 0 and nonzero_rots.max() > 0
    if len(nonzero_rots) > 0:
        abs_rot_ref = float(np.max(np.abs(nonzero_rots)))
    elif isinstance(rotation_label, (int, float)):
        abs_rot_ref = abs(rotation_label)
    else:
        abs_rot_ref = 30
    abs_rot_ref = max(abs_rot_ref, 5)
    flip_sign = False
    if not has_both_signs and len(nonzero_rots) > 0:
        flip_sign = np.sign(nonzero_rots[0]) > 0
    uniq_rounded = sorted(np.unique(np.round(nonzero_rots, 1))) if len(nonzero_rots) > 0 else []
    return abs_rot_ref, flip_sign, has_both_signs, uniq_rounded


def _detect_block_boundaries(obj, exclude_washout=False):
    pp = obj.participantNums[0]
    pDat = obj.df[obj.df['participantNum'] == pp].copy()
    phases = BHT.derivePhase(pDat)
    fitWashout = getattr(obj, 'fitWashout', True)
    include_washout = fitWashout and (not exclude_washout)
    nwM = np.ones(len(phases), dtype=bool) if include_washout else (phases != 'washout')
    pDat_f = pDat[nwM].reset_index(drop=True)
    if 'stage' not in pDat_f.columns:
        return []
    stages = pDat_f['stage'].values
    baseline_n = int(np.sum(phases[nwM] == 'baseline'))
    washout_n = int(np.sum(phases[nwM] == 'washout'))
    total_n = len(pDat_f)
    final_washout_start = total_n - washout_n if washout_n > 0 else total_n
    boundaries = []
    for i in range(1, total_n):
        if stages[i] != stages[i - 1]:
            if i <= baseline_n or i >= final_washout_start:
                continue
            boundaries.append(i)
    return boundaries


def generate_bht_plots(
    fits,
    save_dir,
    max_participants=600,
    num_samples=500,
    debug=False,
    multi_rotation_colors=False,
    block_boundary_lines=False,
    include_implicit=False,
    per_participant_flip=False,
    ding_aggregate=False,
    exclude_washout=False,
):
    os.makedirs(save_dir, exist_ok=True)
    obj_list = extract_fit_shells(fits)
    if ding_aggregate:
        _ding_group_lookup = {}
        for _o in obj_list:
            _dn = getattr(_o, 'datasetName', '')
            _ding_group_lookup[_dn] = _group_condition_key(_dn)
        if debug:
            for gk in sorted(set(_ding_group_lookup.values())):
                members = [k for k, v in _ding_group_lookup.items() if v == gk]
                print(f"Ding group '{gk}': {members}")
    else:
        _ding_group_lookup = None
    xs0 = obj_list[0].xs[0]
    if debug:
        print(
            f"xs[0] has {len(xs0)} elements (expect {_EXPECTED_NPARAMS + 2 if _EXPECTED_NPARAMS else '?'})"
        )
    rot_group_map = None
    group_cmaps = {}
    if multi_rotation_colors:
        all_uniq_rots = set()
        for obj in obj_list:
            _, _, _, uniq_rounded = _detect_rotation_properties(obj)
            all_uniq_rots.update(uniq_rounded)
        all_uniq_rots.discard(0.0)
        all_uniq_rots = sorted(all_uniq_rots)
        if 0 < len(all_uniq_rots) <= len(MULTI_ROT_PALETTE):
            rot_group_map = {0.0: 'baseline'}
            for rv in all_uniq_rots:
                rot_group_map[rv] = f'{rv:.0f}°'
            group_labels_sorted = sorted(
                [g for g in set(rot_group_map.values()) if g != 'baseline']
            )
            for idx_g, g in enumerate(group_labels_sorted):
                name, rgb = MULTI_ROT_PALETTE[idx_g % len(MULTI_ROT_PALETTE)]
                group_cmaps[g] = _make_transparent_cmap(name, rgb)
            group_cmaps['baseline'] = _make_transparent_cmap('baseline_grey', (0.5, 0.5, 0.5))
            if debug:
                print(f"Multi-rotation global pool: {group_labels_sorted}")
        elif len(all_uniq_rots) > len(MULTI_ROT_PALETTE):
            rot_group_map = {'__sign_mode__': True, 0.0: 'baseline'}
            group_cmaps = {
                'Positive': _make_transparent_cmap('rot_pos', (1.0, 0.5, 0.0)),
                'Negative': _make_transparent_cmap('rot_neg', (0.0, 0.7, 0.7)),
                'baseline': _make_transparent_cmap('baseline_grey', (0.5, 0.5, 0.5)),
            }
            if debug:
                print(f"Sign-based grouping (too many unique: {len(all_uniq_rots)})")
    dataset_block_bounds = {}
    if block_boundary_lines:
        for obj in obj_list:
            dataset_block_bounds[id(obj)] = _detect_block_boundaries(
                obj, exclude_washout=exclude_washout
            )
            if debug:
                bn = getattr(obj, 'datasetName', 'dataset')
                print(f"Block boundaries for {bn}: {dataset_block_bounds[id(obj)]}")
    all_precomputed = []
    global_95_counts = []
    for obj in obj_list:
        recon_fn = (
            (
                lambda _obj, _pIdx: reconstruct_shared_participant_data(
                    _obj, _pIdx, exclude_washout=exclude_washout
                )
            )
            if _is_shared_fit(obj)
            else (
                lambda _obj, _pIdx: reconstruct_participant_data(
                    _obj, _pIdx, exclude_washout=exclude_washout
                )
            )
        )
        datasetName = getattr(obj, 'datasetName', 'dataset')
        rotation_label = getattr(obj, 'conVal', '')
        N = len(obj.participantNums)
        n_to_plot = min(max_participants, N)
        abs_rot_ref, flip_sign, _, _ = _detect_rotation_properties(obj)
        if rot_group_map is not None:
            for pp in obj.participantNums[:5]:
                pDat = obj.df[obj.df['participantNum'] == pp]
                for rv_exact in np.unique(pDat['rotation'].values):
                    if rv_exact == 0:
                        continue
                    rv_round = round(float(rv_exact), 1)
                    if rv_round in rot_group_map:
                        rot_group_map[float(rv_exact)] = rot_group_map[rv_round]
        if debug:
            print(
                f"Pass 1 — {datasetName} rot={rotation_label}, N={N}, "
                f"abs_rot_ref={abs_rot_ref:.1f}, flip={flip_sign}, "
                f"multi_rot={'yes' if rot_group_map else 'no'}"
            )
        for pIdx in range(n_to_plot):
            pp = obj.participantNums[pIdx]
            data = _compute_participant_heatmap(
                recon_fn,
                obj,
                pIdx,
                num_samples,
                abs_rot_ref,
                flip_sign,
                rot_groups=rot_group_map,
                include_implicit=include_implicit,
                per_pp_flip=per_participant_flip,
                exclude_washout=exclude_washout,
            )
            for mat in data['count_by_group'].values():
                nz = mat[mat > 0]
                if len(nz) > 0:
                    global_95_counts.append(np.percentile(nz, 95))
            all_precomputed.append((obj, pIdx, pp, data, datasetName, rotation_label))
        if debug:
            print(f"  Sampled {n_to_plot} participants.")
    gv = np.percentile(global_95_counts, 95) if global_95_counts else 400
    gv = max(gv, 200)
    gv_prop = gv / num_samples
    SHARED_NORM = PowerNorm(gamma=0.5, vmin=0, vmax=gv_prop / 3)
    if debug:
        print(f"\nGlobal vmax counts={gv:.0f}, prop={gv_prop:.4f}, norm_vmax={gv_prop / 3:.4f}")
    if rot_group_map is None:
        group_cmaps = {'train': CMAP_TRAIN, 'gen': CMAP_GEN}
    for obj, pIdx, pp, data, datasetName, rotation_label in all_precomputed:
        numTrials = data['numTrials']
        trials = data['trials']
        shift = data['shift']
        washout_trials = data['washout_trials']
        abs_rot = data['abs_rot']
        flip_sign = data['flip_sign']
        human_signed = data['human_signed']
        genMask = data['genMask']
        trainMask = ~genMask
        has_gen = np.any(genMask)
        compMags = data['compMags']
        pp_unique_rots = data['pp_unique_rots']
        trials_shifted = trials - shift
        x_lo = trials_shifted[0] - 0.5
        x_hi = trials_shifted[-1] + 0.5
        extent = [x_lo, x_hi, -180, 180]
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
        ax.set_title(f'{datasetName} | Rot {rotation_label}° | Participant {pp}', fontsize=14)
        ax.set_facecolor('white')
        zord = 2
        for grp, ct in data['count_by_group'].items():
            prop = ct / num_samples
            prop_m = np.ma.masked_where(ct == 0, prop)
            cmap = group_cmaps.get(grp, CMAP_TRAIN)
            ax.imshow(
                prop_m.T,
                extent=extent,
                origin='lower',
                aspect='auto',
                cmap=cmap,
                norm=SHARED_NORM,
                interpolation='nearest',
                zorder=zord,
            )
            zord += 1
        sign_mode = rot_group_map is not None and rot_group_map.get('__sign_mode__', False)
        if rot_group_map is None:
            comp_line = abs_rot if flip_sign else -abs_rot
            over_line = -abs_rot if flip_sign else abs_rot
            ax.axhline(y=comp_line, color='grey', ls='--', lw=2, alpha=0.6, zorder=1)
            ax.axhline(y=over_line, color='grey', ls='--', lw=2, alpha=0.6, zorder=1)
        elif not sign_mode:
            pp_rot_labels = set()
            for rv in pp_unique_rots:
                label = rot_group_map.get(rv, rot_group_map.get(round(float(rv), 1), None))
                if label and label != 'baseline':
                    pp_rot_labels.add(label)
            for rv_str in sorted(pp_rot_labels):
                cmap = group_cmaps.get(rv_str, CMAP_TRAIN)
                rv = float(rv_str.replace('°', ''))
                if flip_sign:
                    rv = -rv
                ax.axhline(y=-rv, color=cmap(0.9), ls='--', lw=2, alpha=0.7, zorder=1)
        ax.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.6, zorder=1)
        ax.axvline(x=0, color='grey', ls='--', lw=2, alpha=0.6, zorder=1)
        if washout_trials > 0:
            ax.axvline(
                x=(numTrials - washout_trials) - shift,
                color='grey',
                ls='--',
                lw=2,
                alpha=0.6,
                zorder=1,
            )
        for bnd_trial in dataset_block_bounds.get(id(obj), []):
            ax.axvline(x=bnd_trial - shift, color='grey', ls='--', lw=2, alpha=0.6, zorder=1)
        valid = ~np.isnan(human_signed)
        if rot_group_map is not None:
            for grp, cmap in group_cmaps.items():
                if grp == 'baseline':
                    grp_mask = compMags == 0
                elif sign_mode:
                    if grp == 'Positive':
                        grp_mask = compMags > 0.5
                    elif grp == 'Negative':
                        grp_mask = compMags < -0.5
                    else:
                        continue
                else:
                    rv = float(grp.replace('°', ''))
                    grp_mask = np.isclose(compMags, rv, atol=0.5)
                sel = valid & grp_mask
                if np.any(sel):
                    ax.scatter(
                        trials_shifted[sel],
                        human_signed[sel],
                        color=cmap(0.95),
                        s=60,
                        alpha=0.5,
                        zorder=8,
                        edgecolor='k',
                        linewidth=3,
                    )
        else:
            ax.scatter(
                trials_shifted[valid & trainMask],
                human_signed[valid & trainMask],
                color='blue',
                s=60,
                alpha=0.5,
                zorder=8,
                edgecolor='darkblue',
                linewidth=3,
            )
            if has_gen and np.any(valid & genMask):
                ax.scatter(
                    trials_shifted[valid & genMask],
                    human_signed[valid & genMask],
                    color='#00FF00',
                    s=60,
                    alpha=0.5,
                    zorder=8,
                    edgecolor='darkgreen',
                    linewidth=3,
                )
        if data.get('implicit') is not None:
            imp_vals = data['implicit']
            ax.plot(
                trials_shifted,
                imp_vals,
                color='darkred',
                lw=5.0,
                alpha=0.7,
                zorder=11,
                label='Implicit (SSM)',
            )
        if data.get('probe_trials') is not None and data.get('probe_hand') is not None:
            pt = data['probe_trials']
            ph = data['probe_hand']
            ax.scatter(
                pt - shift,
                ph,
                color='red',
                s=60,
                alpha=0.5,
                zorder=10,
                edgecolor='darkred',
                linewidth=1.5,
                marker='o',
                label='Human implicit',
            )
        ax.set_ylim(-181, 181)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Aim Angle (°)')
        yticks = np.arange(-180, 181, 15)
        ax.set_yticks(yticks)
        if rot_group_map is not None:
            label_step = 45.0
        else:
            base_rot = float(abs_rot) if abs_rot not in (None, 0) else 15.0
            step_mult = max(1, int(np.ceil(45.0 / base_rot)))
            label_step = base_rot * step_mult
        yticklabels = []
        for t in yticks:
            if t == 0:
                yticklabels.append('0°')
            elif label_step > 0 and np.isclose(t / label_step, np.round(t / label_step), atol=1e-8):
                yticklabels.append(f'{int(t)}°')
            else:
                yticklabels.append('')
        ax.set_yticklabels(yticklabels)
        sns.despine()
        handles = []
        if rot_group_map is not None:
            if sign_mode:
                legend_groups = [g for g in group_cmaps if g != 'baseline']
            else:
                pp_rot_labels = set()
                for rv in pp_unique_rots:
                    label = rot_group_map.get(rv, rot_group_map.get(round(float(rv), 1), None))
                    if label and label != 'baseline':
                        pp_rot_labels.add(label)
                legend_groups = sorted(pp_rot_labels)
            for grp in sorted(legend_groups):
                c = group_cmaps[grp]
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='s',
                        color='w',
                        markerfacecolor=c(0.8),
                        markersize=12,
                        label=f'Model {grp}',
                    )
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='o',
                        color=c(0.95),
                        markersize=8,
                        linestyle='None',
                        markeredgecolor='k',
                        label=f'Human {grp}',
                    )
                )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='s',
                    color='w',
                    markerfacecolor=CMAP_TRAIN(0.8),
                    markersize=12,
                    label='Model (trained)',
                )
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='x',
                    color='blue',
                    markersize=8,
                    linestyle='None',
                    markeredgewidth=2,
                    label='Human (trained)',
                )
            )
            if has_gen:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='s',
                        color='w',
                        markerfacecolor=CMAP_GEN(0.8),
                        markersize=12,
                        label='Model (generalisation)',
                    )
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='x',
                        color='#00FF00',
                        markersize=8,
                        linestyle='None',
                        markeredgewidth=2,
                        label='Human (generalisation)',
                    )
                )
        if data.get('implicit') is not None:
            handles.append(Line2D([0], [0], color='#22AA22', lw=2.5, label='Implicit (SSM)'))
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor='#22AA22',
                    markersize=8,
                    markeredgecolor='darkgreen',
                    linestyle='None',
                    label='Human implicit',
                )
            )
        ax.legend(handles=handles, loc='upper left')
        if _ding_group_lookup is not None:
            _gfolder = _ding_group_lookup.get(datasetName, datasetName)
            _group_dir = os.path.join(save_dir, _gfolder)
            os.makedirs(_group_dir, exist_ok=True)
            save_path = os.path.join(_group_dir, f"{datasetName}_rot{rotation_label}_p{pp}.png")
        else:
            save_path = os.path.join(save_dir, f"{datasetName}_rot{rotation_label}_p{pp}.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except OSError:
            alt = save_path.replace('.png', '_v2.png')
            print(f'  File locked, saving to {alt}')
            plt.savefig(alt, dpi=300, bbox_inches='tight')
        plt.close(fig)
    cb_groups = {k: v for k, v in group_cmaps.items() if k != 'baseline'}
    n_cb = len(cb_groups)
    fig_cb, axes_cb = plt.subplots(1, n_cb, figsize=(0.6 * n_cb, 4))
    if n_cb == 1:
        axes_cb = [axes_cb]
    for ax_cb, (grp, cmap) in zip(axes_cb, cb_groups.items()):
        sm = cm.ScalarMappable(cmap=cmap, norm=SHARED_NORM)
        sm.set_array([])
        cb = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
        ax_cb.set_title(grp, fontsize=8, pad=4)
        cb.set_label('P(sample)', fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout(pad=0.5)
    cb_path = os.path.join(save_dir, 'colorbar_standalone.svg')
    fig_cb.savefig(cb_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cb)
    print(f"Done — {len(all_precomputed)} plots saved to {save_dir}/")
    print(f"Colorbar: {cb_path}")


def _pp_flip_sign(compMags, phases):
    rot_mask = np.array([str(p).lower() == 'rotation' for p in phases])
    rot_vals = compMags[rot_mask]
    nonzero = rot_vals[np.abs(rot_vals) > 0.5]
    if len(nonzero) == 0:
        return False
    return float(nonzero[0]) > 0


def to_signed(ang):
    return np.where(ang > 180, ang - 360, ang)


def generate_bht_phase_plots(
    fits,
    save_dir,
    n_bins=150,
    num_samples=2000,
    debug=False,
    savings_mode=False,
    per_participant_flip=False,
    ding_aggregate=False,
    ylimMax=0.025,
    hmm_fits=None,
):
    os.makedirs(save_dir, exist_ok=True)
    hmm_lookup = {}

    def _full_cond_key(_obj):
        _name = getattr(_obj, 'datasetName', '')
        for _pref in ('AllShared_', 'EqMeans_', 'Ding_', 'HMMAllShared_'):
            if _name.startswith(_pref):
                _name = _name[len(_pref) :]
        return _name

    if hmm_fits is not None:
        if isinstance(hmm_fits, dict):
            hmm_pairs = list(hmm_fits.items())
        else:
            hmm_pairs = [(_full_cond_key(_hobj), _hobj) for _hobj in extract_fit_shells(hmm_fits)]
        for _hkey, _hobj in hmm_pairs:
            if _hobj is None or not hasattr(_hobj, 'participantNums'):
                continue
            for _hi, _pp in enumerate(_hobj.participantNums):
                hmm_lookup[(_hkey, _pp)] = (_hobj, _hi)
                hmm_lookup[_pp] = (_hobj, _hi)
        if debug and len(hmm_lookup) > 0:
            print(
                f"HMM overlay enabled: {sum(1 for k in hmm_lookup if not isinstance(k, tuple))} participants"
            )
    obj_list = extract_fit_shells(fits)
    if ding_aggregate:
        groups = {}
        for obj in obj_list:
            name = getattr(obj, 'datasetName', '')
            # Pool rotation magnitudes: 'EqMeans_Inner_2T_90' becomes 'Inner_2T'.
            parts = name.split('_')
            prefix_parts = [p for p in parts if p not in ('Ding', 'EqMeans')]
            if len(prefix_parts) >= 2:
                group_key = '_'.join(prefix_parts[:-1])
            else:
                group_key = name
            groups.setdefault(group_key, []).append(obj)
        if debug:
            for gk, objs in groups.items():
                print(
                    f"Ding group '{gk}': {len(objs)} sub-conditions, "
                    f"total N={sum(len(o.participantNums) for o in objs)}"
                )
        plot_groups = list(groups.items())
    else:
        plot_groups = [(None, [obj]) for obj in obj_list]
    for group_key, group_objs in plot_groups:
        ref_obj = group_objs[0]
        is_shared = _is_shared_fit(ref_obj)
        abs_rot_ref, dataset_flip, has_both_signs, _ = _detect_rotation_properties(ref_obj)
        rotation_label = getattr(ref_obj, 'conVal', '')
        datasetLabel = group_key if group_key else getattr(ref_obj, 'datasetName', 'dataset')
        pp0 = ref_obj.participantNums[0]
        pDat0 = ref_obj.df[ref_obj.df['participantNum'] == pp0].copy()
        phases0 = BHT.derivePhase(pDat0)
        fitWashout = getattr(ref_obj, 'fitWashout', True)
        nwM0 = np.ones(len(phases0), dtype=bool) if fitWashout else (phases0 != 'washout')
        pDat0_f = pDat0[nwM0].reset_index(drop=True)
        total_trials = len(pDat0_f)
        baseline_n = int(np.sum(phases0[nwM0] == 'baseline'))
        washout_n = int(np.sum(phases0[nwM0] == 'washout'))
        if savings_mode and 'stage' in pDat0_f.columns:
            stages = pDat0_f['stage'].values
            blocks = []
            s3 = np.where(stages == 3)[0]
            if len(s3) > 0:
                blocks.append(('Block1', s3[0], s3[-1] + 1))
            s4 = np.where(stages == 4)[0]
            s5 = np.where(stages == 5)[0]
            wo1_trials = np.concatenate([s4, s5]) if len(s4) > 0 or len(s5) > 0 else np.array([])
            s6 = np.where(stages == 6)[0]
            if len(s6) > 0:
                blocks.append(('Block2', s6[0], s6[-1] + 1))
            s7 = np.where(stages == 7)[0]
            all_phase_ranges = []
            for block_name, rot_start, rot_end in blocks:
                rot_len = rot_end - rot_start
                early_end = rot_start + 24
                mid_start = rot_start + rot_len // 3
                mid_end = mid_start + min(40, rot_len // 3)
                late_start = rot_end - min(40, rot_len // 3)
                all_phase_ranges.extend(
                    [
                        (
                            rot_start,
                            early_end,
                            f'{block_name} First {early_end - rot_start}',
                            'all',
                        ),
                        (mid_start, mid_end, f'{block_name} Middle {mid_end - mid_start}', 'all'),
                        (late_start, rot_end, f'{block_name} Late {rot_end - late_start}', 'all'),
                    ]
                )
                if block_name == 'Block1' and len(wo1_trials) > 0:
                    wo_s, wo_e = (
                        int(wo1_trials[0]),
                        int(wo1_trials[min(6, len(wo1_trials) - 1)]) + 1,
                    )
                    all_phase_ranges.append(
                        (wo_s, wo_e, f'{block_name} Washout {wo_e - wo_s}', 'all')
                    )
                elif block_name == 'Block2' and len(s7) > 0:
                    wo_s, wo_e = int(s7[0]), int(s7[min(6, len(s7) - 1)]) + 1
                    all_phase_ranges.append(
                        (wo_s, wo_e, f'{block_name} Washout {wo_e - wo_s}', 'all')
                    )
        else:
            rot_start = baseline_n
            rot_end = total_trials - washout_n
            rot_len = rot_end - rot_start
            if rot_len < 10:
                if debug:
                    print(f"Skipping {datasetLabel}: rotation too short")
                continue
            early_end = rot_start + 24
            mid_start = rot_start + rot_len // 3
            mid_end = mid_start + min(40, rot_len // 3)
            late_start = rot_end - min(40, rot_len // 3)
            if ding_aggregate:
                n_early = early_end - rot_start
                n_mid = mid_end - mid_start
                n_late = rot_end - late_start
                all_phase_ranges = [
                    (rot_start, early_end, f'Train First {n_early}', 'train'),
                    (mid_start, mid_end, f'Train Middle {n_mid}', 'train'),
                    (late_start, rot_end, f'Train Late {n_late}', 'train'),
                    (rot_start, early_end, f'Gen First {n_early}', 'gen'),
                    (mid_start, mid_end, f'Gen Middle {n_mid}', 'gen'),
                    (late_start, rot_end, f'Gen Late {n_late}', 'gen'),
                ]
            else:
                n_early = early_end - rot_start
                n_mid = mid_end - mid_start
                n_late = rot_end - late_start
                all_phase_ranges = [
                    (rot_start, early_end, f'First {n_early}', 'all'),
                    (mid_start, mid_end, f'Middle {n_mid}', 'all'),
                    (late_start, rot_end, f'Late {n_late}', 'all'),
                ]
                if washout_n > 0:
                    wo_s = rot_end + 1
                    wo_e = wo_s + min(7, washout_n - 1)
                    all_phase_ranges.append((wo_s, wo_e, f'Washout {wo_e - wo_s}', 'all'))
        numPanels = len(all_phase_ranges)
        bin_edges = np.linspace(-180, 180, n_bins + 1)
        human_per_phase = [[] for _ in range(numPanels)]
        model_per_phase = [[] for _ in range(numPanels)]
        hmm_marginals_per_phase = [[] for _ in range(numPanels)]
        total_pp = 0
        for obj in group_objs:
            recon_fn = (
                reconstruct_shared_participant_data
                if _is_shared_fit(obj)
                else reconstruct_participant_data
            )
            N = len(obj.participantNums)
            total_pp += N
            for pIdx in range(N):
                try:
                    (
                        params,
                        hasFeedback,
                        trials,
                        isRotation,
                        compMags,
                        targets,
                        scale_S0,
                        nu_S0,
                        destTargArr,
                        nDest,
                        trialStatus,
                        pp_phases,
                        allAims,
                    ) = recon_fn(obj, pIdx)
                except Exception as e:
                    if debug:
                        print(f"  Skip pp {pIdx}: {e}")
                    continue
                if per_participant_flip:
                    flip = _pp_flip_sign(compMags, pp_phases)
                else:
                    flip = dataset_flip
                numTrials = len(trials)
                samples = BHT.samplePredictive(
                    params,
                    hasFeedback,
                    trials,
                    isRotation,
                    compMags,
                    targets,
                    scale_S0,
                    nu_S0,
                    destTargArr,
                    nDest,
                    num_samples,
                    rngSeed=42 + pIdx + total_pp,
                )
                if flip:
                    samples = -samples
                    aims_signed = np.array(
                        [BHT.wrapAngle(-float(a)) if not np.isnan(a) else np.nan for a in allAims]
                    )
                else:
                    aims_signed = np.array(
                        [BHT.wrapAngle(float(a)) if not np.isnan(a) else np.nan for a in allAims]
                    )
                genMask = np.array([s == 'generalisation' for s in trialStatus], dtype=bool)
                for pi, (start_idx, end_idx, _, target_filter) in enumerate(all_phase_ranges):
                    for t in range(start_idx, min(end_idx, numTrials)):
                        h = aims_signed[t]
                        if np.isnan(h):
                            continue
                        is_gen = genMask[t] if t < len(genMask) else False
                        if target_filter == 'train' and is_gen:
                            continue
                        if target_filter == 'gen' and not is_gen:
                            continue
                        human_per_phase[pi].append(h)
                        model_per_phase[pi].extend(samples[t, :])
                if len(hmm_lookup) > 0:
                    pp_num = obj.participantNums[pIdx]
                    hmm_match = hmm_lookup.get((_full_cond_key(obj), pp_num), None)
                    if hmm_match is None:
                        hmm_match = hmm_lookup.get(pp_num, None)
                    if hmm_match is not None:
                        hmm_obj, hmm_idx = hmm_match
                        fine_angles = np.arange(-180, 181)  # 361 points
                        orig_idx_map = ((180 + fine_angles) % 360).astype(int)
                        hmm_sigma = (
                            float(hmm_obj.xs[hmm_idx][0])
                            if not np.isnan(hmm_obj.xs[hmm_idx][0])
                            else 0.0
                        )
                        hmm_scale0 = (
                            float(hmm_obj.baselineScale[hmm_idx])
                            if hasattr(hmm_obj, 'baselineScale')
                            else hmm_sigma
                        )
                        hmm_nu0 = (
                            float(hmm_obj.baselineNu[hmm_idx])
                            if hasattr(hmm_obj, 'baselineNu')
                            else 10.0
                        )
                        from scipy.stats import t as _student_t_dist
                        from scipy import ndimage as _ndimage

                        def _student_t_kernel(_scale, _nu):
                            if (
                                (not np.isfinite(_scale))
                                or _scale <= 0
                                or (not np.isfinite(_nu))
                                or _nu <= 0
                            ):
                                _k = np.zeros(len(fine_angles), dtype=float)
                                _k[180] = 1.0
                                return _k
                            _k = _student_t_dist.pdf(fine_angles / _scale, df=_nu) / _scale
                            _k = np.maximum(_k, 0)
                            _s = _k.sum()
                            return _k / _s if _s > 0 else _k

                        hmm_kernel0 = _student_t_kernel(hmm_scale0, hmm_nu0)
                        hmm_kernel1 = _student_t_kernel(hmm_sigma, 5.0)
                        conv0 = hmm_kernel0
                        hmm_policies = hmm_obj.model_predictive_policies[hmm_idx]
                        hmm_pi_preds = hmm_obj.pi_preds[hmm_idx]
                        n_hmm_trials = len(hmm_policies)
                        for pi, (start_idx, end_idx, _, target_filter) in enumerate(
                            all_phase_ranges
                        ):
                            for t in range(start_idx, min(end_idx, numTrials, n_hmm_trials)):
                                h = aims_signed[t]
                                if np.isnan(h):
                                    continue
                                is_gen = genMask[t] if t < len(genMask) else False
                                if target_filter == 'train' and is_gen:
                                    continue
                                if target_filter == 'gen' and not is_gen:
                                    continue
                                pi_s0 = hmm_pi_preds[t][0]
                                policy_recentered = np.array(hmm_policies[t])[orig_idx_map]
                                conv1 = _ndimage.convolve1d(
                                    policy_recentered, hmm_kernel1, mode='constant', cval=0.0
                                )
                                conv1 = np.maximum(conv1, 0)
                                cs1 = conv1.sum()
                                conv1 = conv1 / cs1 if cs1 > 0 else conv1
                                marginal = pi_s0 * conv0 + (1 - pi_s0) * conv1
                                if flip:
                                    marginal = marginal[::-1]
                                hmm_marginals_per_phase[pi].append(marginal)
        if debug:
            print(f"\n=== {datasetLabel} rot={rotation_label} | N={total_pp} ===")
            for pi, (s, e, lbl, filt) in enumerate(all_phase_ranges):
                print(f"  {lbl}: trials {s}-{e}, n_human={len(human_per_phase[pi])}")
        fig, axs = plt.subplots(1, numPanels, figsize=(5 * numPanels, 5), sharey=True)
        fig.suptitle(
            f'{datasetLabel} | Rot {rotation_label}° | Phase Distributions (N={total_pp})',
            fontsize=14,
        )
        if numPanels == 1:
            axs = [axs]
        for ax in axs if isinstance(axs, (list, np.ndarray)) else [axs]:
            ax.tick_params(labelleft=True)
        for pi, (start_idx, end_idx, title_str, _) in enumerate(all_phase_ranges):
            ax = axs[pi]
            human_arr = np.array(human_per_phase[pi])
            model_arr = np.array(model_per_phase[pi])
            if len(human_arr) > 0:
                hist_human, _ = np.histogram(human_arr, bins=bin_edges, density=True)
                ax.stairs(
                    hist_human,
                    bin_edges,
                    fill=True,
                    color='blue',
                    label='Human',
                    alpha=0.4,
                    linewidth=0,
                )
            if len(model_arr) > 0:
                hist_model, _ = np.histogram(model_arr, bins=bin_edges, density=True)
                ax.stairs(
                    hist_model,
                    bin_edges,
                    baseline=None,
                    fill=False,
                    color='darkorange',
                    label='BHT',
                    alpha=0.85,
                    linewidth=4.5,
                    zorder=11,
                )
            hmm_margs = hmm_marginals_per_phase[pi]
            if len(hmm_margs) > 0:
                fine_angles_plot = np.arange(-180, 181)
                avg_hmm = np.mean(hmm_margs, axis=0)
                hist_hmm, _ = np.histogram(
                    fine_angles_plot, bins=bin_edges, weights=avg_hmm, density=True
                )
                ax.stairs(
                    hist_hmm,
                    bin_edges,
                    baseline=None,
                    fill=False,
                    color='green',
                    label='HMM',
                    alpha=0.4,
                    linewidth=4.5,
                    zorder=10,
                )
            ax.set_title(title_str, fontsize=11)
            ax.set_xlabel('Signed Angle (°)')
            if pi == 0:
                ax.set_ylabel('Density')
            ax.axvline(0, color='grey', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(abs_rot_ref, color='grey', linestyle='--', linewidth=2.2, alpha=0.8)
            ax.axvline(-abs_rot_ref, color='grey', linestyle='--', linewidth=2.2, alpha=0.8)
            ax.set_xlim(-180, 180)
            ax.set_ylim(0, ylimMax)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{datasetLabel}_rot{rotation_label}_phase_dists.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        if debug:
            print(f"  Saved: {save_path}")
    print(f"Phase distribution plots saved to {save_dir}")


def _get_model_expected_aim(bht, pp):
    from scipy.special import logsumexp as sp_logsumexp

    ps = bht.predState[pp]
    ncomp = ps['predNComp'].astype(int)
    logprob = ps['predCompLogProb']
    means = ps['predCompMean']
    T = len(ncomp)
    expected = np.zeros(T)
    for t in range(T):
        nc = ncomp[t]
        if nc <= 0:
            continue
        lp = logprob[t, :nc].copy()
        m = means[t, :nc]
        ls = sp_logsumexp(lp)
        p = np.exp(lp - ls)
        expected[t] = np.sum(p * m)
    return expected


def _build_savings_plot_df(bht, df):
    rows = []
    for i, pp in enumerate(bht.participantNums):
        pDat = df[df['participantNum'] == pp].sort_values('TN').reset_index(drop=True)
        ccw = pDat['CCW'].iloc[0]
        flip = -1.0 if ccw == 1 else 1.0
        model_aim = _get_model_expected_aim(bht, pp)
        implicit = pDat['implicit'].values
        for t in range(len(pDat)):
            row = pDat.iloc[t]
            tn = int(row['TN'])
            cycle = (tn - 1) // 4
            fbi = int(row['fbi'])
            stage = int(row['stage'])
            phase = row['phase']
            ht = row['hand_theta']
            m_aim = model_aim[t] if t < len(model_aim) else np.nan
            m_hand = (m_aim + implicit[t]) * flip
            m_probe = implicit[t] * flip
            rows.append(
                {
                    'pp': pp,
                    'TN': tn,
                    'cycle': cycle,
                    'stage': stage,
                    'fbi': fbi,
                    'phase': phase,
                    'hand_theta': ht * flip if not np.isnan(ht) else np.nan,
                    'implicit': implicit[t] * flip,
                    'model_aim': m_aim * flip,
                    'model_hand': m_hand,
                    'model_probe': m_probe,
                }
            )
    return pd.DataFrame(rows)


def _cycle_mean_ci(group_df, col, cycle_col='cycle'):
    agg = group_df.groupby([cycle_col, 'pp'])[col].mean().reset_index()
    stats = agg.groupby(cycle_col)[col].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    return stats[cycle_col].values, stats['mean'].values, stats['ci'].values


def plot_savings_panels(bht, df, save_path=None):
    pdf = _build_savings_plot_df(bht, df)

    # Exclude probe trials in stages 3 and 6 from feedback cycles.
    is_lp = (pdf['stage'].isin([3, 6])) & (pdf['fbi'] == 0)
    new_cycle = np.empty(len(pdf), dtype=int)
    for pp in pdf['pp'].unique():
        pp_rows = np.where(pdf['pp'].values == pp)[0]
        counter = 0
        for row in pp_rows:
            new_cycle[row] = counter // 4
            if not is_lp.iloc[row]:
                counter += 1
    pdf['cycle'] = new_cycle

    # Exclude the very first probe trial in Learning 1 (stage 3)
    for pp in pdf['pp'].unique():
        s3_probes = pdf[(pdf['pp'] == pp) & (pdf['stage'] == 3) & (pdf['fbi'] == 0)]
        if len(s3_probes) > 0:
            first_probe_idx = s3_probes.index[0]
            pdf.loc[first_probe_idx, 'hand_theta'] = np.nan

    # Probe cycle: group every 4 consecutive probes within a stage
    pdf['probe_cycle'] = np.nan
    for pp in pdf['pp'].unique():
        for stg in [3, 6]:
            mask = (pdf['pp'] == pp) & (pdf['stage'] == stg) & (pdf['fbi'] == 0)
            idx = pdf.index[mask]
            if len(idx) == 0:
                continue
            fb_cycles = pdf.loc[idx, 'cycle'].values
            n = len(idx)
            for i in range(n):
                group_start = (i // 4) * 4
                group_end = min(group_start + 4, n)
                group_mean = fb_cycles[group_start:group_end].mean()
                pdf.loc[idx[i], 'probe_cycle'] = group_mean

    stage_bounds = {
        1: (0, 10),
        2: (10, 20),
        3: (20, 60),
        4: (60, 70),
        5: (70, 110),
        6: (110, 150),
        7: (150, 160),
    }
    labels = {
        1: 'No FB\nBase',
        2: 'FB\nBase',
        3: 'Learning 1',
        4: 'No FB\nAftereffect 1',
        5: 'FB Washout',
        6: 'Learning 2',
        7: 'No FB\nAftereffect 2',
    }

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    ax_b = fig.add_subplot(gs[0, :])
    for st in [1, 4, 7]:
        lo, hi = stage_bounds[st]
        ax_b.axvspan(lo, hi, color='lightgray', alpha=0.3, zorder=0)
    for st, (lo, hi) in stage_bounds.items():
        ax_b.text((lo + hi) / 2, 44, labels[st], ha='center', va='bottom', fontsize=7, color='gray')
    for boundary in [10, 20, 60, 70, 110, 150]:
        ax_b.axvline(boundary, color='gray', lw=0.5, ls='--', alpha=0.5)

    for stg in [3, 6]:
        seg = pdf[(pdf['fbi'] == 1) & (pdf['stage'] == stg)]
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#C8A2C8', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=50, color='#7B3F8D', zorder=3, alpha=0.7, edgecolors='none')

    for stg in [3, 6]:
        seg = pdf[(pdf['fbi'] == 0) & (pdf['stage'] == stg)].copy()
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta', cycle_col='probe_cycle')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#E8888A', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=50, color='#D44', zorder=3, alpha=0.7, edgecolors='none')

    for stg in [1, 2, 4, 5, 7]:
        seg = pdf[pdf['stage'] == stg]
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#E8888A', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=50, color='#D44', zorder=3, alpha=0.7, edgecolors='none')

    cyc, m, _ = _cycle_mean_ci(pdf, 'model_hand')
    ax_b.plot(cyc, m, color='#7B3F8D', lw=2, zorder=4, alpha=0.9)

    red_data = pdf[~((pdf['fbi'] == 1) & (pdf['stage'].isin([3, 6])))]
    cyc, m, _ = _cycle_mean_ci(red_data, 'model_probe')
    ax_b.plot(cyc, m, color='darkred', lw=2, zorder=4, alpha=0.9)

    ax_b.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_b.set_xlim(-1, 161)
    ax_b.set_ylim(-5, 48)
    ax_b.set_xlabel('Cycle Number (4 Movements)')
    ax_b.set_ylabel('Hand Angle (deg)')
    ax_b.set_title(f'Full Experiment Timecourse (N={pdf["pp"].nunique()})', fontsize=11)
    ax_b.legend(
        handles=[
            Line2D(
                [],
                [],
                marker='o',
                color='#7B3F8D',
                markersize=5,
                ls='None',
                label='Human total (aim+implicit)',
            ),
            Line2D(
                [], [], marker='o', color='#D44', markersize=5, ls='None', label='Human implicit'
            ),
            Line2D([], [], color='#7B3F8D', lw=2, label='Model aim+implicit'),
            Line2D([], [], color='darkred', lw=2, label='Model implicit'),
        ],
        fontsize=8,
        loc='upper left',
        framealpha=0.7,
    )
    sns.despine(ax=ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    r1 = pdf[(pdf['stage'] == 3) & (pdf['fbi'] == 1)].copy()
    r1['rel_cycle'] = r1['cycle'] - 20
    r2 = pdf[(pdf['stage'] == 6) & (pdf['fbi'] == 1)].copy()
    r2['rel_cycle'] = r2['cycle'] - 110
    for rdf, color in [(r1, '#C8A2C8'), (r2, '#4B0082')]:
        cyc, m, ci = _cycle_mean_ci(rdf, 'hand_theta', cycle_col='rel_cycle')
        ax_c.fill_between(cyc, m - ci, m + ci, color=color, alpha=0.25)
        ax_c.scatter(cyc, m, s=50, color=color, alpha=0.7, edgecolors='none')
    for rdf, color in [(r1, '#C8A2C8'), (r2, '#4B0082')]:
        cyc, m, _ = _cycle_mean_ci(rdf, 'model_hand', cycle_col='rel_cycle')
        ax_c.plot(cyc, m, color=color, lw=2.5, alpha=0.9)
    ax_c.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_c.set_xlim(-1, 41)
    ax_c.set_ylim(-5, 48)
    ax_c.set_xlabel('Rotation Cycles')
    ax_c.set_ylabel('Hand Angle (deg)')
    ax_c.set_title('Rotation 1 vs. 2 (Feedback)')
    ax_c.legend(
        handles=[
            Line2D(
                [], [], marker='o', color='#C8A2C8', markersize=5, ls='None', label='Human rot 1'
            ),
            Line2D(
                [], [], marker='o', color='#4B0082', markersize=5, ls='None', label='Human rot 2'
            ),
            Line2D([], [], color='#C8A2C8', lw=2.5, label='Model rot 1'),
            Line2D([], [], color='#4B0082', lw=2.5, label='Model rot 2'),
        ],
        fontsize=8,
        loc='lower right',
        framealpha=0.7,
    )
    sns.despine(ax=ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    p1 = pdf[(pdf['stage'] == 3) & (pdf['fbi'] == 0)].copy()
    p1['rel_probe_cycle'] = p1['probe_cycle'] - 20
    p2 = pdf[(pdf['stage'] == 6) & (pdf['fbi'] == 0)].copy()
    p2['rel_probe_cycle'] = p2['probe_cycle'] - 110
    for rdf, color in [(p1, '#E8888A'), (p2, '#8B0000')]:
        cyc, m, ci = _cycle_mean_ci(rdf, 'hand_theta', cycle_col='rel_probe_cycle')
        ax_d.fill_between(cyc, m - ci, m + ci, color=color, alpha=0.25)
        ax_d.scatter(cyc, m, s=50, color=color, alpha=0.7, edgecolors='none')
    for rdf, color in [(p1, '#E8888A'), (p2, '#8B0000')]:
        cyc, m, _ = _cycle_mean_ci(rdf, 'model_probe', cycle_col='rel_probe_cycle')
        ax_d.plot(cyc, m, color=color, lw=2.5, alpha=0.9)
    ax_d.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_d.set_xlim(-1, 41)
    ax_d.set_ylim(-5, 35)
    ax_d.set_xlabel('Rotation Cycles')
    ax_d.set_ylabel('Hand Angle (deg)')
    ax_d.set_title('Probe 1 vs. 2')
    ax_d.legend(
        handles=[
            Line2D(
                [], [], marker='o', color='#E8888A', markersize=5, ls='None', label='Human probe 1'
            ),
            Line2D(
                [], [], marker='o', color='#8B0000', markersize=5, ls='None', label='Human probe 2'
            ),
            Line2D([], [], color='#E8888A', lw=2.5, label='Model probe 1'),
            Line2D([], [], color='#8B0000', lw=2.5, label='Model probe 2'),
        ],
        fontsize=8,
        loc='lower right',
        framealpha=0.7,
    )
    sns.despine(ax=ax_d)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved savings panels to {save_path}")
    plt.close(fig)


_WC_MAG_COLOURS = {
    0: '#2ca02c',
    15: '#4363d8',
    30: '#e6194B',
    45: '#9467bd',
}
_WC_BL_WO_COLOUR = '#e68a00'


def plot_wildcard_figure(csv_path, model_path, save_path=None, max_epoch=60, final_block=4):
    import json as _json
    from collections import defaultdict
    from scipy.special import logsumexp as sp_logsumexp
    from matplotlib.ticker import MultipleLocator

    def _parse(s):
        return _json.loads(s.replace('nan', 'null'))

    def _phase_bounds(rots):
        n = len(rots)
        first = next(i for i in range(n) if rots[i] is not None and rots[i] != 0)
        last = max(i for i in range(n) if rots[i] is not None and rots[i] != 0)
        return first, last

    N_BL = 5
    N_WO = 5
    df_wide = pd.read_csv(csv_path)
    comp_aim, bl_aim, wo_aim, pp_mags = {}, {}, {}, {}
    for idx in range(len(df_wide)):
        row = df_wide.iloc[idx]
        pp = int(row['participantNum'])
        rots = _parse(row['rotation'])
        aims = _parse(row['aim'])
        n = len(rots)
        first_rot, last_rot = _phase_bounds(rots)
        bl_s = max(0, first_rot - N_BL)
        bl_aim[pp] = np.array(
            [aims[i] if aims[i] is not None else np.nan for i in range(bl_s, first_rot)]
        )
        wo_s = last_rot + 1
        wo_aim[pp] = np.array(
            [aims[i] if aims[i] is not None else np.nan for i in range(wo_s, min(wo_s + N_WO, n))]
        )
        mag_trials = defaultdict(list)
        for i in range(first_rot, last_rot + 1):
            r, a = rots[i], aims[i]
            if a is None:
                continue
            mag = abs(r) if (r is not None and r != 0) else 0
            if mag == 0:
                mag_trials[0].append(a)
            else:
                mag_trials[mag].append(-a * np.sign(r))
        comp_aim[pp] = {m: np.array(v) for m, v in mag_trials.items()}
        pp_mags[pp] = set(mag_trials.keys())

    mags_present = sorted(set.union(*pp_mags.values()))
    arr = np.load(model_path, allow_pickle=True)
    bht = arr[0]
    model_aim = {}
    for i, pp in enumerate(bht.participantNums):
        row = df_wide[df_wide['participantNum'] == pp].iloc[0]
        rots = _parse(row['rotation'])
        n = len(rots)
        ps = bht.predState[pp]
        comp_logp = np.array(ps['predCompLogProb'])
        comp_mean = np.array(ps['predCompMean'])
        n_comp = np.array(ps['predNComp'])
        n_model = len(n_comp)
        m_aim = np.full(n_model, np.nan)
        for t in range(n_model):
            nc = int(n_comp[t])
            if nc == 0:
                continue
            lp = comp_logp[t, :nc]
            mn = comp_mean[t, :nc]
            wts = np.exp(lp - sp_logsumexp(lp))
            m_aim[t] = np.dot(wts, mn)
        if n_model < n:
            m_aim = np.concatenate([m_aim, np.full(n - n_model, np.nan)])
        first_rot, last_rot = _phase_bounds(rots)
        mag_aim = defaultdict(list)
        for j in range(first_rot, min(last_rot + 1, len(m_aim))):
            r = rots[j]
            mag = abs(r) if (r is not None and r != 0) else 0
            aim_val = m_aim[j]
            if np.isfinite(aim_val):
                raw = aim_val if mag == 0 else -aim_val * np.sign(r)
                raw = np.clip(raw, -200, 200)
                mag_aim[mag].append(raw)
            else:
                mag_aim[mag].append(np.nan)
        model_aim[pp] = {m: np.array(v) for m, v in mag_aim.items()}

    def _epoch_mat(data, mag, max_ep):
        arrays = [v[mag] for v in data.values() if mag in v]
        if not arrays:
            return None
        ml = min(max(len(a) for a in arrays), max_ep)
        mat = np.full((len(arrays), ml), np.nan)
        for k, a in enumerate(arrays):
            mat[k, : min(len(a), ml)] = a[:ml]
        return mat

    def _ep_stats(data, mags, max_ep):
        out = {}
        for mag in mags:
            mat = _epoch_mat(data, mag, max_ep)
            if mat is None:
                continue
            nv = np.sum(~np.isnan(mat), axis=0)
            m = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(nv, 1))
            ci = 1.96 * sem
            out[mag] = (np.arange(1, mat.shape[1] + 1), m, m - ci, m + ci)
        return out

    def _ph_stats(ph_dict):
        arrays = list(ph_dict.values())
        ml = max(len(a) for a in arrays)
        mat = np.full((len(arrays), ml), np.nan)
        for k, a in enumerate(arrays):
            mat[k, : len(a)] = a
        nv = np.sum(~np.isnan(mat), axis=0)
        m = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(nv, 1))
        ci = 1.96 * sem
        return np.arange(1, ml + 1), m, m - ci, m + ci

    h_rot = _ep_stats(comp_aim, mags_present, max_epoch)
    h_bl = _ph_stats(bl_aim)
    h_wo = _ph_stats(wo_aim)
    m_rot = _ep_stats(model_aim, mags_present, max_epoch)

    nonzero_mags = [m for m in mags_present if m > 0]
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.35}
    )

    bl_ep, bl_m, bl_lo, bl_hi = h_bl
    n_bl = len(bl_ep)
    x_bl = np.arange(1, n_bl + 1)
    ax_a.plot(x_bl, bl_m, '-', marker='o', color=_WC_BL_WO_COLOUR, ms=2.5, lw=2.4, zorder=5)
    ax_a.fill_between(x_bl, bl_lo, bl_hi, color=_WC_BL_WO_COLOUR, alpha=0.18)

    rot_start = n_bl + 1
    max_rot_len = 0
    for mag in mags_present:
        if mag not in h_rot:
            continue
        ep, m, lo, hi = h_rot[mag]
        x = ep + rot_start - 1
        c = _WC_MAG_COLOURS.get(mag, '#333')
        ax_a.fill_between(x, lo, hi, color=c, alpha=0.15)
        lbl = f'{int(mag)}°'
        ax_a.plot(x, m, '-', color=c, lw=1.3, label=lbl, zorder=4)
        max_rot_len = max(max_rot_len, int(ep[-1]))
        if mag in m_rot:
            mep, mm, mlo, mhi = m_rot[mag]
            mx = mep + rot_start - 1
            ax_a.plot(mx, mm, '--', color=c, lw=1.3, alpha=0.85, zorder=4)
            ax_a.fill_between(mx, mlo, mhi, color=c, alpha=0.06)

    wo_ep, wo_m, wo_lo, wo_hi = h_wo
    wo_start = rot_start + max_rot_len
    x_wo = wo_ep + wo_start - 1
    ax_a.plot(x_wo, wo_m, '-', marker='o', color=_WC_BL_WO_COLOUR, ms=2.5, lw=2.4, zorder=5)
    ax_a.fill_between(x_wo, wo_lo, wo_hi, color=_WC_BL_WO_COLOUR, alpha=0.18)

    ax_a.axvline(n_bl + 0.5, color='grey', ls=':', lw=1.8, alpha=0.6)
    ax_a.axvline(wo_start - 0.5, color='grey', ls=':', lw=1.8, alpha=0.6)
    for h in nonzero_mags:
        ax_a.axhline(h, color=_WC_MAG_COLOURS.get(h, '#333'), ls=':', lw=1.8, alpha=0.4)
    ax_a.axhline(0, color='grey', ls=':', lw=1.8, alpha=0.4)
    ax_a.set_ylabel('Compensatory aim (°)')
    ax_a.set_xlabel('Movement epoch')
    ax_a.legend(
        loc='lower right', frameon=True, framealpha=0.85, edgecolor='none', ncol=2, fontsize=8
    )
    h_max = max(nonzero_mags) if nonzero_mags else 45
    ax_a.set_ylim(-30, 60)
    ax_a.yaxis.set_major_locator(MultipleLocator(15))
    ax_a.set_title('A: Compensatory Aim Timecourse', fontsize=10)
    sns.despine(ax=ax_a)

    positions = list(range(len(mags_present)))
    rng = np.random.default_rng(42)
    for pos, mag in enumerate(mags_present):
        c = _WC_MAG_COLOURS.get(mag, '#333')
        h_pp_means = []
        for pp, d in comp_aim.items():
            if mag in d and len(d[mag]) >= final_block:
                h_pp_means.append(np.nanmean(d[mag][-final_block:]))
        h_arr = np.array(h_pp_means)
        jitter_h = rng.uniform(-0.35, -0.15, len(h_arr))
        ax_b.scatter(pos + jitter_h, h_arr, color=c, s=50, alpha=0.15, edgecolors='none', zorder=5)
        if len(h_arr) > 0:
            ax_b.plot(
                pos - 0.15,
                np.nanmean(h_arr),
                'o',
                color=c,
                ms=10,
                markeredgecolor='k',
                markeredgewidth=0.8,
                zorder=8,
            )
        m_pp_means = []
        for pp, d in model_aim.items():
            if mag in d and len(d[mag]) >= final_block:
                m_pp_means.append(np.nanmean(d[mag][-final_block:]))
        m_arr = np.array(m_pp_means)
        jitter_m = rng.uniform(0.15, 0.35, len(m_arr))
        ax_b.scatter(
            pos + jitter_m,
            m_arr,
            color=c,
            s=50,
            alpha=0.15,
            edgecolors='none',
            zorder=5,
            marker='D',
        )
        if len(m_arr) > 0:
            ax_b.plot(
                pos + 0.15,
                np.nanmean(m_arr),
                'D',
                color=c,
                ms=10,
                markeredgecolor='k',
                markeredgewidth=0.8,
                zorder=8,
            )

    for h in nonzero_mags:
        ax_b.axhline(h, color=_WC_MAG_COLOURS.get(h, '#333'), ls=':', lw=1.8, alpha=0.4)
    ax_b.axhline(0, color='grey', ls=':', lw=1.8, alpha=0.4)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([f'{int(m)}°' for m in mags_present])
    ax_b.set_xlabel('Rotation magnitude (°)')
    ax_b.set_ylabel('Compensatory aim (°)')
    ax_b.set_title(f'B: Final {final_block} Epochs', fontsize=10)
    ax_b.yaxis.set_major_locator(MultipleLocator(15))
    ax_b.set_ylim(-30, 60)
    ax_b.legend(
        handles=[
            Line2D(
                [],
                [],
                marker='o',
                color='grey',
                markersize=5,
                ls='None',
                markeredgecolor='k',
                label='Human mean',
            ),
            Line2D(
                [],
                [],
                marker='D',
                color='grey',
                markersize=5,
                ls='None',
                markeredgecolor='k',
                label='Model mean',
            ),
        ],
        fontsize=8,
        loc='upper left',
        framealpha=0.7,
    )
    sns.despine(ax=ax_b)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved wildcard figure to {save_path}")
    plt.close(fig)


def _compute_state_probs(ps):
    """Compute P(M0), P(rotation), P(translation) from predState dict.

    Returns arrays of shape (T,) for each.
    H0 (idx 0) = harmonic/translation, H1 (idx 1) = DC/rotation.
    """
    from scipy.special import softmax as sp_softmax

    postLogQ0 = ps['postLogQ0']  # log P(M0) after update
    structLogW = ps['predStructLogW']  # (T, 2) log P(H_m)
    T = len(postLogQ0)
    p_m0 = np.exp(np.clip(postLogQ0, -500, 0))
    p_m1 = 1.0 - p_m0
    p_struct = sp_softmax(structLogW, axis=1)  # (T, 2)
    p_rot = p_m1 * p_struct[:, 1]  # DC = rotation
    p_trans = p_m1 * p_struct[:, 0]  # harmonic = translation
    return p_m0, p_rot, p_trans


def _build_dir_weights(ps):
    """Construct synthetic per-run per-target weights from the new model's arrays.

    The new model stores:
      - predSelfOtherW[t, k, :] = [w_self, w_other]  (Beta-Bernoulli self/other)
      - predGibbsW[t, i, j]     = Gibbs base-measure weight from i to j
      - predTgtIdx[t]            = index of trial t's target

    Returns an array of shape (T, MAXRUN, K) equivalent to the old predDirWeights:
      dirW[t, k, j] = w_self   if j == tgtIdx[t]
                     = w_other * gibbsW_norm[tgtIdx[t], j]   otherwise
    where gibbsW_norm is the row-normalised Gibbs weights excluding the self-target.
    """
    soW = ps['predSelfOtherW']  # (T, MAXRUN, 2)
    gW = ps['predGibbsW']  # (T, K_full, K_full)
    tIdx = ps['predTgtIdx']  # (T,)
    T, maxrun, _ = soW.shape
    K = gW.shape[1]
    dirW = np.zeros((T, maxrun, K))
    for t in range(T):
        i = int(tIdx[t])
        if i < 0 or i >= K:
            continue
        for k in range(maxrun):
            w_s = soW[t, k, 0]
            w_o = soW[t, k, 1]
            dirW[t, k, i] = w_s
            # Distribute w_other across non-self targets via Gibbs weights
            gw_row = gW[t, i, :K].copy()
            gw_row[i] = 0.0  # exclude self
            gw_sum = gw_row.sum()
            if gw_sum > 1e-30 and w_o > 1e-30:
                for j in range(K):
                    if j != i:
                        dirW[t, k, j] = w_o * gw_row[j] / gw_sum
    return dirW


def _compute_dirichlet_self_vs_other(ps, destTargArr, gen_indices=None):
    """Compute run-weighted self/other Dirichlet weights per trial.

    Returns:
        self_w:  (T,) self-weight for current target
        other_w: (T,) sum of other-target weights
    For gen_indices split, returns dict keyed by 'train' and 'gen' with
    sub-dicts {'self': ..., 'other_train': ..., 'other_gen': ...}.
    """
    from scipy.special import logsumexp as sp_lse

    dirW = _build_dir_weights(ps)  # (T, MAXRUN, K)
    tgtIdx = ps['predTgtIdx']  # (T,)
    preLogR = ps['preLogR']  # (T, MAXRUN)
    preLogQ0 = ps['preLogQ0']  # (T,)
    T, maxrun, K = dirW.shape
    nDest = len(destTargArr)

    self_w = np.full(T, np.nan)
    other_w = np.full(T, np.nan)

    for t in range(T):
        ti = int(tgtIdx[t])
        if ti < 0 or ti >= nDest:
            continue
        # Run probabilities (conditional on M1)
        logR = preLogR[t, :maxrun].copy()
        # Normalise run probs (exclude M0)
        valid_runs = logR > -500
        if not np.any(valid_runs):
            continue
        logR_norm = logR.copy()
        logR_norm[~valid_runs] = -1e30
        ls = sp_lse(logR_norm)
        run_p = np.exp(logR_norm - ls)
        eff_w = np.zeros(nDest)
        any_valid = False
        for k in range(maxrun):
            if run_p[k] < 1e-15:
                continue
            dw_row = dirW[t, k, :nDest]
            if np.any(np.isnan(dw_row)):
                continue
            row_sum = dw_row.sum()
            if row_sum < 1e-30:
                continue
            eff_w += run_p[k] * dw_row / row_sum
            any_valid = True
        if not any_valid:
            continue
        self_w[t] = eff_w[ti]
        other_w[t] = np.sum(eff_w) - eff_w[ti]
    return self_w, other_w


def _compute_dirichlet_per_target(ps, destTargArr):
    """Compute run-weighted Dirichlet weight for each target on every trial.

    Unlike _compute_dirichlet_self_vs_other which only reports the active
    target's self-weight, this returns a (T, nDest) array so each target's
    weight can be plotted continuously across all trials.

    Returns:
        weights: (T, nDest) array of normalised Dirichlet weights
    """
    from scipy.special import logsumexp as sp_lse

    dirW = _build_dir_weights(ps)  # (T, MAXRUN, K)
    preLogR = ps['preLogR']  # (T, MAXRUN)
    T, maxrun, K = dirW.shape
    nDest = len(destTargArr)

    weights = np.full((T, nDest), np.nan)

    for t in range(T):
        logR = preLogR[t, :maxrun].copy()
        valid_runs = logR > -500
        if not np.any(valid_runs):
            continue
        logR_norm = logR.copy()
        logR_norm[~valid_runs] = -1e30
        ls = sp_lse(logR_norm)
        run_p = np.exp(logR_norm - ls)
        eff_w = np.zeros(nDest)
        any_valid = False
        for k in range(maxrun):
            if run_p[k] < 1e-15:
                continue
            dw_row = dirW[t, k, :nDest]
            if np.any(np.isnan(dw_row)):
                continue
            row_sum = dw_row.sum()
            if row_sum < 1e-30:
                continue
            eff_w += run_p[k] * dw_row / row_sum
            any_valid = True
        if any_valid:
            weights[t, :] = eff_w
    return weights


def _compute_dirichlet_current_split(ps, destTargArr, train_indices, gen_indices):
    """Compute self / other-train / other-gen weights on EVERY trial.

    Uses _compute_dirichlet_per_target so weights are available on all trials,
    not just when a particular target type was active.

    Returns 6 arrays (T,):
        train_self, train_ot, train_og — mean self/other-train/other-gen when
            a training target is the active target
        gen_self, gen_ot, gen_og — same when a gen target is active
    All arrays are filled on every trial (no NaN gaps from target schedule).
    """
    wt = _compute_dirichlet_per_target(ps, destTargArr)  # (T, nDest)
    tgtIdx = ps['predTgtIdx']
    T = wt.shape[0]
    nDest = len(destTargArr)
    train_set = set(train_indices)
    gen_set_idx = set(gen_indices)

    tr_s = np.full(T, np.nan)
    tr_ot = np.full(T, np.nan)
    tr_og = np.full(T, np.nan)
    ge_s = np.full(T, np.nan)
    ge_ot = np.full(T, np.nan)
    ge_og = np.full(T, np.nan)

    for t in range(T):
        if np.isnan(wt[t, 0]):
            continue
        ti = int(tgtIdx[t])
        if ti < 0 or ti >= nDest:
            continue
        sw = wt[t, ti]
        ot = sum(wt[t, j] for j in train_indices if j != ti)
        og = sum(wt[t, j] for j in gen_indices if j != ti)
        if ti in train_set:
            tr_s[t], tr_ot[t], tr_og[t] = sw, ot, og
        elif ti in gen_set_idx:
            ge_s[t], ge_ot[t], ge_og[t] = sw, ot, og
    return tr_s, tr_ot, tr_og, ge_s, ge_ot, ge_og


def _compute_dirichlet_by_target_type(ps, destTargArr, gen_set):
    """Split Dirichlet into self/other for train vs gen, filled on ALL trials.

    Returns dict with bool masks and weight arrays, all shape (T,).
    Weights are computed for every trial using _compute_dirichlet_per_target.
    """
    wt = _compute_dirichlet_per_target(ps, destTargArr)  # (T, nDest)
    tgtIdx = ps['predTgtIdx']
    T = wt.shape[0]
    nDest = len(destTargArr)

    gen_idx_set = set()
    train_idx_set = set()
    for j, tg in enumerate(destTargArr):
        if tg in gen_set:
            gen_idx_set.add(j)
        else:
            train_idx_set.add(j)

    self_w = np.full(T, np.nan)
    train_other_w = np.full(T, np.nan)
    gen_other_w = np.full(T, np.nan)
    is_train = np.zeros(T, dtype=bool)
    is_gen = np.zeros(T, dtype=bool)

    for t in range(T):
        ti = int(tgtIdx[t])
        if ti < 0 or ti >= nDest or np.isnan(wt[t, 0]):
            continue
        is_train[t] = ti in train_idx_set
        is_gen[t] = ti in gen_idx_set

        self_w[t] = wt[t, ti]
        tw = sum(wt[t, j] for j in train_idx_set if j != ti)
        gw = sum(wt[t, j] for j in gen_idx_set if j != ti)
        train_other_w[t] = tw
        gen_other_w[t] = gw

    return {
        'is_train': is_train,
        'is_gen': is_gen,
        'self_w': self_w,
        'train_other_w': train_other_w,
        'gen_other_w': gen_other_w,
    }


def _compute_dirichlet_for_target_set(ps, destTargArr, target_indices):
    """Compute per-target-set Dirichlet self/other on every trial.

    Uses _compute_dirichlet_per_target, so it works for both old fits with
    stored predDirWeights and newer fits that reconstruct target weights from
    predSelfOtherW/predGibbsW.
    """
    wt = _compute_dirichlet_per_target(ps, destTargArr)  # (T, nDest)
    T, nDest = wt.shape

    self_w = np.full(T, np.nan)
    other_w = np.full(T, np.nan)

    valid_targets = [j for j in target_indices if 0 <= j < nDest]
    if len(valid_targets) == 0:
        return self_w, other_w

    for t in range(T):
        row = wt[t, :nDest]
        if not np.all(np.isfinite(row)):
            continue
        row_sum = np.sum(row)
        self_vals = np.array([row[j] for j in valid_targets], dtype=float)
        self_w[t] = np.mean(self_vals)
        other_w[t] = np.mean(row_sum - self_vals)
    return self_w, other_w


def _compute_dirichlet_for_target_set_split(
    ps, destTargArr, query_indices, train_indices, gen_indices
):
    """Like _compute_dirichlet_for_target_set but splits 'other' into
    other-train and other-gen.

    Returns (self_w, other_train_w, other_gen_w) each shape (T,).
    """
    wt = _compute_dirichlet_per_target(ps, destTargArr)  # (T, nDest)
    T, nDest = wt.shape

    self_w = np.full(T, np.nan)
    other_train_w = np.full(T, np.nan)
    other_gen_w = np.full(T, np.nan)

    valid_query = [j for j in query_indices if 0 <= j < nDest]
    valid_train = [j for j in train_indices if 0 <= j < nDest]
    valid_gen = [j for j in gen_indices if 0 <= j < nDest]
    if len(valid_query) == 0:
        return self_w, other_train_w, other_gen_w

    for t in range(T):
        row = wt[t, :nDest]
        if not np.all(np.isfinite(row)):
            continue

        s_acc = 0.0
        ot_acc = 0.0
        og_acc = 0.0
        n_q = 0
        for j in valid_query:
            s_acc += row[j]
            ot_acc += sum(row[i] for i in valid_train if i != j)
            og_acc += sum(row[i] for i in valid_gen if i != j)
            n_q += 1
        if n_q > 0:
            self_w[t] = s_acc / n_q
            other_train_w[t] = ot_acc / n_q
            other_gen_w[t] = og_acc / n_q
    return self_w, other_train_w, other_gen_w


def plot_bic_comparison(
    bht_fits, hmm_path, save_path, dataset_label=None, debug=False, ding_aggregate=False
):
    """ΔBIC (HMM − BHT) per participant, grouped by condition.

    Parameters
    ----------
    bht_fits : ndarray
        Loaded BHT .npy (dict or array of FitShell objects).
    hmm_path : str
        Path to the HMM .npy file.
    save_path : str
        Output figure path.
    dataset_label : str, optional
        Override for figure title.
    debug : bool
    """
    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8

    try:
        hmm_raw = np.load(hmm_path, allow_pickle=True)
    except FileNotFoundError:
        if debug:
            print(f"HMM file not found: {hmm_path}, skipping.")
        return
    hmm_loaded = hmm_raw.item() if np.ndim(hmm_raw) == 0 else hmm_raw
    if isinstance(hmm_loaded, dict):
        hmm_entries = list(hmm_loaded.items())
        hmm_list = [v for _, v in hmm_entries]
    else:
        hmm_list = extract_fit_shells(hmm_loaded)
        hmm_entries = [(None, hobj) for hobj in hmm_list]

    # Build ppNum → (bic, conVal) lookup for HMM
    # HMM may be pooled (single object, conVal='none') or per-condition
    def _norm_name(_name):
        _parts = str(_name).split('_')
        _core = [p for p in _parts if p not in ('Ding', 'EqMeans', 'AllShared', 'HMMAllShared')]
        return '_'.join(_core) if len(_core) > 0 else str(_name)

    def _group_name(_name):
        _norm = _norm_name(_name)
        _parts = _norm.split('_')
        return '_'.join(_parts[:-1]) if len(_parts) >= 2 else _norm

    hmm_is_pooled = (
        len(hmm_list) == 1
        and not isinstance(hmm_loaded, dict)
        and getattr(hmm_list[0], 'conVal', 'none') in ('none', None, '')
    )
    hmm_pp_bic = {}
    hmm_cond_lookup = {}  # pp -> {conVal: bic}
    hmm_name_lookup = {}  # (condition/group, pp) -> bic
    if hmm_is_pooled:
        hobj = hmm_list[0]
        for i, pp in enumerate(hobj.participantNums):
            hmm_pp_bic[pp] = hobj.bics[i]
    else:
        for hkey, hobj in hmm_entries:
            if hobj is None or not hasattr(hobj, 'participantNums'):
                continue
            hcv = str(getattr(hobj, 'conVal', 'none'))
            if hkey is None:
                if hasattr(hobj, 'df') and 'condition' in hobj.df.columns and len(hobj.df) > 0:
                    _vals = hobj.df['condition'].dropna().astype(str).unique()
                    hfull = _vals[0] if len(_vals) > 0 else getattr(hobj, 'datasetName', hcv)
                else:
                    hfull = getattr(hobj, 'datasetName', hcv)
            else:
                hfull = str(hkey)
            hfull_norm = _norm_name(hfull)
            hgroup = _group_name(hfull)
            for i, pp in enumerate(hobj.participantNums):
                _bic = hobj.bics[i]
                hmm_cond_lookup.setdefault(pp, {})[hcv] = _bic
                hmm_name_lookup[(hfull_norm, pp)] = _bic
                hmm_name_lookup[(hgroup, pp)] = _bic

    bht_objs = extract_fit_shells(bht_fits)
    if ding_aggregate:
        # Merge conditions into groups (e.g. Inner_2T pools 90/270/315)
        _grouped = _aggregate_obj_groups(bht_objs)
        bht_objs = [_merge_obj_group(gk, gos) for gk, gos in _grouped]
    cond_data = []  # list of (label, diffs_array)

    for bobj in bht_objs:
        bcv = str(getattr(bobj, 'conVal', 'none'))
        blabel = bcv if bcv not in ('none', 'None', '') else getattr(bobj, 'datasetName', '?')
        bfull = _norm_name(getattr(bobj, 'datasetName', blabel))
        bgroup = _group_name(getattr(bobj, 'datasetName', blabel))
        diffs = []
        for i, pp in enumerate(bobj.participantNums):
            if hmm_is_pooled:
                hbic = hmm_pp_bic.get(pp)
            else:
                hbic = (
                    hmm_name_lookup.get((bgroup, pp))
                    if ding_aggregate
                    else hmm_name_lookup.get((bfull, pp))
                )
                if hbic is None:
                    hbic = hmm_cond_lookup.get(pp, {}).get(bcv)
                if hbic is None and ding_aggregate:
                    hbic = hmm_name_lookup.get((bfull, pp))
            if hbic is not None:
                diffs.append(hbic - bobj.bics[i])
        if len(diffs) > 0:
            cond_data.append((blabel, np.array(diffs)))
            if debug:
                d = np.array(diffs)
                print(
                    f"  {blabel}: N={len(d)}, mean ΔBIC={np.mean(d):.1f}, median={np.median(d):.1f}"
                )

    if len(cond_data) == 0:
        if debug:
            print(f"No matching participants between BHT and HMM, skipping.")
        return

    n_conds = len(cond_data)
    fig, ax = plt.subplots(figsize=(max(4, 1.5 * n_conds + 2), 5))

    positions = np.arange(n_conds)
    labels = []

    for ci, (label, diffs) in enumerate(cond_data):
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(diffs))
        ax.scatter(
            ci + jitter, diffs, s=18, alpha=0.35, color='#555555', edgecolors='none', zorder=2
        )
        m = np.mean(diffs)
        sem = np.std(diffs) / np.sqrt(len(diffs))
        ax.errorbar(
            ci,
            m,
            yerr=1.96 * sem,
            fmt='o',
            color='#d62728',
            markersize=8,
            capsize=5,
            capthick=1.5,
            lw=1.5,
            zorder=3,
        )
        labels.append(f'{label}\n(n={len(diffs)})')

    ax.axhline(0, color='black', ls='--', lw=2.4, zorder=1)

    xlims = ax.get_xlim()
    ax.text(
        xlims[1],
        0,
        '  BHT = HMM',
        va='center',
        ha='left',
        fontsize=FS_LEGEND,
        style='italic',
        color='#666666',
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('ΔBIC  (HMM − BHT)', fontsize=FS_LABEL)
    ax.tick_params(axis='y', labelsize=FS_TICK)

    # Robust y-axis limits (clip extreme outliers for readability)
    all_diffs = np.concatenate([d for _, d in cond_data])
    q01, q99 = np.percentile(all_diffs, [1, 99])
    iqr = q99 - q01
    ylo = min(q01 - 0.5 * iqr, -10)
    yhi = max(q99 + 0.5 * iqr, 10)
    ylo = min(ylo, -0.05 * (yhi - ylo))
    yhi = max(yhi, 0.05 * (yhi - ylo))
    ax.set_ylim(ylo, yhi)

    # Count and annotate clipped points per condition
    for ci, (label, diffs) in enumerate(cond_data):
        n_below = np.sum(diffs < ylo)
        n_above = np.sum(diffs > yhi)
        if n_below > 0:
            ax.annotate(
                f'{n_below} below',
                xy=(ci, ylo),
                fontsize=FS_LEGEND - 1,
                ha='center',
                va='bottom',
                color='#721c24',
                style='italic',
            )
        if n_above > 0:
            ax.annotate(
                f'{n_above} above',
                xy=(ci, yhi),
                fontsize=FS_LEGEND - 1,
                ha='center',
                va='top',
                color='#155724',
                style='italic',
            )

    ax.set_xlim(-0.5, n_conds - 0.5)
    ylims = ax.get_ylim()

    ax.fill_between([-0.5, n_conds - 0.5], 0, ylims[1], color='#d4edda', alpha=0.15, zorder=0)
    ax.fill_between([-0.5, n_conds - 0.5], ylims[0], 0, color='#f8d7da', alpha=0.15, zorder=0)

    ax.text(
        n_conds - 0.55,
        ylims[1] * 0.92,
        'BHT better',
        ha='right',
        fontsize=FS_LEGEND,
        color='#155724',
        style='italic',
    )
    ax.text(
        n_conds - 0.55,
        ylims[0] + 0.08 * (ylims[1] - ylims[0]),
        'HMM better',
        ha='right',
        fontsize=FS_LEGEND,
        color='#721c24',
        style='italic',
    )

    title = dataset_label or 'BIC comparison'
    ax.set_title(f'{title}: ΔBIC (HMM − BHT)', fontsize=FS_TITLE)
    sns.despine(ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if debug:
        print(f"  Saved: {save_path}")


def plot_state_dirichlet_timeseries(fits, save_dir, debug=False, ding_aggregate=False):
    """Plot group-mean state probabilities and Dirichlet self/other over trials.

    Uses stacked-area fill style (probability determines filled height).

    Layout (1 column):
        Row 0: state probs (M0, Rotation, Translation) – stacked to 1.0
        Row 1: Dirichlet – training targets – stacked to 1.0
        Row 2: Dirichlet – gen targets      (only if dataset has gen targets)

    Dirichlet weights are evaluated for ALL trials (not just when a particular
    target type was presented), so there are no NaN gaps from the target schedule.
    """
    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8
    os.makedirs(save_dir, exist_ok=True)
    obj_list = extract_fit_shells(fits)

    if ding_aggregate:
        _items = [(gk, _merge_obj_group(gk, gos)) for gk, gos in _aggregate_obj_groups(obj_list)]
    else:
        _items = [(getattr(o, 'datasetName', ''), o) for o in obj_list]

    for _gk, obj in _items:
        _apply_fit_model_config(obj, debug=debug)
        datasetLabel = _gk if _gk else getattr(obj, 'datasetName', 'dataset')
        conVal = getattr(obj, 'conVal', '')
        suffix = f"_{conVal}" if conVal and conVal != 'none' else ''
        N = len(obj.participantNums)

        has_predState = hasattr(obj, 'predState') and obj.predState is not None
        if not has_predState:
            if debug:
                print(f"Skipping {datasetLabel}{suffix}: no predState")
            continue

        # Reference trial structure from first participant
        pp0 = obj.participantNums[0]
        pDat0 = obj.df[obj.df['participantNum'] == pp0].copy()
        phases0 = BHT.derivePhase(pDat0)
        fitWashout = getattr(obj, 'fitWashout', True)
        nwM0 = np.ones(len(phases0), dtype=bool) if fitWashout else (phases0 != 'washout')
        phases_filt = phases0[nwM0]
        T_ref = len(phases_filt)

        any_gen = False
        for pp in obj.participantNums:
            if len(getattr(obj, 'genTargets', {}).get(pp, set())) > 0:
                any_gen = True
                break

        block_bounds = _detect_block_boundaries(obj) if hasattr(obj, 'df') else []

        all_m0, all_rot, all_trans = [], [], []
        # For gen datasets: target-set weights on every trial, split into
        # self / other-train / other-gen for training vs gen query targets
        all_train_self, all_train_ot, all_train_og = [], [], []
        all_gen_self, all_gen_ot, all_gen_og = [], [], []
        all_self_w, all_other_w = [], []

        def _pad(arr, Tref):
            if len(arr) >= Tref:
                return arr[:Tref]
            return np.concatenate([arr, np.full(Tref - len(arr), np.nan)])

        skipped = 0
        for pIdx in range(N):
            pp = obj.participantNums[pIdx]
            ps = obj.predState.get(pp, None)
            if ps is None:
                skipped += 1
                continue

            p_m0, p_rot, p_trans = _compute_state_probs(ps)
            all_m0.append(_pad(p_m0, T_ref))
            all_rot.append(_pad(p_rot, T_ref))
            all_trans.append(_pad(p_trans, T_ref))

            destTargets_pp = getattr(obj, 'destTargets', {}).get(pp, [])
            destTA = np.array([x for x in destTargets_pp if not np.isnan(x)], dtype=np.float64)
            gen_set = getattr(obj, 'genTargets', {}).get(pp, set())
            nDest = len(destTA)

            if any_gen and len(gen_set) > 0:
                gen_idx = [j for j in range(nDest) if destTA[j] in gen_set]
                train_idx = [j for j in range(nDest) if destTA[j] not in gen_set]

                sw_tr, ot_tr, og_tr = _compute_dirichlet_for_target_set_split(
                    ps, destTA, train_idx, train_idx, gen_idx
                )
                sw_ge, ot_ge, og_ge = _compute_dirichlet_for_target_set_split(
                    ps, destTA, gen_idx, train_idx, gen_idx
                )

                all_train_self.append(_pad(sw_tr, T_ref))
                all_train_ot.append(_pad(ot_tr, T_ref))
                all_train_og.append(_pad(og_tr, T_ref))
                all_gen_self.append(_pad(sw_ge, T_ref))
                all_gen_ot.append(_pad(ot_ge, T_ref))
                all_gen_og.append(_pad(og_ge, T_ref))
            elif any_gen:
                # This pp has no gen targets but the dataset does
                all_train_self.append(np.full(T_ref, np.nan))
                all_train_ot.append(np.full(T_ref, np.nan))
                all_train_og.append(np.full(T_ref, np.nan))
                all_gen_self.append(np.full(T_ref, np.nan))
                all_gen_ot.append(np.full(T_ref, np.nan))
                all_gen_og.append(np.full(T_ref, np.nan))
            else:
                sw, ow = _compute_dirichlet_self_vs_other(ps, destTA)
                all_self_w.append(_pad(sw, T_ref))
                all_other_w.append(_pad(ow, T_ref))

        if debug:
            print(
                f"{datasetLabel}{suffix}: N={N}, skipped={skipped}, "
                f"T_ref={T_ref}, any_gen={any_gen}"
            )

        m0_mean = np.nanmean(all_m0, axis=0)
        rot_mean = np.nanmean(all_rot, axis=0)
        tra_mean = np.nanmean(all_trans, axis=0)

        trials = np.arange(T_ref)

        def _shade_phases(ax):
            bl_end = int(np.sum(phases_filt == 'baseline'))
            if bl_end > 0:
                ax.axvspan(-0.5, bl_end - 0.5, color='#f0f0f0', zorder=0)
            wo_mask = phases_filt == 'washout'
            if np.any(wo_mask):
                wo_idx = np.where(wo_mask)[0]
                ax.axvspan(wo_idx[0] - 0.5, wo_idx[-1] + 0.5, color='#f0f0f0', zorder=0)
            for bb in block_bounds:
                ax.axvline(bb, color='grey', ls=':', lw=0.8, alpha=0.6)

        def _style(ax, ylabel, title, show_xlabel=False):
            ax.set_ylabel(ylabel, fontsize=FS_LABEL)
            ax.set_title(title, fontsize=FS_TITLE)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, T_ref - 1)
            ax.tick_params(labelsize=FS_TICK)
            if show_xlabel:
                ax.set_xlabel('Trial', fontsize=FS_LABEL)
            _shade_phases(ax)
            sns.despine(ax=ax)

        def _plot_stacked(ax, means_list, labels, colors, ylabel, title, show_xlabel=False):
            """Stacked area fill from a list of (T,) mean arrays."""
            stack = np.array(means_list)
            totals = np.nansum(stack, axis=0)
            totals[totals == 0] = 1.0
            stack = stack / totals  # normalise to 1
            cum = np.zeros(T_ref)
            for m, label, color in zip(stack, labels, colors):
                top = cum + np.nan_to_num(m)
                ax.fill_between(trials, cum, top, color=color, alpha=0.7, label=label, lw=0)
                cum = top
            ax.legend(fontsize=FS_LEGEND, loc='upper right')
            _style(ax, ylabel, title, show_xlabel=show_xlabel)

        n_rows = 3 if any_gen else 2
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(14, 3.0 * n_rows), sharex=True, gridspec_kw={'hspace': 0.35}
        )
        for ax in axes if hasattr(axes, '__iter__') else [axes]:
            ax.tick_params(labelbottom=True)

        _plot_stacked(
            axes[0],
            [m0_mean, rot_mean, tra_mean],
            ['P(M$_0$)', 'P(Rotation)', 'P(Translation)'],
            ['#AAAAAA', '#4B0082', '#E6550D'],
            'P(state)',
            'State probabilities',
        )

        if any_gen:
            tr_self_mean = np.nanmean(all_train_self, axis=0)
            tr_ot_mean = np.nanmean(all_train_ot, axis=0)
            tr_og_mean = np.nanmean(all_train_og, axis=0)
            ge_self_mean = np.nanmean(all_gen_self, axis=0)
            ge_ot_mean = np.nanmean(all_gen_ot, axis=0)
            ge_og_mean = np.nanmean(all_gen_og, axis=0)

            dir_labels = ['Self', 'Other (train)', 'Other (gen)']
            dir_colors = ['#2ca02c', '#ff7f0e', '#9467bd']

            _plot_stacked(
                axes[1],
                [tr_self_mean, tr_ot_mean, tr_og_mean],
                dir_labels,
                dir_colors,
                'Dirichlet weight',
                'Dirichlet – training targets',
            )
            _plot_stacked(
                axes[2],
                [ge_self_mean, ge_ot_mean, ge_og_mean],
                dir_labels,
                dir_colors,
                'Dirichlet weight',
                'Dirichlet – generalisation target',
                show_xlabel=True,
            )
        else:
            self_mean = np.nanmean(all_self_w, axis=0)
            other_mean = np.nanmean(all_other_w, axis=0)
            _plot_stacked(
                axes[1],
                [self_mean, other_mean],
                ['Self', 'Other'],
                ['#2ca02c', '#ff7f0e'],
                'Dirichlet weight',
                'Dirichlet weights',
                show_xlabel=True,
            )

        fig.suptitle(f'{datasetLabel}{suffix} (N={N - skipped})', fontsize=FS_TITLE + 2, y=1.01)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{datasetLabel}{suffix}_state_dirichlet_ts.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        if debug:
            print(f"  Saved: {save_path}")

    print(f"State/Dirichlet time series saved to {save_dir}")


def plot_wildcard_state_dirichlet(csv_path, model_path, save_dir, max_epoch=60, debug=False):
    """Wildcard state probs + Dirichlet with movement-epoch x-axis matching F2.

    Builds the same baseline / rotation / washout epoch layout used by
    plot_wildcard_figure so the two figures can be stacked vertically with
    x-axes aligned.
    """
    import json as _json
    from scipy.special import logsumexp as sp_lse

    os.makedirs(save_dir, exist_ok=True)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8
    N_BL = 5
    N_WO = 5

    def _parse(s):
        return _json.loads(s.replace('nan', 'null'))

    def _phase_bounds(rots):
        n = len(rots)
        first = next(i for i in range(n) if rots[i] is not None and rots[i] != 0)
        last = max(i for i in range(n) if rots[i] is not None and rots[i] != 0)
        return first, last

    arr = np.load(model_path, allow_pickle=True)
    bht = arr[0]
    df_wide = pd.read_csv(csv_path)

    # Align all participants to the same baseline, rotation and washout epoch boundaries.

    pp_phase_info = {}
    for idx in range(len(df_wide)):
        row = df_wide.iloc[idx]
        pp = int(row['participantNum'])
        rots = _parse(row['rotation'])
        first_rot, last_rot = _phase_bounds(rots)
        n_rot = last_rot - first_rot + 1
        bl_s = max(0, first_rot - N_BL)
        n_bl_actual = first_rot - bl_s
        wo_s = last_rot + 1
        n_wo_actual = min(N_WO, len(rots) - wo_s)
        pp_phase_info[pp] = {
            'first_rot': first_rot,
            'last_rot': last_rot,
            'bl_start': bl_s,
            'n_bl': n_bl_actual,
            'wo_start': wo_s,
            'n_wo': n_wo_actual,
            'n_rot': n_rot,
        }
    max_rot_len = min(max(info['n_rot'] for info in pp_phase_info.values()), max_epoch)
    n_bl = N_BL
    rot_start = n_bl + 1
    wo_start = rot_start + max_rot_len
    total_epochs = wo_start + N_WO

    all_m0 = []
    all_rot_p = []
    all_trans = []
    all_self_w = []
    all_other_w = []

    for pIdx in range(len(bht.participantNums)):
        pp = bht.participantNums[pIdx]
        ps = bht.predState.get(pp)
        if ps is None or pp not in pp_phase_info:
            continue
        info = pp_phase_info[pp]

        p_m0, p_rot, p_trans = _compute_state_probs(ps)
        destTargets_pp = getattr(bht, 'destTargets', {}).get(pp, [])
        destTA = np.array([x for x in destTargets_pp if not np.isnan(x)], dtype=np.float64)
        sw, ow = _compute_dirichlet_self_vs_other(ps, destTA)

        T_model = len(p_m0)

        ep_m0 = np.full(total_epochs, np.nan)
        ep_rot = np.full(total_epochs, np.nan)
        ep_trans = np.full(total_epochs, np.nan)
        ep_sw = np.full(total_epochs, np.nan)
        ep_ow = np.full(total_epochs, np.nan)

        # Baseline: last N_BL trials before rotation onset
        bl_model_start = info['first_rot'] - info['n_bl']
        for i in range(min(info['n_bl'], N_BL)):
            t_model = bl_model_start + i
            ep_idx = (N_BL - info['n_bl']) + i  # right-align to fill epochs 0..n_bl-1
            if 0 <= t_model < T_model and 0 <= ep_idx < total_epochs:
                ep_m0[ep_idx] = p_m0[t_model]
                ep_rot[ep_idx] = p_rot[t_model]
                ep_trans[ep_idx] = p_trans[t_model]
                ep_sw[ep_idx] = sw[t_model]
                ep_ow[ep_idx] = ow[t_model]

        # Rotation: first max_rot_len trials of rotation phase
        for i in range(min(info['n_rot'], max_rot_len)):
            t_model = info['first_rot'] + i
            ep_idx = rot_start + i - 1  # epoch index (0-based array)
            if 0 <= t_model < T_model and 0 <= ep_idx < total_epochs:
                ep_m0[ep_idx] = p_m0[t_model]
                ep_rot[ep_idx] = p_rot[t_model]
                ep_trans[ep_idx] = p_trans[t_model]
                ep_sw[ep_idx] = sw[t_model]
                ep_ow[ep_idx] = ow[t_model]

        # Washout: first N_WO trials after rotation
        for i in range(min(info['n_wo'], N_WO)):
            t_model = info['wo_start'] + i
            ep_idx = wo_start + i - 1
            if 0 <= t_model < T_model and 0 <= ep_idx < total_epochs:
                ep_m0[ep_idx] = p_m0[t_model]
                ep_rot[ep_idx] = p_rot[t_model]
                ep_trans[ep_idx] = p_trans[t_model]
                ep_sw[ep_idx] = sw[t_model]
                ep_ow[ep_idx] = ow[t_model]

        all_m0.append(ep_m0)
        all_rot_p.append(ep_rot)
        all_trans.append(ep_trans)
        all_self_w.append(ep_sw)
        all_other_w.append(ep_ow)

    N_pp = len(all_m0)
    m0_mean = np.nanmean(all_m0, axis=0)
    rot_mean = np.nanmean(all_rot_p, axis=0)
    tra_mean = np.nanmean(all_trans, axis=0)
    self_mean = np.nanmean(all_self_w, axis=0)
    other_mean = np.nanmean(all_other_w, axis=0)

    epochs = np.arange(1, total_epochs + 1)

    fig, (ax_sp, ax_dir) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'hspace': 0.30}
    )
    ax_sp.tick_params(labelbottom=True)
    ax_dir.tick_params(labelbottom=True)

    def _shade(ax):
        ax.axvline(n_bl + 0.5, color='grey', ls=':', lw=1.8, alpha=0.6)
        ax.axvline(wo_start - 0.5, color='grey', ls=':', lw=1.8, alpha=0.6)

    _shade(ax_sp)
    ax_sp.fill_between(epochs, 0, m0_mean, color='#AAAAAA', alpha=0.7, label='P(M$_0$)')
    ax_sp.fill_between(
        epochs, m0_mean, m0_mean + rot_mean, color='#4B0082', alpha=0.7, label='P(Rotation)'
    )
    ax_sp.fill_between(
        epochs, m0_mean + rot_mean, 1.0, color='#E6550D', alpha=0.7, label='P(Translation)'
    )
    ax_sp.set_ylim(0, 1)
    ax_sp.set_ylabel('Probability', fontsize=FS_LABEL)
    ax_sp.set_title('State Probabilities', fontsize=FS_TITLE)
    ax_sp.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
    ax_sp.tick_params(labelsize=FS_TICK)
    sns.despine(ax=ax_sp)

    _shade(ax_dir)
    total = self_mean + other_mean
    total[total == 0] = 1.0
    s_norm = self_mean / total
    o_norm = other_mean / total
    ax_dir.fill_between(epochs, 0, s_norm, color='#2ca02c', alpha=0.7, label='Self')
    ax_dir.fill_between(epochs, s_norm, 1.0, color='#ff7f0e', alpha=0.7, label='Other')
    ax_dir.set_ylim(0, 1)
    ax_dir.set_ylabel('Dirichlet weight', fontsize=FS_LABEL)
    ax_dir.set_title('Dirichlet Weights', fontsize=FS_TITLE)
    ax_dir.set_xlabel('Movement epoch', fontsize=FS_LABEL)
    ax_dir.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
    ax_dir.tick_params(labelsize=FS_TICK)
    ax_dir.set_xlim(0.5, total_epochs + 0.5)
    sns.despine(ax=ax_dir)

    fig.suptitle(f'Wildcard (N={N_pp})', fontsize=FS_TITLE + 2, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'Wildcard_state_dirichlet_ts.svg')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if debug:
        print(f"  Saved: {save_path}")


def plot_ding_companion(fits, save_dir, debug=False, ding_aggregate=False, exclude_washout=False):
    """Per-participant: state probs + Dirichlet weights, matching heatmap width."""
    os.makedirs(save_dir, exist_ok=True)
    obj_list = extract_fit_shells(fits)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8

    if ding_aggregate:
        _items = [(gk, _merge_obj_group(gk, gos)) for gk, gos in _aggregate_obj_groups(obj_list)]
    else:
        _items = [(getattr(o, 'datasetName', ''), o) for o in obj_list]

    for _gk, obj in _items:
        recon_fn = (
            (
                lambda _obj, _pIdx: reconstruct_shared_participant_data(
                    _obj, _pIdx, exclude_washout=exclude_washout
                )
            )
            if _is_shared_fit(obj)
            else (
                lambda _obj, _pIdx: reconstruct_participant_data(
                    _obj, _pIdx, exclude_washout=exclude_washout
                )
            )
        )
        datasetName = _gk if _gk else getattr(obj, 'datasetName', 'dataset')

        for pIdx in range(len(obj.participantNums)):
            pp = obj.participantNums[pIdx]
            (
                params,
                hasFeedback,
                trials,
                isRotation,
                compMags,
                targets,
                scale_S0,
                nu_S0,
                destTargArr,
                nDest,
                trialStatus,
                phases,
                allAims,
            ) = recon_fn(obj, pIdx)

            ps = obj.predState.get(pp)
            if ps is None:
                continue

            p_m0, p_rot, p_trans = _compute_state_probs(ps)
            # Truncate to predState length (excludes washout in merged objects)
            T_ps = len(p_m0)
            if len(trials) > T_ps:
                trials = trials[:T_ps]
                phases = phases[:T_ps]
                compMags = compMags[:T_ps]
                trialStatus = trialStatus[:T_ps]
                allAims = allAims[:T_ps]
            baseline_trials = int(np.sum(phases == 'baseline'))
            shift = baseline_trials
            numTrials = len(trials)
            trials_shifted = trials - shift
            x_lo = trials_shifted[0] - 0.5
            x_hi = trials_shifted[-1] + 0.5
            flip_sign = _pp_flip_sign(compMags, phases)

            gen_set = getattr(obj, 'genTargets', {}).get(pp, set())
            has_gen = len(gen_set) > 0

            if has_gen:
                dir_data = _compute_dirichlet_by_target_type(ps, destTargArr, gen_set)
            else:
                self_w, other_w = _compute_dirichlet_self_vs_other(ps, destTargArr)

            n_rows = 3 if has_gen else 2
            fig, axes = plt.subplots(
                n_rows, 1, figsize=(14, 3 * n_rows), sharex=True, constrained_layout=True
            )
            for _ax in axes if hasattr(axes, '__iter__') else [axes]:
                _ax.tick_params(labelbottom=True)
            fig.suptitle(f'{datasetName} | PP {pp} — State & Source Weights', fontsize=FS_TITLE + 1)
            x = trials_shifted

            ax = axes[0]
            ax.fill_between(x, 0, p_m0, color='#AAAAAA', alpha=0.7, label='P(M$_0$)')
            ax.fill_between(x, p_m0, p_m0 + p_rot, color='#4B0082', alpha=0.7, label='P(Rotation)')
            ax.fill_between(
                x, p_m0 + p_rot, 1.0, color='#E6550D', alpha=0.7, label='P(Translation)'
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', fontsize=FS_LABEL)
            ax.set_title('State Probabilities', fontsize=FS_TITLE)
            ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
            ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
            ax.tick_params(labelsize=FS_TICK)
            sns.despine(ax=ax)

            if has_gen:
                ax2 = axes[1]
                train_mask = dir_data['is_train']
                valid_train = train_mask & np.isfinite(dir_data['self_w'])
                xt = x[valid_train]
                ax2.plot(
                    xt,
                    dir_data['self_w'][valid_train],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    label='Self',
                )
                ax2.plot(
                    xt,
                    dir_data['train_other_w'][valid_train],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other training',
                )
                ax2.plot(
                    xt,
                    dir_data['gen_other_w'][valid_train],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    ls=':',
                    label='Gen target',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Training Targets', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

                ax3 = axes[2]
                gen_mask = dir_data['is_gen']
                valid_gen = gen_mask & np.isfinite(dir_data['self_w'])
                xg = x[valid_gen]
                ax3.plot(
                    xg,
                    dir_data['self_w'][valid_gen],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    label='Self (gen)',
                )
                ax3.plot(
                    xg,
                    dir_data['train_other_w'][valid_gen],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Training targets',
                )
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_ylabel('Weight', fontsize=FS_LABEL)
                ax3.set_title('Dirichlet Weights — Generalisation Target', fontsize=FS_TITLE)
                ax3.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax3.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax3.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax3)
            else:
                ax2 = axes[1]
                valid = np.isfinite(self_w)
                ax2.plot(x[valid], self_w[valid], color='#4B0082', lw=1.5, alpha=0.8, label='Self')
                ax2.plot(
                    x[valid],
                    other_w[valid],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Self vs Other', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

            axes[-1].set_xlabel('Trial (relative to rotation onset)', fontsize=FS_LABEL)
            axes[-1].set_xlim(x_lo, x_hi)

            save_path = os.path.join(save_dir, f'{datasetName}_pp{pp}_companion.png')
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            if debug:
                print(f"  Saved {save_path}")


def plot_savings_companion(bht, df, save_path=None):
    """Combined figure: BCD timecourse + state probs + rot/probe + gain/diff.

    Layout (4 rows x 2 cols):
      Row 0 (span): Full experiment timecourse
      Row 1 (span): Group-mean state probabilities
      Row 2 left:   Rotation 1 vs 2      Row 2 right: Probe 1 vs 2
      Row 3 left:   State probs block1v2  Row 3 right: Kalman gain block1v2
    """
    from scipy.special import logsumexp as sp_lse

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8
    FS_STAGE = 7

    pdf = _build_savings_plot_df(bht, df)

    # Exclude probe trials in stages 3 and 6 from feedback cycles.
    is_lp = (pdf['stage'].isin([3, 6])) & (pdf['fbi'] == 0)
    new_cycle = np.empty(len(pdf), dtype=int)
    for pp in pdf['pp'].unique():
        pp_rows = np.where(pdf['pp'].values == pp)[0]
        counter = 0
        for row in pp_rows:
            new_cycle[row] = counter // 4
            if not is_lp.iloc[row]:
                counter += 1
    pdf['cycle'] = new_cycle

    # Exclude the very first probe trial in Learning 1 (stage 3)
    for pp in pdf['pp'].unique():
        s3_probes = pdf[(pdf['pp'] == pp) & (pdf['stage'] == 3) & (pdf['fbi'] == 0)]
        if len(s3_probes) > 0:
            first_probe_idx = s3_probes.index[0]
            pdf.loc[first_probe_idx, 'hand_theta'] = np.nan

    pdf['probe_cycle'] = np.nan
    for pp in pdf['pp'].unique():
        for stg in [3, 6]:
            mask = (pdf['pp'] == pp) & (pdf['stage'] == stg) & (pdf['fbi'] == 0)
            idx = pdf.index[mask]
            if len(idx) == 0:
                continue
            fb_cycles = pdf.loc[idx, 'cycle'].values
            n = len(idx)
            for i in range(n):
                group_start = (i // 4) * 4
                group_end = min(group_start + 4, n)
                group_mean = fb_cycles[group_start:group_end].mean()
                pdf.loc[idx[i], 'probe_cycle'] = group_mean

    sp_rows = []
    for pp in bht.participantNums:
        ps = bht.predState.get(pp)
        if ps is None:
            continue
        p_m0, p_rot, p_trans = _compute_state_probs(ps)
        pDat = df[df['participantNum'] == pp].sort_values('TN').reset_index(drop=True)
        ccw = pDat['CCW'].iloc[0]
        T = len(pDat)

        # Kalman gain: run-weighted effective gain
        gains = np.full(T, np.nan)
        for t in range(T):
            ti = int(ps['predTgtIdx'][t])
            if ti < 0:
                continue
            rl = int(ps['runLimits'][t])
            s_val = ps['predSVals'][t, ti]
            obs_var = ps['predObsVar'][t]
            logR = ps['preLogR'][t, :rl].copy()
            valid_r = logR > -500
            if not np.any(valid_r):
                continue
            logR[~valid_r] = -1e30
            run_p = np.exp(logR - sp_lse(logR))
            eff_gain = 0.0
            for k in range(rl):
                if run_p[k] < 1e-15:
                    continue
                c_val = ps['predCPred'][t, k, ti]
                cs = c_val * s_val
                post_var = s_val * (1.0 - cs) if cs < 1.0 else max(s_val * (1.0 - cs), 1e-6)
                g_k = post_var / (post_var + obs_var) if (post_var + obs_var) > 0 else 0.0
                eff_gain += run_p[k] * max(0.0, min(1.0, g_k))
            gains[t] = eff_gain

        for t in range(T):
            row = pDat.iloc[t]
            tn = int(row['TN'])
            stage = int(row['stage'])
            fbi = int(row['fbi'])
            sp_rows.append(
                {
                    'pp': pp,
                    'stage': stage,
                    'fbi': fbi,
                    'p_m0': p_m0[t],
                    'p_rot': p_rot[t],
                    'p_trans': p_trans[t],
                    'gain': gains[t],
                }
            )
    sp_df = pd.DataFrame(sp_rows)

    is_lp_sp = (sp_df['stage'].isin([3, 6])) & (sp_df['fbi'] == 0)
    new_cycle_sp = np.empty(len(sp_df), dtype=int)
    for pp in sp_df['pp'].unique():
        pp_rows = np.where(sp_df['pp'].values == pp)[0]
        counter = 0
        for row in pp_rows:
            new_cycle_sp[row] = counter // 4
            if not is_lp_sp.iloc[row]:
                counter += 1
    sp_df['cycle'] = new_cycle_sp

    stage_bounds = {
        1: (0, 10),
        2: (10, 20),
        3: (20, 60),
        4: (60, 70),
        5: (70, 110),
        6: (110, 150),
        7: (150, 160),
    }
    labels = {
        1: 'No FB\nBase',
        2: 'FB\nBase',
        3: 'Learning 1',
        4: 'No FB\nAE 1',
        5: 'FB Washout',
        6: 'Learning 2',
        7: 'No FB\nAE 2',
    }

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.40, wspace=0.30)

    def _shade_and_label(ax, y_frac=1.02):
        for st in [1, 4, 7]:
            lo, hi = stage_bounds[st]
            ax.axvspan(lo, hi, color='lightgray', alpha=0.3, zorder=0)
        for st, (lo, hi) in stage_bounds.items():
            ax.text(
                (lo + hi) / 2,
                y_frac,
                labels[st],
                ha='center',
                va='bottom',
                fontsize=FS_STAGE,
                color='gray',
                transform=ax.get_xaxis_transform(),
            )
        for boundary in [10, 20, 60, 70, 110, 150]:
            ax.axvline(boundary, color='gray', lw=0.5, ls='--', alpha=0.5)

    def _style_ax(
        ax, title, xlabel, ylabel, legend_loc='upper right', legend_ncol=1, xlim=(-1, 161)
    ):
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(*xlim)
        sns.despine(ax=ax)

    N_pp = pdf['pp'].nunique()

    ax_b = fig.add_subplot(gs[0, :])
    _shade_and_label(ax_b)

    for stg in [3, 6]:
        seg = pdf[(pdf['fbi'] == 1) & (pdf['stage'] == stg)]
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#C8A2C8', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=40, color='#7B3F8D', zorder=3, alpha=0.7, edgecolors='none')

    for stg in [3, 6]:
        seg = pdf[(pdf['fbi'] == 0) & (pdf['stage'] == stg)].copy()
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta', cycle_col='probe_cycle')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#E8888A', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=40, color='#D44', zorder=3, alpha=0.7, edgecolors='none')

    for stg in [1, 2, 4, 5, 7]:
        seg = pdf[pdf['stage'] == stg]
        if len(seg) == 0:
            continue
        cyc, m, ci = _cycle_mean_ci(seg, 'hand_theta')
        ax_b.fill_between(cyc, m - ci, m + ci, color='#E8888A', alpha=0.3, zorder=1)
        ax_b.scatter(cyc, m, s=40, color='#D44', zorder=3, alpha=0.7, edgecolors='none')

    cyc, m, _ = _cycle_mean_ci(pdf, 'model_hand')
    ax_b.plot(cyc, m, color='#7B3F8D', lw=2, zorder=4, alpha=0.9)

    red_data = pdf[~((pdf['fbi'] == 1) & (pdf['stage'].isin([3, 6])))]
    cyc, m, _ = _cycle_mean_ci(red_data, 'model_probe')
    ax_b.plot(cyc, m, color='darkred', lw=2, zorder=4, alpha=0.9)

    ax_b.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_b.set_ylim(-5, 48)
    _style_ax(
        ax_b,
        f'Full Experiment Timecourse (N={N_pp})',
        'Cycle Number (4 Movements)',
        'Hand Angle (deg)',
    )
    ax_b.legend(
        handles=[
            Line2D(
                [],
                [],
                marker='o',
                color='#7B3F8D',
                markersize=5,
                ls='None',
                label='Human aim+implicit',
            ),
            Line2D(
                [], [], marker='o', color='#D44', markersize=5, ls='None', label='Human implicit'
            ),
            Line2D([], [], color='#7B3F8D', lw=2, label='Model aim+implicit'),
            Line2D([], [], color='darkred', lw=2, label='Model implicit'),
        ],
        fontsize=FS_LEGEND,
        loc='upper left',
        framealpha=0.7,
    )

    ax_sp = fig.add_subplot(gs[1, :])
    _shade_and_label(ax_sp)

    cyc_stats = (
        sp_df.groupby('cycle')
        .agg(
            p_m0_mean=('p_m0', 'mean'),
            p_rot_mean=('p_rot', 'mean'),
            p_trans_mean=('p_trans', 'mean'),
        )
        .reset_index()
    )
    c = cyc_stats['cycle'].values
    ax_sp.fill_between(c, 0, cyc_stats['p_m0_mean'], color='#AAAAAA', alpha=0.7, label='P(M$_0$)')
    ax_sp.fill_between(
        c,
        cyc_stats['p_m0_mean'],
        cyc_stats['p_m0_mean'] + cyc_stats['p_rot_mean'],
        color='#4B0082',
        alpha=0.7,
        label='P(Rotation)',
    )
    ax_sp.fill_between(
        c,
        cyc_stats['p_m0_mean'] + cyc_stats['p_rot_mean'],
        1.0,
        color='#E6550D',
        alpha=0.7,
        label='P(Translation)',
    )
    ax_sp.set_ylim(0, 1)
    _style_ax(ax_sp, 'Group-Mean State Probabilities', 'Cycle Number (4 Movements)', 'Probability')
    ax_sp.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)

    ax_c = fig.add_subplot(gs[2, 0])
    r1 = pdf[(pdf['stage'] == 3) & (pdf['fbi'] == 1)].copy()
    r1['rel_cycle'] = r1['cycle'] - 20
    r2 = pdf[(pdf['stage'] == 6) & (pdf['fbi'] == 1)].copy()
    r2['rel_cycle'] = r2['cycle'] - 110
    for rdf, color in [(r1, '#C8A2C8'), (r2, '#4B0082')]:
        cyc, m, ci = _cycle_mean_ci(rdf, 'hand_theta', cycle_col='rel_cycle')
        ax_c.fill_between(cyc, m - ci, m + ci, color=color, alpha=0.25)
        ax_c.scatter(cyc, m, s=40, color=color, alpha=0.7, edgecolors='none')
    for rdf, color in [(r1, '#C8A2C8'), (r2, '#4B0082')]:
        cyc, m, _ = _cycle_mean_ci(rdf, 'model_hand', cycle_col='rel_cycle')
        ax_c.plot(cyc, m, color=color, lw=2.5, alpha=0.9)
    ax_c.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_c.set_ylim(-5, 48)
    _style_ax(
        ax_c, 'Rotation 1 vs. 2 (Feedback)', 'Rotation Cycles', 'Hand Angle (deg)', xlim=(-1, 41)
    )
    ax_c.legend(
        handles=[
            Line2D(
                [], [], marker='o', color='#C8A2C8', markersize=5, ls='None', label='Human rot 1'
            ),
            Line2D(
                [], [], marker='o', color='#4B0082', markersize=5, ls='None', label='Human rot 2'
            ),
            Line2D([], [], color='#C8A2C8', lw=2.5, label='Model rot 1'),
            Line2D([], [], color='#4B0082', lw=2.5, label='Model rot 2'),
        ],
        fontsize=FS_LEGEND,
        loc='lower right',
        framealpha=0.7,
    )

    ax_d = fig.add_subplot(gs[2, 1])
    p1 = pdf[(pdf['stage'] == 3) & (pdf['fbi'] == 0)].copy()
    p1['rel_probe_cycle'] = p1['probe_cycle'] - 20
    p2 = pdf[(pdf['stage'] == 6) & (pdf['fbi'] == 0)].copy()
    p2['rel_probe_cycle'] = p2['probe_cycle'] - 110
    for rdf, color in [(p1, '#E8888A'), (p2, '#8B0000')]:
        cyc, m, ci = _cycle_mean_ci(rdf, 'hand_theta', cycle_col='rel_probe_cycle')
        ax_d.fill_between(cyc, m - ci, m + ci, color=color, alpha=0.25)
        ax_d.scatter(cyc, m, s=40, color=color, alpha=0.7, edgecolors='none')
    for rdf, color in [(p1, '#E8888A'), (p2, '#8B0000')]:
        cyc, m, _ = _cycle_mean_ci(rdf, 'model_probe', cycle_col='rel_probe_cycle')
        ax_d.plot(cyc, m, color=color, lw=2.5, alpha=0.9)
    ax_d.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_d.set_ylim(-5, 35)
    _style_ax(ax_d, 'Probe 1 vs. 2', 'Rotation Cycles', 'Hand Angle (deg)', xlim=(-1, 41))
    ax_d.legend(
        handles=[
            Line2D(
                [], [], marker='o', color='#E8888A', markersize=5, ls='None', label='Human probe 1'
            ),
            Line2D(
                [], [], marker='o', color='#8B0000', markersize=5, ls='None', label='Human probe 2'
            ),
            Line2D([], [], color='#E8888A', lw=2.5, label='Model probe 1'),
            Line2D([], [], color='#8B0000', lw=2.5, label='Model probe 2'),
        ],
        fontsize=FS_LEGEND,
        loc='lower right',
        framealpha=0.7,
    )

    ax_diff = fig.add_subplot(gs[3, 0])
    diff_rows = []
    for pp in sp_df['pp'].unique():
        pp_sp = sp_df[sp_df['pp'] == pp]
        for stg, offset in [(3, 20), (6, 110)]:
            seg = pp_sp[pp_sp['stage'] == stg].copy()
            if len(seg) == 0:
                continue
            seg = seg.copy()
            seg['rel_cycle'] = seg['cycle'] - offset
            for _, row in seg.iterrows():
                diff_rows.append(
                    {
                        'pp': pp,
                        'block': 1 if stg == 3 else 2,
                        'rel_cycle': row['rel_cycle'],
                        'p_rot': row['p_rot'],
                        'p_trans': row['p_trans'],
                        'p_m0': row['p_m0'],
                    }
                )
    diff_df = pd.DataFrame(diff_rows)

    for block, color_rot, color_trans, ls in [
        (1, '#C8A2C8', '#F4C2A0', '-'),
        (2, '#4B0082', '#E6550D', '-'),
    ]:
        bdf = diff_df[diff_df['block'] == block]
        for col, color, label in [
            ('p_rot', color_rot, f'P(Rot) block {block}'),
            ('p_trans', color_trans, f'P(Trans) block {block}'),
        ]:
            stats = bdf.groupby('rel_cycle')[col].agg(['mean', 'std', 'count']).reset_index()
            stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
            ax_diff.fill_between(
                stats['rel_cycle'],
                stats['mean'] - stats['ci'],
                stats['mean'] + stats['ci'],
                color=color,
                alpha=0.15,
            )
            ax_diff.plot(
                stats['rel_cycle'], stats['mean'], color=color, lw=2, alpha=0.9, ls=ls, label=label
            )

    ax_diff.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_diff.set_ylim(-0.05, 1.05)
    _style_ax(
        ax_diff, 'State Probs: Learning 1 vs. 2', 'Rotation Cycles', 'Probability', xlim=(-1, 41)
    )
    ax_diff.legend(fontsize=FS_LEGEND - 1, loc='center right', framealpha=0.7, ncol=1)

    ax_gain = fig.add_subplot(gs[3, 1])
    gain_rows = []
    for pp in sp_df['pp'].unique():
        pp_sp = sp_df[sp_df['pp'] == pp]
        for stg, offset in [(3, 20), (6, 110)]:
            seg = pp_sp[pp_sp['stage'] == stg].copy()
            if len(seg) == 0:
                continue
            seg = seg.copy()
            seg['rel_cycle'] = seg['cycle'] - offset
            for _, row in seg.iterrows():
                if np.isfinite(row['gain']):
                    gain_rows.append(
                        {
                            'pp': pp,
                            'block': 1 if stg == 3 else 2,
                            'rel_cycle': row['rel_cycle'],
                            'gain': row['gain'],
                        }
                    )
    gain_df = pd.DataFrame(gain_rows)

    for block, color, label in [(1, '#C8A2C8', 'Block 1'), (2, '#4B0082', 'Block 2')]:
        bdf = gain_df[gain_df['block'] == block]
        stats = bdf.groupby('rel_cycle')['gain'].agg(['mean', 'std', 'count']).reset_index()
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        ax_gain.fill_between(
            stats['rel_cycle'],
            stats['mean'] - stats['ci'],
            stats['mean'] + stats['ci'],
            color=color,
            alpha=0.2,
        )
        ax_gain.plot(stats['rel_cycle'], stats['mean'], color=color, lw=2.5, alpha=0.9, label=label)

    ax_gain.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_gain.set_ylim(-0.05, 1.05)
    _style_ax(
        ax_gain, 'Kalman Gain: Learning 1 vs. 2', 'Rotation Cycles', 'Effective Gain', xlim=(-1, 41)
    )
    ax_gain.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved savings combined figure to {save_path}")
    plt.close(fig)


def plot_savings_individual_companion(fits, save_dir, debug=False):
    """Per-participant savings companion: aim + state probs + Dirichlet.

    Trial-based x-axis (relative to rotation onset) to align with
    individual heatmap plots from generate_bht_plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    obj_list = extract_fit_shells(fits)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8

    for obj in obj_list:
        recon_fn = (
            reconstruct_shared_participant_data
            if _is_shared_fit(obj)
            else reconstruct_participant_data
        )
        datasetName = getattr(obj, 'datasetName', 'dataset')

        for pIdx in range(len(obj.participantNums)):
            pp = obj.participantNums[pIdx]
            (
                params,
                hasFeedback,
                trials,
                isRotation,
                compMags,
                targets,
                scale_S0,
                nu_S0,
                destTargArr,
                nDest,
                trialStatus,
                phases,
                allAims,
            ) = recon_fn(obj, pIdx)

            ps = obj.predState.get(pp)
            if ps is None:
                continue

            baseline_trials = int(np.sum(phases == 'baseline'))
            shift = baseline_trials
            numTrials = len(trials)
            trials_shifted = trials - shift
            x_lo = trials_shifted[0] - 0.5
            x_hi = trials_shifted[-1] + 0.5
            flip_sign = _pp_flip_sign(compMags, phases)

            pDat = obj.df[obj.df['participantNum'] == pp].sort_values('TN').reset_index(drop=True)
            ccw = pDat['CCW'].iloc[0]
            flip = -1.0 if ccw == 1 else 1.0

            p_m0, p_rot, p_trans = _compute_state_probs(ps)

            gen_set = getattr(obj, 'genTargets', {}).get(pp, set())
            has_gen = len(gen_set) > 0

            if has_gen:
                dir_data = _compute_dirichlet_by_target_type(ps, destTargArr, gen_set)
            else:
                self_w, other_w = _compute_dirichlet_self_vs_other(ps, destTargArr)

            model_aim = _get_model_expected_aim(obj, pp)
            implicit = pDat['implicit'].values
            model_hand = (model_aim + implicit) * flip
            human_hand = pDat['hand_theta'].values * flip

            n_rows = 4 if has_gen else 3
            fig, axes = plt.subplots(
                n_rows, 1, figsize=(14, 3 * n_rows), sharex=True, constrained_layout=True
            )
            for _ax in axes if hasattr(axes, '__iter__') else [axes]:
                _ax.tick_params(labelbottom=True)
            fig.suptitle(f'{datasetName} | PP {pp} — Individual Companion', fontsize=FS_TITLE)
            x = trials_shifted

            ax = axes[0]
            valid_h = np.isfinite(human_hand)
            ax.scatter(
                x[valid_h],
                human_hand[valid_h],
                s=12,
                color='#7B3F8D',
                alpha=0.5,
                edgecolors='none',
                zorder=2,
                label='Human',
            )
            ax.plot(x, model_hand, color='#7B3F8D', lw=1.5, alpha=0.9, zorder=3, label='Model')
            ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
            ax.set_ylabel('Hand Angle (deg)', fontsize=FS_LABEL)
            ax.set_title('Aim + Implicit Timecourse', fontsize=FS_TITLE)
            ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
            ax.tick_params(labelsize=FS_TICK)
            sns.despine(ax=ax)

            ax = axes[1]
            ax.fill_between(x, 0, p_m0, color='#AAAAAA', alpha=0.7, label='P(M$_0$)')
            ax.fill_between(x, p_m0, p_m0 + p_rot, color='#4B0082', alpha=0.7, label='P(Rotation)')
            ax.fill_between(
                x, p_m0 + p_rot, 1.0, color='#E6550D', alpha=0.7, label='P(Translation)'
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', fontsize=FS_LABEL)
            ax.set_title('State Probabilities', fontsize=FS_TITLE)
            ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
            ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
            ax.tick_params(labelsize=FS_TICK)
            sns.despine(ax=ax)

            if has_gen:
                ax2 = axes[2]
                train_mask = dir_data['is_train']
                valid_train = train_mask & np.isfinite(dir_data['self_w'])
                xt = x[valid_train]
                ax2.plot(
                    xt,
                    dir_data['self_w'][valid_train],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    label='Self',
                )
                ax2.plot(
                    xt,
                    dir_data['train_other_w'][valid_train],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other training',
                )
                ax2.plot(
                    xt,
                    dir_data['gen_other_w'][valid_train],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    ls=':',
                    label='Gen target',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Training Targets', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

                ax3 = axes[3]
                gen_mask = dir_data['is_gen']
                valid_gen = gen_mask & np.isfinite(dir_data['self_w'])
                xg = x[valid_gen]
                ax3.plot(
                    xg,
                    dir_data['self_w'][valid_gen],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    label='Self (gen)',
                )
                ax3.plot(
                    xg,
                    dir_data['train_other_w'][valid_gen],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Training targets',
                )
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_ylabel('Weight', fontsize=FS_LABEL)
                ax3.set_title('Dirichlet Weights — Generalisation Target', fontsize=FS_TITLE)
                ax3.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax3.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax3.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax3)
            else:
                ax2 = axes[2]
                valid = np.isfinite(self_w)
                ax2.plot(x[valid], self_w[valid], color='#4B0082', lw=1.5, alpha=0.8, label='Self')
                ax2.plot(
                    x[valid],
                    other_w[valid],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Self vs Other', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

            axes[-1].set_xlabel('Trial (relative to rotation onset)', fontsize=FS_LABEL)
            axes[-1].set_xlim(x_lo, x_hi)

            save_path = os.path.join(save_dir, f'{datasetName}_pp{pp}_companion.svg')
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            if debug:
                print(f"  Saved {save_path}")


def plot_wildcard_individual_companion(fits, save_dir, debug=False):
    """Per-participant wildcard companion: aim timecourse + state probs + Dirichlet."""
    os.makedirs(save_dir, exist_ok=True)
    obj_list = extract_fit_shells(fits)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8

    for obj in obj_list:
        recon_fn = (
            reconstruct_shared_participant_data
            if _is_shared_fit(obj)
            else reconstruct_participant_data
        )
        datasetName = getattr(obj, 'datasetName', 'dataset')

        for pIdx in range(len(obj.participantNums)):
            pp = obj.participantNums[pIdx]
            (
                params,
                hasFeedback,
                trials,
                isRotation,
                compMags,
                targets,
                scale_S0,
                nu_S0,
                destTargArr,
                nDest,
                trialStatus,
                phases,
                allAims,
            ) = recon_fn(obj, pIdx)

            ps = obj.predState.get(pp)
            if ps is None:
                continue

            baseline_trials = int(np.sum(phases == 'baseline'))
            shift = baseline_trials
            numTrials = len(trials)
            trials_shifted = trials - shift
            x_lo = trials_shifted[0] - 0.5
            x_hi = trials_shifted[-1] + 0.5
            flip_sign = _pp_flip_sign(compMags, phases)

            p_m0, p_rot, p_trans = _compute_state_probs(ps)

            gen_set = getattr(obj, 'genTargets', {}).get(pp, set())
            has_gen = len(gen_set) > 0

            if has_gen:
                dir_data = _compute_dirichlet_by_target_type(ps, destTargArr, gen_set)
            else:
                self_w, other_w = _compute_dirichlet_self_vs_other(ps, destTargArr)

            model_aim = _get_model_expected_aim(obj, pp)
            flip = -1.0 if flip_sign else 1.0
            model_aim_f = model_aim * flip
            allAims_f = allAims * flip
            compMags_f = compMags * flip

            n_rows = 4 if has_gen else 3
            fig, axes = plt.subplots(
                n_rows, 1, figsize=(14, 3 * n_rows), sharex=True, constrained_layout=True
            )
            for _ax in axes if hasattr(axes, '__iter__') else [axes]:
                _ax.tick_params(labelbottom=True)
            fig.suptitle(f'{datasetName} | PP {pp} — Individual Companion', fontsize=FS_TITLE)
            x = trials_shifted

            ax = axes[0]
            valid_aim = np.isfinite(allAims_f)
            ax.scatter(
                x[valid_aim],
                allAims_f[valid_aim],
                s=12,
                color='#7B3F8D',
                alpha=0.5,
                edgecolors='none',
                zorder=2,
                label='Human aim',
            )
            ax.plot(x, model_aim_f, color='#7B3F8D', lw=1.5, alpha=0.9, zorder=3, label='Model aim')
            ax.fill_between(
                x,
                0,
                compMags_f,
                color='lightcoral',
                alpha=0.15,
                step='mid',
                label='Rotation',
                zorder=0,
            )
            ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
            ax.set_ylabel('Aim (deg)', fontsize=FS_LABEL)
            ax.set_title('Aim Timecourse', fontsize=FS_TITLE)
            ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
            ax.tick_params(labelsize=FS_TICK)
            sns.despine(ax=ax)

            ax = axes[1]
            ax.fill_between(x, 0, p_m0, color='#AAAAAA', alpha=0.7, label='P(M$_0$)')
            ax.fill_between(x, p_m0, p_m0 + p_rot, color='#4B0082', alpha=0.7, label='P(Rotation)')
            ax.fill_between(
                x, p_m0 + p_rot, 1.0, color='#E6550D', alpha=0.7, label='P(Translation)'
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', fontsize=FS_LABEL)
            ax.set_title('State Probabilities', fontsize=FS_TITLE)
            ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
            ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
            ax.tick_params(labelsize=FS_TICK)
            sns.despine(ax=ax)

            if has_gen:
                ax2 = axes[2]
                train_mask = dir_data['is_train']
                valid_train = train_mask & np.isfinite(dir_data['self_w'])
                xt = x[valid_train]
                ax2.plot(
                    xt,
                    dir_data['self_w'][valid_train],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    label='Self',
                )
                ax2.plot(
                    xt,
                    dir_data['train_other_w'][valid_train],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other training',
                )
                ax2.plot(
                    xt,
                    dir_data['gen_other_w'][valid_train],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    ls=':',
                    label='Gen target',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Training Targets', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

                ax3 = axes[3]
                gen_mask = dir_data['is_gen']
                valid_gen = gen_mask & np.isfinite(dir_data['self_w'])
                xg = x[valid_gen]
                ax3.plot(
                    xg,
                    dir_data['self_w'][valid_gen],
                    color='#E6550D',
                    lw=1.5,
                    alpha=0.8,
                    label='Self (gen)',
                )
                ax3.plot(
                    xg,
                    dir_data['train_other_w'][valid_gen],
                    color='#4B0082',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Training targets',
                )
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_ylabel('Weight', fontsize=FS_LABEL)
                ax3.set_title('Dirichlet Weights — Generalisation Target', fontsize=FS_TITLE)
                ax3.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax3.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, framealpha=0.7)
                ax3.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax3)
            else:
                ax2 = axes[2]
                valid = np.isfinite(self_w)
                ax2.plot(x[valid], self_w[valid], color='#4B0082', lw=1.5, alpha=0.8, label='Self')
                ax2.plot(
                    x[valid],
                    other_w[valid],
                    color='#7B68EE',
                    lw=1.5,
                    alpha=0.8,
                    ls='--',
                    label='Other',
                )
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel('Weight', fontsize=FS_LABEL)
                ax2.set_title('Dirichlet Weights — Self vs Other', fontsize=FS_TITLE)
                ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
                ax2.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.7)
                ax2.tick_params(labelsize=FS_TICK)
                sns.despine(ax=ax2)

            axes[-1].set_xlabel('Trial (relative to rotation onset)', fontsize=FS_LABEL)
            axes[-1].set_xlim(x_lo, x_hi)

            save_path = os.path.join(save_dir, f'{datasetName}_pp{pp}_companion.svg')
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            if debug:
                print(f"  Saved {save_path}")


def plot_changepoint_aligned_bht(
    bht_fits,
    ssm_fits,
    changepoints_csv,
    save_path=None,
    debug=False,
    rel_range=(-20, 20),
    extended_range=(-30, 31),
):
    """
    Plot human |aim| aligned to human changepoint, with BHT predictive
    distribution as 2D contour overlay. One subplot per rotation condition.

    Args:
        bht_fits: array of BHT FitShell objects (one per rotation)
        ssm_fits: array of SSM FitShell objects (one per rotation, for human aims)
        changepoints_csv: path to CSV with human changepoints
        save_path: if given, save figure
        debug: verbose output
        rel_range: (min, max) relative trials to plot
        extended_range: (min, max+1) extended range for padding
    """
    from scipy.stats import t as t_dist
    from scipy.special import logsumexp as sp_logsumexp
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    import matplotlib.colors as mcolors
    from matplotlib import cm

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8
    baseline_length = 40
    washout_length = 40

    def normalize_angle(angle):
        if np.isscalar(angle):
            if np.isnan(angle):
                return np.nan
            angle = angle - 360 * np.round(angle / 360)
            return ((angle + 180) % 360) - 180
        else:
            normalized = np.full_like(angle, np.nan)
            mask = ~np.isnan(angle)
            unwrapped = angle[mask] - 360 * np.round(angle[mask] / 360)
            normalized[mask] = ((unwrapped + 180) % 360) - 180
            return normalized

    def darken_color(hex_color, factor=0.7):
        rgb = mcolors.hex2color(hex_color)
        return mcolors.to_hex(tuple(c * factor for c in rgb))

    cp_df = pd.read_csv(changepoints_csv)
    cp_df = cp_df[cp_df['dataset'] == 'CGVanilla']

    rotation_colors_dict = {
        90: '#44AA99',
        60: '#88CCEE',
        45: '#FF9825',
        30: '#CC6677',
        15: '#AA4499',
    }

    rel_positions = list(range(extended_range[0], extended_range[1]))
    plot_min, plot_max = rel_range
    magnitudes = np.arange(181)
    angles = np.arange(-180, 181).astype(float)
    # nu (Student-t df) extracted per participant from fits below

    if isinstance(bht_fits, np.ndarray) and bht_fits.ndim == 0:
        bht_fits = [bht_fits.item()]
    elif isinstance(bht_fits, np.ndarray):
        bht_fits = list(bht_fits)
    if isinstance(ssm_fits, np.ndarray) and ssm_fits.ndim == 0:
        ssm_fits = [ssm_fits.item()]
    elif isinstance(ssm_fits, np.ndarray):
        ssm_fits = list(ssm_fits)

    n_rots = len(bht_fits)
    n_cols = min(3, n_rots)
    n_rows = (n_rots + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axs_flat = [axs]
    elif n_rows == 1 or n_cols == 1:
        axs_flat = list(axs.flatten()) if hasattr(axs, 'flatten') else [axs]
    else:
        axs_flat = axs.flatten().tolist()

    for rotIdx in range(n_rots):
        bht = bht_fits[rotIdx]
        _apply_fit_model_config(bht, debug=debug)
        ssm = ssm_fits[rotIdx]
        rotation = abs(bht.conVal)
        ax = axs_flat[rotIdx]

        rot_color = rotation_colors_dict.get(rotation, 'black')

        if not hasattr(bht, 'predState') or bht.predState is None:
            ax.set_visible(False)
            continue

        bht_ppnums = np.array(bht.participantNums)
        n_pp_ssm = len(ssm.allAims)
        n_trials = len(ssm.allAims[0])
        bl = baseline_length
        min_t = bl
        max_t = n_trials - (washout_length + min(40, bl))

        rot_cps = cp_df[cp_df['rotation'] == rotation]

        human_series_list = []
        average_mag_pdf = np.zeros((len(rel_positions), 181))
        n_valid = np.zeros(len(rel_positions))
        valid_parts = 0

        for pId in range(n_pp_ssm):
            human_aims = np.array(ssm.allAims[pId]).ravel() % 360.0
            human_aims_signed = normalize_angle(human_aims)
            aim = np.abs(human_aims_signed)

            part_cps = rot_cps[rot_cps['participant_id'] == pId]
            if part_cps.empty:
                continue
            best_t = int(part_cps['changepoint_position'].min())
            if best_t < min_t or best_t > max_t:
                continue

            human_series = np.full(len(rel_positions), np.nan)
            for ii, rel in enumerate(rel_positions):
                trial_idx = best_t + rel
                if 0 <= trial_idx < n_trials:
                    human_series[ii] = aim[trial_idx]
            human_series_list.append(human_series)

            # Match to BHT predState
            pp_key = bht_ppnums[pId] if pId < len(bht_ppnums) else None
            if pp_key is None or pp_key not in bht.predState:
                continue

            ps = bht.predState[pp_key]
            comp_logp = np.array(ps['predCompLogProb'])
            comp_mean = np.array(ps['predCompMean'])
            comp_scale = np.array(ps['predCompScale'])
            n_comp_arr = np.array(ps['predNComp']).astype(int)
            n_model = len(n_comp_arr)

            _, pp_nu, _ = _extract_params(bht.xs[pId])

            for ii, rel in enumerate(rel_positions):
                trial_idx = best_t + rel
                if trial_idx < 0 or trial_idx >= min(n_trials, n_model):
                    continue

                nc = n_comp_arr[trial_idx]
                if nc <= 0:
                    continue

                lp = comp_logp[trial_idx, :nc]
                mn = comp_mean[trial_idx, :nc]
                sc = comp_scale[trial_idx, :nc]

                valid_comp = np.isfinite(lp) & (sc > 0)
                if not np.any(valid_comp):
                    continue

                lp_v = lp[valid_comp]
                mn_v = mn[valid_comp]
                sc_v = sc[valid_comp]
                wts = np.exp(lp_v - sp_logsumexp(lp_v))

                # Vectorized: (361, nc) @ (nc,) -> (361,)
                pdf_matrix = t_dist.pdf(
                    angles[:, None], df=pp_nu, loc=mn_v[None, :], scale=sc_v[None, :]
                )
                pdf_vals = pdf_matrix @ wts

                # Fold to magnitude [0, 180]
                mag_pdf = np.zeros(181)
                mag_pdf[0] = pdf_vals[180]
                for m in range(1, 180):
                    mag_pdf[m] = pdf_vals[180 - m] + pdf_vals[180 + m]
                mag_pdf[180] = pdf_vals[0]

                average_mag_pdf[ii] += mag_pdf
                n_valid[ii] += 1

            valid_parts += 1

        if debug:
            print(f'Rot {rotation}: {valid_parts} valid participants')

        if valid_parts == 0:
            ax.text(
                0.5,
                0.5,
                'No data',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=FS_LABEL,
            )
            continue

        for ii in range(len(rel_positions)):
            if n_valid[ii] > 0:
                average_mag_pdf[ii] /= n_valid[ii]

        # Forward/backward fill within plot range
        idx_min = rel_positions.index(plot_min)
        idx_max = rel_positions.index(plot_max - 1)

        last_valid = None
        for ii in range(idx_min, idx_max + 1):
            if n_valid[ii] > 0:
                last_valid = average_mag_pdf[ii].copy()
            elif last_valid is not None:
                average_mag_pdf[ii] = last_valid
                n_valid[ii] = 1
        last_valid = None
        for ii in range(idx_max, idx_min - 1, -1):
            if n_valid[ii] > 0:
                last_valid = average_mag_pdf[ii].copy()
            elif last_valid is not None:
                average_mag_pdf[ii] = last_valid
                n_valid[ii] = 1

        rel_arr = np.array(rel_positions)
        Z = average_mag_pdf.T

        contour_norm = LogNorm(vmin=1e-3, vmax=0.05)
        levels = np.logspace(np.log10(1e-3), np.log10(0.05), 30)

        darkened_hex = darken_color(rot_color, factor=0.8)
        darkened_rgb = mcolors.hex2color(darkened_hex)
        cdict = {
            'red': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[0], darkened_rgb[0])],
            'green': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[1], darkened_rgb[1])],
            'blue': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[2], darkened_rgb[2])],
            'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        }
        custom_cmap = LinearSegmentedColormap(f'custom_{rotation}', cdict)

        ax.contourf(
            rel_arr,
            magnitudes,
            Z,
            levels=levels,
            cmap=custom_cmap,
            norm=contour_norm,
            zorder=3,
            extend='max',
        )

        ax.axhline(y=0, color='grey', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.axhline(y=rotation, color=rot_color, alpha=0.5, linestyle='--', linewidth=2)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        human_arr = np.stack(human_series_list)
        mean_human = np.nanmean(human_arr, axis=0)

        plot_slice = [i for i, r in enumerate(rel_positions) if plot_min <= r < plot_max]

        ax.scatter(
            [rel_positions[i] for i in plot_slice],
            [mean_human[i] for i in plot_slice],
            color=rot_color,
            edgecolor='black',
            linewidth=1,
            s=50,
            zorder=6,
            label='Human mean',
        )

        for series in human_series_list:
            valid_mask = ~np.isnan(series)
            xs_j = []
            ys_j = []
            for i in plot_slice:
                if valid_mask[i]:
                    xs_j.append(rel_positions[i] + np.random.uniform(-0.15, 0.15))
                    ys_j.append(series[i])
            if xs_j:
                ax.scatter(
                    xs_j,
                    ys_j,
                    color=rot_color,
                    alpha=0.1,
                    edgecolor='black',
                    linewidth=0.3,
                    s=15,
                    zorder=5,
                )

        ax.set_title(f'{rotation}\u00b0', fontsize=FS_TITLE)
        ax.set_xlabel('Trial rel. to changepoint', fontsize=FS_LABEL)
        ax.set_ylabel('|Aim| (\u00b0)', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(plot_min - 0.5, plot_max - 0.5)
        ax.set_ylim(-5, 181)

        yticks = sorted(set([0, rotation, 180]))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(t)}\u00b0' for t in yticks])

    for i in range(n_rots, len(axs_flat)):
        axs_flat[i].set_visible(False)

    sns.despine()
    fig.suptitle(
        'BHT predictive aligned to human changepoint (CG)', fontsize=FS_TITLE + 2, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if debug:
            print(f'Saved to {save_path}')
    plt.close(fig)


def plot_changepoint_aligned_bht_combined(
    bht_fits,
    ssm_fits,
    changepoints_csv,
    save_path=None,
    debug=False,
    rel_range=(-20, 20),
    extended_range=(-30, 31),
):
    """Overlay all CG rotation conditions on one axis.

    Human mean |aim| traces are shown as colored lines/points and the
    corresponding BHT predictive distributions are overlaid as translucent
    filled contours in matching colors.
    """
    from scipy.stats import t as t_dist
    from scipy.special import logsumexp as sp_logsumexp
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    import matplotlib.colors as mcolors

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8
    baseline_length = 40
    washout_length = 40

    def normalize_angle(angle):
        if np.isscalar(angle):
            if np.isnan(angle):
                return np.nan
            angle = angle - 360 * np.round(angle / 360)
            return ((angle + 180) % 360) - 180
        normalized = np.full_like(angle, np.nan)
        mask = ~np.isnan(angle)
        unwrapped = angle[mask] - 360 * np.round(angle[mask] / 360)
        normalized[mask] = ((unwrapped + 180) % 360) - 180
        return normalized

    def darken_color(hex_color, factor=0.8):
        rgb = mcolors.hex2color(hex_color)
        return mcolors.to_hex(tuple(c * factor for c in rgb))

    cp_df = pd.read_csv(changepoints_csv)
    cp_df = cp_df[cp_df['dataset'] == 'CGVanilla']

    rotation_colors_dict = {
        90: '#44AA99',
        60: '#88CCEE',
        45: '#FF9825',
        30: '#CC6677',
        15: '#AA4499',
    }

    rel_positions = list(range(extended_range[0], extended_range[1]))
    plot_min, plot_max = rel_range
    magnitudes = np.arange(181)
    angles = np.arange(-180, 181).astype(float)

    if isinstance(bht_fits, np.ndarray) and bht_fits.ndim == 0:
        bht_fits = [bht_fits.item()]
    elif isinstance(bht_fits, np.ndarray):
        bht_fits = list(bht_fits)
    if isinstance(ssm_fits, np.ndarray) and ssm_fits.ndim == 0:
        ssm_fits = [ssm_fits.item()]
    elif isinstance(ssm_fits, np.ndarray):
        ssm_fits = list(ssm_fits)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    legend_handles = []
    rotations_found = []

    for rotIdx in range(len(bht_fits)):
        bht = bht_fits[rotIdx]
        _apply_fit_model_config(bht, debug=debug)
        ssm = ssm_fits[rotIdx]
        rotation = abs(bht.conVal)
        rot_color = rotation_colors_dict.get(rotation, 'black')

        if not hasattr(bht, 'predState') or bht.predState is None:
            continue
        if not hasattr(ssm, 'allAims') or len(ssm.allAims) == 0:
            continue

        bht_ppnums = np.array(bht.participantNums)
        n_pp_ssm = len(ssm.allAims)
        n_trials = len(ssm.allAims[0])
        bl = baseline_length
        min_t = bl
        max_t = n_trials - (washout_length + min(40, bl))

        rot_cps = cp_df[cp_df['rotation'] == rotation]
        human_series_list = []
        average_mag_pdf = np.zeros((len(rel_positions), 181))
        n_valid = np.zeros(len(rel_positions))
        valid_parts = 0

        for pId in range(n_pp_ssm):
            human_aims = np.array(ssm.allAims[pId]).ravel() % 360.0
            human_aims_signed = normalize_angle(human_aims)
            aim = np.abs(human_aims_signed)

            part_cps = rot_cps[rot_cps['participant_id'] == pId]
            if part_cps.empty:
                continue
            best_t = int(part_cps['changepoint_position'].min())
            if best_t < min_t or best_t > max_t:
                continue

            human_series = np.full(len(rel_positions), np.nan)
            for ii, rel in enumerate(rel_positions):
                trial_idx = best_t + rel
                if 0 <= trial_idx < n_trials:
                    human_series[ii] = aim[trial_idx]
            human_series_list.append(human_series)

            pp_key = bht_ppnums[pId] if pId < len(bht_ppnums) else None
            if pp_key is None or pp_key not in bht.predState:
                continue

            ps = bht.predState[pp_key]
            comp_logp = np.array(ps['predCompLogProb'])
            comp_mean = np.array(ps['predCompMean'])
            comp_scale = np.array(ps['predCompScale'])
            n_comp_arr = np.array(ps['predNComp']).astype(int)
            n_model = len(n_comp_arr)

            _, pp_nu, _ = _extract_params(bht.xs[pId])

            for ii, rel in enumerate(rel_positions):
                trial_idx = best_t + rel
                if trial_idx < 0 or trial_idx >= min(n_trials, n_model):
                    continue

                nc = n_comp_arr[trial_idx]
                if nc <= 0:
                    continue

                lp = comp_logp[trial_idx, :nc]
                mn = comp_mean[trial_idx, :nc]
                sc = comp_scale[trial_idx, :nc]

                valid_comp = np.isfinite(lp) & (sc > 0)
                if not np.any(valid_comp):
                    continue

                lp_v = lp[valid_comp]
                mn_v = mn[valid_comp]
                sc_v = sc[valid_comp]
                wts = np.exp(lp_v - sp_logsumexp(lp_v))

                pdf_matrix = t_dist.pdf(
                    angles[:, None], df=pp_nu, loc=mn_v[None, :], scale=sc_v[None, :]
                )
                pdf_vals = pdf_matrix @ wts

                mag_pdf = np.zeros(181)
                mag_pdf[0] = pdf_vals[180]
                for m in range(1, 180):
                    mag_pdf[m] = pdf_vals[180 - m] + pdf_vals[180 + m]
                mag_pdf[180] = pdf_vals[0]

                average_mag_pdf[ii] += mag_pdf
                n_valid[ii] += 1

            valid_parts += 1

        if debug:
            print(f'Combined rot {rotation}: {valid_parts} valid participants')

        if valid_parts == 0 or len(human_series_list) == 0:
            continue

        for ii in range(len(rel_positions)):
            if n_valid[ii] > 0:
                average_mag_pdf[ii] /= n_valid[ii]

        idx_min = rel_positions.index(plot_min)
        idx_max = rel_positions.index(plot_max - 1)

        last_valid = None
        for ii in range(idx_min, idx_max + 1):
            if n_valid[ii] > 0:
                last_valid = average_mag_pdf[ii].copy()
            elif last_valid is not None:
                average_mag_pdf[ii] = last_valid
                n_valid[ii] = 1
        last_valid = None
        for ii in range(idx_max, idx_min - 1, -1):
            if n_valid[ii] > 0:
                last_valid = average_mag_pdf[ii].copy()
            elif last_valid is not None:
                average_mag_pdf[ii] = last_valid
                n_valid[ii] = 1

        rel_arr = np.array(rel_positions)
        Z = average_mag_pdf.T
        human_arr = np.stack(human_series_list)
        mean_human = np.nanmean(human_arr, axis=0)
        plot_slice = [i for i, r in enumerate(rel_positions) if plot_min <= r < plot_max]
        x_plot = [rel_positions[i] for i in plot_slice]
        y_plot = [mean_human[i] for i in plot_slice]

        contour_norm = LogNorm(vmin=1e-3, vmax=0.05)
        levels = np.logspace(np.log10(1e-3), np.log10(0.05), 24)

        darkened_hex = darken_color(rot_color, factor=0.82)
        darkened_rgb = mcolors.hex2color(darkened_hex)
        cdict = {
            'red': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[0], darkened_rgb[0])],
            'green': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[1], darkened_rgb[1])],
            'blue': [(0.0, 1.0, 1.0), (1.0, darkened_rgb[2], darkened_rgb[2])],
            'alpha': [(0.0, 0.0, 0.0), (1.0, 0.52, 0.52)],
        }
        custom_cmap = LinearSegmentedColormap(f'combined_{rotation}', cdict)

        ax.contourf(
            rel_arr,
            magnitudes,
            Z,
            levels=levels,
            cmap=custom_cmap,
            norm=contour_norm,
            zorder=2,
            extend='max',
        )

        for series in human_series_list:
            valid_mask = ~np.isnan(series)
            xs_j = []
            ys_j = []
            for i in plot_slice:
                if valid_mask[i]:
                    xs_j.append(rel_positions[i] + np.random.uniform(-0.12, 0.12))
                    ys_j.append(series[i])
            if xs_j:
                ax.scatter(
                    xs_j,
                    ys_j,
                    color=rot_color,
                    alpha=0.08,
                    edgecolor='black',
                    linewidth=0.25,
                    s=10,
                    zorder=3,
                )

        ax.scatter(x_plot, y_plot, color=rot_color, edgecolor='black', linewidth=1, s=50, zorder=4)
        ax.axhline(y=rotation, color=rot_color, alpha=0.18, linestyle='--', linewidth=1.1, zorder=1)

        rotations_found.append(rotation)
        legend_handles.append(
            Line2D(
                [],
                [],
                color=rot_color,
                marker='o',
                ls='None',
                markersize=7,
                markeredgecolor='black',
                markeredgewidth=0.8,
                label=f'{int(rotation)}\u00b0',
            )
        )

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1.2, alpha=0.35, zorder=0)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.3, zorder=0)
    ax.set_title('CG changepoint-aligned BHT predictive (combined)', fontsize=FS_TITLE)
    ax.set_xlabel('Trial rel. to changepoint', fontsize=FS_LABEL)
    ax.set_ylabel('|Aim| (\u00b0)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(plot_min - 0.5, plot_max - 0.5)
    ax.set_ylim(-5, 181)

    yticks = sorted(set([0, 180] + [int(r) for r in rotations_found]))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(t)}\u00b0' for t in yticks])

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title='Rotation',
            fontsize=FS_LEGEND,
            title_fontsize=FS_LEGEND,
            frameon=True,
            framealpha=0.9,
            loc='upper left',
        )

    ax.text(
        0.99,
        0.02,
        'Lines/points: human mean\nFilled contours: BHT predictive',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=FS_LEGEND,
        bbox=dict(facecolor='white', edgecolor='0.85', alpha=0.9, pad=3.0),
    )

    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if debug:
            print(f'Saved to {save_path}')
    plt.close(fig)


def plot_wildcard_dirichlet(csv_path, model_path, save_dir, debug=False):
    """Self vs other Dirichlet weights for wildcard (group + individual).

    Plots per-target weights on EVERY trial (not just when active),
    giving smooth continuous lines instead of spiky traces.
    """
    import json as _json
    from scipy.special import logsumexp as sp_lse

    os.makedirs(save_dir, exist_ok=True)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_LEGEND = 8

    def _parse(s):
        return _json.loads(s.replace('nan', 'null'))

    df_wide = pd.read_csv(csv_path)
    arr = np.load(model_path, allow_pickle=True)
    bht = arr[0]
    _apply_fit_model_config(bht, debug=debug)

    all_self_traces, all_other_traces, all_epochs = [], [], []

    for pIdx, pp in enumerate(bht.participantNums):
        ps = bht.predState.get(pp)
        if ps is None:
            continue

        destTargArr = np.array(
            [x for x in bht.destTargets[pp] if not np.isnan(x)], dtype=np.float64
        )
        nDest = len(destTargArr)

        wt = _compute_dirichlet_per_target(ps, destTargArr)  # (T, nDest)
        tgtIdx = ps['predTgtIdx']  # (T,)
        T = wt.shape[0]

        row = df_wide[df_wide['participantNum'] == pp].iloc[0]
        rots = _parse(row['rotation'])
        n = len(rots)
        first_rot = next(i for i in range(n) if rots[i] is not None and rots[i] != 0)
        last_rot = max(i for i in range(n) if rots[i] is not None and rots[i] != 0)

        epochs = np.full(T, np.nan)
        rot_counter = 0
        for t in range(T):
            if t >= first_rot and t < n:
                epochs[t] = rot_counter // 4
                rot_counter += 1
            elif t < first_rot:
                epochs[t] = -(first_rot - t + 3) // 4

        # Self weight follows the active target; wt contains all target weights.
        self_w = np.full(T, np.nan)
        other_w = np.full(T, np.nan)
        for t in range(T):
            ti = int(tgtIdx[t])
            if ti < 0 or ti >= nDest or np.isnan(wt[t, 0]):
                continue
            self_w[t] = wt[t, ti]
            other_w[t] = np.sum(wt[t, :]) - wt[t, ti]

        all_self_traces.append(self_w)
        all_other_traces.append(other_w)
        all_epochs.append(epochs)

        valid = np.any(np.isfinite(wt), axis=1)
        if np.sum(valid) > 5:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
            trials_shifted = np.arange(T) - first_rot
            x_lo = trials_shifted[0] - 0.5
            x_hi = trials_shifted[-1] + 0.5

            tgt_colors = plt.cm.tab10(np.linspace(0, 1, nDest))
            for j in range(nDest):
                w_j = wt[:, j]
                v = np.isfinite(w_j)
                lbl = f'Target {int(destTargArr[j])}'
                ax.plot(
                    trials_shifted[v], w_j[v], color=tgt_colors[j], lw=1.5, alpha=0.8, label=lbl
                )

            ax.axvline(x=0, color='grey', ls='--', lw=2, alpha=0.6)
            wo_start = last_rot + 1 - first_rot
            ax.axvline(x=wo_start, color='grey', ls='--', lw=2, alpha=0.6)
            eq_line = 1.0 / nDest
            ax.axhline(y=eq_line, color='grey', ls=':', lw=1, alpha=0.4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel('Trial (relative to rotation onset)', fontsize=FS_LABEL)
            ax.set_ylabel('Dirichlet Source Weight', fontsize=FS_LABEL)
            ax.set_title(
                f'Wildcard | PP {pp} — Per-target Dirichlet Weights', fontsize=FS_TITLE + 1
            )
            ax.tick_params(labelsize=FS_TICK)
            ax.legend(fontsize=FS_LEGEND + 1, loc='upper right', framealpha=0.7, ncol=2)
            sns.despine(ax=ax)
            fpath = os.path.join(save_dir, f'wildcard_pp{pp}_dirichlet.png')
            fig.savefig(fpath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            if debug and pIdx < 3:
                print(f"  Saved {fpath}")

    max_ep = 80
    ep_self = {e: [] for e in range(-5, max_ep)}
    ep_other = {e: [] for e in range(-5, max_ep)}

    for sw, ow, ep in zip(all_self_traces, all_other_traces, all_epochs):
        for t in range(len(sw)):
            if not np.isfinite(sw[t]) or not np.isfinite(ep[t]):
                continue
            e = int(round(ep[t]))
            if e in ep_self:
                ep_self[e].append(sw[t])
                ep_other[e].append(ow[t])

    epochs_sorted = sorted(e for e in ep_self if len(ep_self[e]) > 0)
    ep_arr = np.array(epochs_sorted)
    self_mean = np.array([np.mean(ep_self[e]) for e in epochs_sorted])
    other_mean = np.array([np.mean(ep_other[e]) for e in epochs_sorted])
    self_ci = np.array(
        [1.96 * np.std(ep_self[e]) / np.sqrt(len(ep_self[e])) for e in epochs_sorted]
    )
    other_ci = np.array(
        [1.96 * np.std(ep_other[e]) / np.sqrt(len(ep_other[e])) for e in epochs_sorted]
    )

    fig, ax = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
    ax.fill_between(ep_arr, self_mean - self_ci, self_mean + self_ci, color='#4B0082', alpha=0.15)
    ax.plot(ep_arr, self_mean, color='#4B0082', lw=2, alpha=0.9, label='Self')
    ax.fill_between(
        ep_arr, other_mean - other_ci, other_mean + other_ci, color='#7B68EE', alpha=0.15
    )
    ax.plot(ep_arr, other_mean, color='#7B68EE', lw=2, alpha=0.9, ls='--', label='Other')
    ax.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Epoch (4 trials per epoch)', fontsize=FS_LABEL)
    ax.set_ylabel('Source weight', fontsize=FS_LABEL)
    ax.set_title('Wildcard — Dirichlet Self vs Other (group)', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='center right', framealpha=0.7)
    sns.despine(ax=ax)

    sp = os.path.join(save_dir, 'wildcard_dirichlet_group')
    fig.savefig(sp + '.svg', dpi=200, bbox_inches='tight')
    fig.savefig(sp + '.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved {sp}")


def plot_cross_dataset_params(datasets_spec, save_path=None):
    from matplotlib.patches import Patch

    param_names = [
        r'Baseline aim s.d. ($\sigma_0$)',
        r'Baseline $\nu$',
        r'Switch delay ($-\log h$)',
        r'Harmonic var. ($\sigma^2_{\mathrm{T}}$)',
        r'Alpha ($\alpha$)',
        r'Kappa ($\kappa$)',
        r'Prior odds (struct.)',
        r'Vis. noise coeff.',
        r'Tempering ($\beta$)',
    ]
    # xs starts with baseline scale and degrees of freedom, then seven free parameters.
    log_indices = {2, 3, 4, 7, 8}
    logscale_axes = {2, 3, 4, 7, 8}
    rot_markers = {
        15: '^',
        30: 's',
        45: 'D',
        60: 'v',
        90: 'P',
    }
    default_marker = 'o'
    entries = []
    for spec in datasets_spec:
        label = spec['label']
        color = spec['color']
        rot = spec.get('rot_deg', None)
        marker = rot_markers.get(rot, default_marker) if rot else default_marker
        obj = spec['obj']
        if isinstance(obj, list):
            all_xs = []
            for o in obj:
                all_xs.extend(o.xs)
        else:
            all_xs = list(obj.xs)
        param_arr = np.array(all_xs, dtype=float)
        entries.append((label, color, marker, param_arr, spec.get('family', label)))
    n_params = len(param_names)
    n_entries = len(entries)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = axes.ravel()
    rng = np.random.default_rng(42)
    for pidx in range(n_params):
        ax = axes[pidx]
        do_exp = pidx in log_indices
        use_log = pidx in logscale_axes
        for eidx, (label, color, marker, param_arr, family) in enumerate(entries):
            if pidx >= param_arr.shape[1]:
                continue
            raw = param_arr[:, pidx].copy()
            if do_exp:
                vals = np.exp(np.clip(raw, -300, 300))
            else:
                vals = raw.copy()
            vals = np.where(np.isfinite(vals), vals, np.nan)
            if use_log:
                vals = np.where(vals > 0, vals, np.nan)
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                continue
            jitter = rng.uniform(-0.28, 0.28, len(valid))
            ax.scatter(
                eidx + jitter,
                valid,
                color=color,
                marker=marker,
                s=14,
                alpha=0.3,
                edgecolors='none',
                zorder=3,
                rasterized=True,
            )
            median_val = np.nanmedian(valid)
            q25, q75 = np.nanpercentile(valid, [25, 75])
            ax.plot(
                [eidx - 0.32, eidx + 0.32],
                [median_val, median_val],
                color='k',
                lw=2.0,
                zorder=7,
                solid_capstyle='round',
            )
            ax.plot([eidx, eidx], [q25, q75], color='k', lw=2.4, zorder=6)
            ax.plot([eidx - 0.15, eidx + 0.15], [q25, q25], color='k', lw=1.0, zorder=6)
            ax.plot([eidx - 0.15, eidx + 0.15], [q75, q75], color='k', lw=1.0, zorder=6)
        if use_log:
            ax.set_yscale('log')
        ax.set_title(param_names[pidx], fontsize=10, fontweight='bold', pad=6)
        ax.set_xticks(range(n_entries))
        ax.set_xticklabels([e[0] for e in entries], rotation=55, ha='right', fontsize=6.5)
        ax.tick_params(axis='y', labelsize=8)
        ax.yaxis.grid(True, alpha=0.15, lw=0.5)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    for pidx in range(n_params, len(axes)):
        axes[pidx].set_visible(False)
    seen_families = {}
    for label, color, marker, param_arr, family in entries:
        if family not in seen_families:
            seen_families[family] = color
    colour_handles = [
        Patch(facecolor=c, edgecolor='none', label=fam) for fam, c in seen_families.items()
    ]
    leg1 = fig.legend(
        handles=colour_handles,
        loc='upper left',
        fontsize=8,
        framealpha=0.85,
        title='Dataset',
        title_fontsize=9,
        bbox_to_anchor=(0.005, 1.005),
        ncol=len(colour_handles),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    used_rots = set()
    for spec in datasets_spec:
        r = spec.get('rot_deg', None)
        if r is not None:
            used_rots.add(r)
    marker_handles = [
        Line2D(
            [],
            [],
            marker=rot_markers[r],
            color='grey',
            markersize=6,
            ls='None',
            markeredgecolor='none',
            label=f'{int(r)}°',
        )
        for r in sorted(used_rots)
        if r in rot_markers
    ]
    if any(spec.get('rot_deg') is None for spec in datasets_spec):
        marker_handles.append(
            Line2D(
                [],
                [],
                marker='o',
                color='grey',
                markersize=6,
                ls='None',
                markeredgecolor='none',
                label='Mixed',
            )
        )
    fig.legend(
        handles=marker_handles,
        loc='upper right',
        fontsize=8,
        framealpha=0.85,
        title='Rotation',
        title_fontsize=9,
        bbox_to_anchor=(0.995, 1.005),
        ncol=len(marker_handles),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.add_artist(leg1)
    fig.suptitle('Fitted model parameters across datasets', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter comparison to {save_path}")
    plt.close(fig)
