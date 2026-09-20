import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm, t as student_t_dist
from scipy.stats.qmc import LatinHypercube
import cma
import pickle
import hashlib

SEED = 42
np.random.seed(SEED)
import multiprocessing as mp
from multiprocessing import Manager
from numba import njit
import math
import time
from types import SimpleNamespace
import os

n_actions = 361
action_centers = np.arange(-180, 181, dtype=np.float64)
n_mag = 181
action_mags = np.arange(0, 181, dtype=np.float64)
HMM_LIKELIHOOD_VERSION = 'student_t_single_image_wrapped_angle_v4'
DEFAULT_NU = 5.0
MIN_NU = 10.0
MAX_BASELINE_NU = 1000.0
GAUSSIAN_APPROX_NU = 1000.0
_MAX_BASELINE_LOG_NU_SPAN = math.log(MAX_BASELINE_NU - MIN_NU)
MIN_SCALE = 1.0
CACHE_FAILURE_NLL = 1e11


@njit(cache=True)
def get_action_index(action):
    rounded = round(action)
    shifted = rounded + 180
    if shifted < 0:
        shifted = 0
    elif shifted > n_actions - 1:
        shifted = n_actions - 1
    return int(shifted)


@njit(cache=True)
def angular_dist(x):
    return np.abs(((x + 180.0) % 360.0 - 180.0))


@njit(cache=True)
def signed_angular_dist(x):
    return (x + 180.0) % 360.0 - 180.0


@njit(cache=True)
def gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return np.full_like(x, -1e30)
    return -np.log(sigma * np.sqrt(2 * np.pi)) - (x**2) / (2 * sigma**2)


@njit(cache=True)
def scalar_gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return -1e30
    return -np.log(sigma * np.sqrt(2 * np.pi)) - (x**2) / (2 * sigma**2)


@njit(cache=True)
def logStudentTPdf(x, mu, sigma, nu):
    if sigma <= 0.0 or nu <= 0.0:
        return -1e30
    if nu >= GAUSSIAN_APPROX_NU:
        return scalar_gaussianLogPdf(x - mu, sigma)
    z = (x - mu) / sigma
    return (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - math.log(sigma)
        - 0.5 * (nu + 1.0) * math.log(1.0 + z * z / nu)
    )


@njit(cache=True)
def logWrappedStudentTPdf(x, mu, sigma, nu, period=360.0):
    if sigma <= 0.0 or nu <= 0.0:
        return -1e30
    if nu >= GAUSSIAN_APPROX_NU:
        dx = ((x - mu + 180.0) % 360.0) - 180.0
        return scalar_gaussianLogPdf(dx, sigma)
    arr = np.empty(3, dtype=np.float64)
    arr[0] = logStudentTPdf(x, mu, sigma, nu)
    arr[1] = logStudentTPdf(x - mu + period, 0.0, sigma, nu)
    arr[2] = logStudentTPdf(x - mu - period, 0.0, sigma, nu)
    return logsumexp(arr)


@njit(cache=True)
def logWrappedAngleStudentTSingleImagePdf(x, mu, sigma, nu):
    """Wrap the angular difference once, then evaluate a single-image Student-t."""
    dx = signed_angular_dist(x - mu)
    return logStudentTPdf(dx, 0.0, sigma, nu)


@njit(cache=True)
def safe_exp(z):
    """Clamp to avoid denorm underflow; error <1e-300."""
    threshold = -700.0
    below = z < threshold
    result = np.zeros_like(z)
    result[~below] = np.exp(z[~below])
    return result


@njit(cache=True)
def logsumexp(arr):
    if arr.size == 0:
        return -1e30
    m = np.max(arr)
    s = np.sum(safe_exp(arr - m))
    if s == 0:
        return m - 1e10
    return m + np.log(s)


@njit(cache=True)
def softmax(log_probs):
    max_log_p = np.max(log_probs)
    exp_p = safe_exp(log_probs - max_log_p)
    sum_p = np.sum(exp_p)
    if sum_p > 0:
        return exp_p / sum_p
    else:
        return np.array([0.5, 0.5])


def fitBaselineStudentT(bl_aims):
    """Fit baseline Student-t params with the same constraints used by dingBLR."""
    bl = np.asarray(bl_aims, dtype=np.float64)
    bl = bl[~np.isnan(bl)]
    if len(bl) < 5:
        if len(bl) > 1:
            sc = max(1.4826 * np.median(np.abs(bl - np.median(bl))), MIN_SCALE)
        else:
            sc = 3.0
        return sc, MIN_NU

    def negll(params):
        log_scale, log_nu_minus_min = params
        scale = max(np.exp(np.clip(log_scale, -700.0, 700.0)), MIN_SCALE)
        nu = MIN_NU + np.exp(np.clip(log_nu_minus_min, -700.0, _MAX_BASELINE_LOG_NU_SPAN))
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

    mad_scale = max(1.4826 * np.median(np.abs(bl - np.median(bl))), MIN_SCALE)
    best_val = np.inf
    best_params = (mad_scale, float(MIN_NU))
    for nu_init in [MIN_NU + 0.1, 30.0, 100.0]:
        x0 = [np.log(mad_scale), np.log(min(max(nu_init - MIN_NU, 1e-6), MAX_BASELINE_NU - MIN_NU))]
        try:
            res = minimize(
                negll,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6},
            )
            if res.fun < best_val:
                best_val = res.fun
                sc = max(np.exp(np.clip(res.x[0], -700.0, 700.0)), MIN_SCALE)
                nu = MIN_NU + np.exp(np.clip(res.x[1], -700.0, _MAX_BASELINE_LOG_NU_SPAN))
                best_params = (sc, nu)
        except Exception:
            continue
    return best_params


def _select_first_block(pdat):
    if 'blockNum' not in pdat.columns:
        return pdat.copy()
    block_nums = pdat['blockNum'].dropna().unique()
    if len(block_nums) == 0:
        return pdat.copy()
    return pdat[pdat['blockNum'] == block_nums[0]].copy()


def _get_baseline_student_t_from_df(pdat_full):
    pdat0 = _select_first_block(pdat_full)
    phases = None
    try:
        from dingBLR import derivePhase as _derivePhase

        phases = np.asarray(_derivePhase(pdat0))
    except Exception:
        if 'phase' in pdat0.columns:
            phases = pdat0['phase'].astype(str).values

    if phases is not None and len(phases) == len(pdat0):
        baseline_mask = np.array([str(p).lower() == 'baseline' for p in phases], dtype=bool)
    else:
        rots = (
            pdat0['rotation'].values.astype(np.float64)
            if 'rotation' in pdat0.columns
            else np.zeros(len(pdat0))
        )
        nz = np.flatnonzero(np.abs(rots) > 1e-9)
        first_nonzero = int(nz[0]) if len(nz) > 0 else len(rots)
        baseline_mask = np.arange(len(pdat0)) < first_nonzero

    bl = pdat0['aim'].values.astype(np.float64)[baseline_mask]
    return fitBaselineStudentT(bl)


def _get_phase_labels_from_df(pdat_full):
    pdat0 = _select_first_block(pdat_full)
    phases = None
    try:
        from dingBLR import derivePhase as _derivePhase

        phases = np.asarray(_derivePhase(pdat0))
    except Exception:
        if 'phase' in pdat0.columns:
            phases = pdat0['phase'].astype(str).values

    if phases is not None and len(phases) == len(pdat0):
        return phases

    rots = (
        pdat0['rotation'].values.astype(np.float64)
        if 'rotation' in pdat0.columns
        else np.zeros(len(pdat0))
    )
    nz = np.flatnonzero(np.abs(rots) > 1e-9)
    first_nonzero = int(nz[0]) if len(nz) > 0 else len(rots)
    last_nonzero = int(nz[-1]) if len(nz) > 0 else -1
    phases = np.empty(len(rots), dtype=object)
    for i in range(len(rots)):
        if i < first_nonzero:
            phases[i] = 'baseline'
        elif i <= last_nonzero:
            phases[i] = 'rotation'
        else:
            phases[i] = 'washout'
    return phases


def _sanitize_cache_token(value):
    token = str(value)
    cleaned = []
    for char in token:
        if char.isalnum() or char in ('-', '_'):
            cleaned.append(char)
        elif char in ('.', ' '):
            cleaned.append('_')
    cleaned = ''.join(cleaned).strip('_')
    return cleaned or 'none'


def _stable_df_digest(df):
    hash_cols = [
        col
        for col in [
            'participantNum',
            'blockNum',
            'trial',
            'trialNum',
            'phase',
            'rotation',
            'targetPosition',
            'aim',
            'condition',
            'targetCount',
            'reward_delay',
            'hasAutocorrection',
        ]
        if col in df.columns
    ]
    if len(hash_cols) == 0:
        return 'nodata'
    hashed = pd.util.hash_pandas_object(df[hash_cols], index=True).values
    return hashlib.sha1(hashed.tobytes()).hexdigest()[:12]


def _resolve_cache_dir(
    dataset_name, df, condition, conVal, fitPhase, annealing_mode, flipRot, fitWashout
):
    if dataset_name not in (None, '', 'default'):
        return dataset_name

    tokens = ['HMM']
    if condition not in (None, 'none'):
        tokens.append(_sanitize_cache_token(condition))
    if conVal not in (None, 'none'):
        tokens.append(_sanitize_cache_token(conVal))
    else:
        tokens.append('allparticipants')

    if 'targetCount' in df.columns:
        target_counts = pd.unique(df['targetCount'].dropna())
        if len(target_counts) == 1:
            try:
                tokens.append(f"{int(target_counts[0])}tar")
            except Exception:
                tokens.append(_sanitize_cache_token(target_counts[0]))

    if 'hasAutocorrection' in df.columns:
        ac_vals = pd.unique(df['hasAutocorrection'].dropna())
        if len(ac_vals) == 1:
            tokens.append(f"ac{int(ac_vals[0])}")

    if 'reward_delay' in df.columns:
        reward_delays = pd.unique(df['reward_delay'].dropna())
        if len(reward_delays) == 1:
            try:
                tokens.append(f"delay{int(round(float(reward_delays[0])))}")
            except Exception:
                tokens.append(_sanitize_cache_token(reward_delays[0]))

    if fitPhase is None:
        tokens.append('allphases')
    else:
        tokens.append(f"phase_{_sanitize_cache_token(fitPhase)}")

    tokens.append(f"ann{int(annealing_mode)}")
    tokens.append('fliprot' if flipRot else 'noflip')
    tokens.append('washout' if fitWashout else 'nowash')
    tokens.append(f"pps{df['participantNum'].nunique() if 'participantNum' in df.columns else 0}")
    tokens.append(_stable_df_digest(df))

    return '_'.join(tokens)


# State 0 is baseline; state 1 learns separate magnitude and direction policies.
@njit(cache=True)
def computeNegLl(
    params, aims, rots, targetPositions, annealing_mode=0, scale_S0=3.0, nu_S0=MIN_NU
):  # 0: none, 1: typical, 2: principled
    noise = params[0]
    kappa = params[1]
    noise_context = params[2]
    alpha = params[3]  # Q-learning learning rate
    beta = params[4]  # Policy inverse temperature (target/max)
    beta_dir = params[5]  # Sign direction inverse temperature
    if (
        noise <= 0.0
        or kappa < 0.0
        or kappa >= 1.0
        or noise_context <= 0.0
        or alpha <= 0.0
        or beta <= 0.0
        or beta_dir <= 0.0
    ):
        return 1e100
    scale_S0 = max(float(scale_S0), MIN_SCALE)
    nu_S0 = min(max(float(nu_S0), MIN_NU), MAX_BASELINE_NU)
    logLikelihood = 0.0
    n = len(aims)
    si = 0.0
    pi = np.array([1.0, 0])
    A = np.zeros((2, 2))
    A[0, 0] = 1.0 - kappa
    A[0, 1] = kappa
    A[1, 0] = kappa
    A[1, 1] = 1.0 - kappa
    log_A = np.log(A + 1e-301)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)  # Scalar Q for high-level sign branches
    prev_rot = 0.0
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
        target_pos = targetPositions[t]
        has_aim = not np.isnan(aim)
        log_pi_pred = np.empty(2)
        for sp in range(2):
            log_trans = np.log(pi + 1e-300) + log_A[:, sp]
            log_pi_pred[sp] = logsumexp(log_trans)
        pi_pred = softmax(log_pi_pred)
        if annealing_mode == 1:  # typical: linear from 0 to beta
            beta_t = beta * (t / (n - 1.0)) if n > 1 else beta
        elif annealing_mode == 2:  # principled: beta * pi_pred[1]
            beta_t = beta * pi_pred[1]
        else:  # none: static beta
            beta_t = beta
        mOut0 = 0.0
        # Low-level policy (shared magnitude across signs)
        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta_t * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)
        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
        m_mag = np.dot(probs_mag, action_mags)
        mOut1 = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)
        mean_mOut = pi_pred[0] * mOut0 + pi_pred[1] * mOut1
        full_comp = signed_angular_dist(rot)
        has_data = has_aim
        if has_data:
            target_t = aim
            log_em0 = logWrappedAngleStudentTSingleImagePdf(target_t, mOut0, scale_S0, nu_S0)
            # state 1: policy mixture over branches
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                p_mags = probs_mag
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    g_log1 = logWrappedAngleStudentTSingleImagePdf(target_t, act, noise, DEFAULT_NU)
                    log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            log_marg = logsumexp(log_pi_pred + np.array([log_em0, log_em1]))
            logLikelihood += log_marg
        log_em_context0 = logWrappedAngleStudentTSingleImagePdf(
            mOut0 + full_comp, 0.0, scale_S0, nu_S0
        )
        # state 1: policy mixture over branches
        unnorm_context1 = np.full(n_actions, -1e30)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            p_mags = probs_mag
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                g_log_context1 = logWrappedAngleStudentTSingleImagePdf(
                    act + full_comp, 0.0, noise_context, DEFAULT_NU
                )
                log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                j = get_action_index(act)
                unnorm_context1[j] = log_p_action + g_log_context1
        log_em_context1 = logsumexp(unnorm_context1)
        log_alpha = log_pi_pred + np.array([log_em_context0, log_em_context1])
        pi = softmax(log_alpha)
        # High-level Q_dir update (off-policy Q-learning)
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * pi[1] * td_e_pos
        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * pi[1] * td_e_neg
        # Low-level Q_mag update (shared, max over signs off-policy Q-learning)
        for k in range(n_mag):
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos

            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg

            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]

            Q_mag[k] += alpha * pi[1] * td_e
        prev_rot = rot
    return -logLikelihood


class Objective:
    """Wraps HMM's computeNegLl for normalized [0,1] parameter space."""

    def __init__(
        self, aims, rots, targetPositions, bounds, annealing_mode=0, scale_S0=3.0, nu_S0=MIN_NU
    ):
        self.aims = aims
        self.rots = rots
        self.targetPositions = targetPositions
        self.annealing_mode = annealing_mode
        self.scale_S0 = max(float(scale_S0), MIN_SCALE)
        self.nu_S0 = min(max(float(nu_S0), MIN_NU), MAX_BASELINE_NU)
        self.bounds = np.array(bounds)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

    def denormalize(self, pN):
        """Convert from [0,1]^d to actual parameter space."""
        return np.clip(self.lower + pN * (self.upper - self.lower), self.lower, self.upper)

    def __call__(self, pN):
        """Evaluate objective in normalized space."""
        return computeNegLl(
            self.denormalize(pN),
            self.aims,
            self.rots,
            self.targetPositions,
            self.annealing_mode,
            self.scale_S0,
            self.nu_S0,
        )


def _run_cma_restarts(objFunc, nFree, nRestarts, nPolish, maxfevals_per_run, sigma0, seed, pp=None):
    """CMA-ES multistart with LHS seeding and L-BFGS-B polishing.

    Parameters
    ----------
    objFunc : callable
        Objective on [0, 1]^nFree.
    nFree : int
        Number of free parameters.
    nRestarts : int
        Number of CMA-ES restarts.
    nPolish : int
        Number of top CMA solutions to polish with L-BFGS-B.
    maxfevals_per_run : int
        Max function evaluations per CMA-ES restart.
    sigma0 : float
        Initial step size (in [0, 1] normalised space).
    seed : int
        RNG seed.
    pp : str or None
        Participant label for progress printing.
    """
    sampler = LatinHypercube(d=nFree, seed=seed)
    starts = sampler.random(n=nRestarts)
    normalizedBounds = [(0.0, 1.0)] * nFree

    results = []  # list of (negll, x_normalised)
    bestVal = np.inf
    bestX = None
    totalEvals = 0
    t_start = time.time()

    cma_opts = {
        'bounds': [0.0, 1.0],
        'maxfevals': maxfevals_per_run,
        'verbose': -9,  # silence CMA's own output
        'seed': seed,
        'tolfun': 1e-8,
        'tolx': 1e-8,
    }

    for ri in range(nRestarts):
        x0 = np.clip(starts[ri], 0.02, 0.98)  # keep away from boundary
        cma_opts_i = dict(cma_opts)
        cma_opts_i['seed'] = seed + ri
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts_i)
            es.optimize(objFunc)
            res_x = np.clip(es.result.xbest, 0.0, 1.0)
            res_f = es.result.fbest
            totalEvals += es.result.evaluations
        except Exception:
            continue
        results.append((res_f, res_x))
        if res_f < bestVal:
            bestVal = res_f
            bestX = res_x.copy()

        if (ri + 1) % 5 == 0 or ri == nRestarts - 1:
            elapsed = time.time() - t_start
            frac = (ri + 1) / nRestarts
            eta = elapsed / frac * (1.0 - frac) if frac < 1.0 else 0.0
            eta_polish = elapsed / max(ri + 1, 1) * nPolish * 0.3  # rough polish estimate
            eta_total = eta + eta_polish
            print(
                f"  {pp} CMA restart {ri + 1}/{nRestarts}: "
                f"best={bestVal:.2f}  evals={totalEvals}  "
                f"ETA {eta_total:.0f}s",
                flush=True,
            )

    results.sort(key=lambda x: x[0])
    nPolish = min(nPolish, len(results))
    for pi, (fun, xNorm) in enumerate(results[:nPolish]):
        try:
            resDeep = minimize(
                objFunc,
                xNorm,
                method='L-BFGS-B',
                bounds=normalizedBounds,
                options={'maxiter': 500, 'ftol': 1e-10},
            )
            totalEvals += resDeep.nfev
            if resDeep.fun < bestVal:
                bestVal = resDeep.fun
                bestX = resDeep.x.copy()
        except Exception:
            pass

    elapsed_total = time.time() - t_start
    print(
        f"  {pp} CMA done: best={bestVal:.2f}  total_evals={totalEvals}  [{elapsed_total:.1f}s]",
        flush=True,
    )

    return results, bestVal, bestX


def fitSingle(data, boundsSingle, popSizeMultiplier, datasetName):
    """Fit HMM for a single participant using CMA-ES.

    Parameters
    ----------
    data : tuple
        (aims, rots, targetPositions, pp, annealing_mode[, scale_S0, nu_S0])
    boundsSingle : list of tuples
        Parameter bounds in original space
    popSizeMultiplier : float
        Scaling factor for number of restarts
    datasetName : str
        Dataset directory for caching

    Returns
    -------
    bestX : ndarray
        Best parameters found
    bestValue : float
        Objective value at best parameters
    negll : float
        Negative log-likelihood
    """
    if len(data) >= 7:
        aims, rots, targetPositions, pp, annealing_mode, scale_S0, nu_S0 = data[:7]
    else:
        aims, rots, targetPositions, pp, annealing_mode = data
        scale_S0, nu_S0 = 3.0, MIN_NU
    objFunc = Objective(
        aims,
        rots,
        targetPositions,
        boundsSingle,
        annealing_mode,
        scale_S0=scale_S0,
        nu_S0=nu_S0,
    )
    numSamples = np.sum(~np.isnan(aims))
    if numSamples == 0:
        return np.zeros(len(boundsSingle)), 0.0, 0.0
    nParams = len(boundsSingle)
    save_path = f"{datasetName}/{pp}.pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            saved = pickle.load(f)
        if (
            saved.get('likelihood_version') == HMM_LIKELIHOOD_VERSION
            and 'bestX' in saved
            and len(saved['bestX']) == nParams
        ):
            cached_best_x = np.asarray(saved['bestX'], dtype=np.float64)
            cached_best_value = float(saved.get('bestValue', np.inf))
            cached_negll = float(saved.get('negll', cached_best_value))
            if (
                np.isfinite(cached_best_value)
                and np.isfinite(cached_negll)
                and cached_best_value < CACHE_FAILURE_NLL
                and cached_negll < CACHE_FAILURE_NLL
            ):
                current_negll = computeNegLl(
                    cached_best_x,
                    aims,
                    rots,
                    targetPositions,
                    annealing_mode,
                    scale_S0=scale_S0,
                    nu_S0=nu_S0,
                )
                if np.isfinite(current_negll) and current_negll < CACHE_FAILURE_NLL:
                    if abs(current_negll - cached_negll) > 1e-9:
                        with open(save_path, 'wb') as f:
                            pickle.dump(
                                {
                                    'bestX': cached_best_x,
                                    'bestValue': current_negll,
                                    'negll': current_negll,
                                    'likelihood_version': HMM_LIKELIHOOD_VERSION,
                                },
                                f,
                            )
                    print(f"{pp} using cached HMM fit from {save_path}", flush=True)
                    return cached_best_x, current_negll, current_negll
        print(f"{pp} cached HMM fit unusable; refitting", flush=True)

    t_start = time.time()

    nRestarts = int(40 * popSizeMultiplier)
    nPolish = max(5, nRestarts // 8)
    maxfevals_per_run = 8000

    results, bestVal, bestX = _run_cma_restarts(
        objFunc,
        nFree=nParams,
        nRestarts=nRestarts,
        nPolish=nPolish,
        maxfevals_per_run=maxfevals_per_run,
        sigma0=0.3,
        seed=SEED,
        pp=pp,
    )

    bestX_real = objFunc.denormalize(bestX) if bestX is not None else np.zeros(nParams)
    bestValue = bestVal

    t_elapsed = time.time() - t_start
    print(f"{pp} Final ({nParams}p): {bestValue:.2f} params={bestX_real}  [{t_elapsed:.1f}s]")
    with open(save_path, 'wb') as f:
        pickle.dump(
            {
                'bestX': bestX_real,
                'bestValue': bestValue,
                'negll': bestValue,
                'likelihood_version': HMM_LIKELIHOOD_VERSION,
            },
            f,
        )
    return bestX_real, bestValue, bestValue


def _student_t_bin_mass(low, high, mu, sigma, nu):
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(nu) or nu <= 0.0:
        return 0.0
    if nu >= GAUSSIAN_APPROX_NU:
        z_hi = (high - mu) / sigma
        z_lo = (low - mu) / sigma
        total = norm.cdf(z_hi) - norm.cdf(z_lo)
        return float(max(total, 0.0))
    z_hi = (high - mu) / sigma
    z_lo = (low - mu) / sigma
    total = student_t_dist.cdf(z_hi, df=nu) - student_t_dist.cdf(z_lo, df=nu)
    return float(max(total, 0.0))


def run_forward_pass(
    params, aims, rots, targetPositions, annealing_mode=0, scale_S0=3.0, nu_S0=MIN_NU
):
    params = np.asarray(params, dtype=np.float64)
    aims = np.asarray(aims, dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)
    targetPositions = np.asarray(targetPositions, dtype=np.float64)
    noise = params[0]
    kappa = params[1]
    noise_context = params[2]
    alpha = params[3]
    beta = params[4]
    beta_dir = params[5]
    scale_S0 = max(float(scale_S0), MIN_SCALE)
    nu_S0 = min(max(float(nu_S0), MIN_NU), MAX_BASELINE_NU)

    n = len(aims)
    pi = np.array([1.0, 0.0])
    A = np.zeros((2, 2))
    A[0, 0] = 1.0 - kappa
    A[0, 1] = kappa
    A[1, 0] = kappa
    A[1, 1] = 1.0 - kappa
    log_A = np.log(A + 1e-301)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)

    totErr = []
    log_liks = []
    observed = []
    pi_preds = []
    human_explicits = []
    model_explicits_valid = []
    valid_trials = []
    expected_aims = []
    model_predictive_pis = []
    model_predictive_policies = []
    model_predictive_dirs = []
    all_model_explicits = []

    for t in range(n):
        log_pi_pred = np.empty(2)
        for sp in range(2):
            log_trans = np.log(pi + 1e-300) + log_A[:, sp]
            log_pi_pred[sp] = logsumexp(log_trans)
        rot = rots[t]
        aim = aims[t]
        pi_pred = softmax(log_pi_pred)
        pi_preds.append(pi_pred.copy())
        model_predictive_pis.append(pi_pred.copy())

        if annealing_mode == 1:
            beta_t = beta * (t / (n - 1.0)) if n > 1 else beta
        elif annealing_mode == 2:
            beta_t = beta * pi_pred[1]
        else:
            beta_t = beta

        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta_t * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)

        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
        model_predictive_dirs.append(probs_dir.copy())

        mOut0 = 0.0
        m_mag = np.dot(probs_mag, action_mags)
        mOut1 = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)

        full_policy = np.zeros(n_actions)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                j = get_action_index(act)
                full_policy[j] += p_d * probs_mag[k]
        model_predictive_policies.append(full_policy)

        mean_mOut = pi_pred[0] * mOut0 + pi_pred[1] * mOut1
        mean_mOut = np.mod(mean_mOut + 180.0, 360.0) - 180.0
        all_model_explicits.append(mean_mOut)

        has_aim = not np.isnan(aim)
        if has_aim:
            valid_trials.append(t)
            target_t = aim
            log_em0 = logWrappedAngleStudentTSingleImagePdf(target_t, mOut0, scale_S0, nu_S0)
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    g_log1 = logWrappedAngleStudentTSingleImagePdf(target_t, act, noise, DEFAULT_NU)
                    log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            log_marg = logsumexp(log_pi_pred + np.array([log_em0, log_em1]))
            log_liks.append(log_marg)
            observed.append(target_t)
            angularError = signed_angular_dist(target_t - mean_mOut)
            totErr.append(angularError)
            expected_aims.append(mean_mOut)
            human_explicits.append(target_t)
        else:
            human_explicits.append(np.nan)
        model_explicits_valid.append(mean_mOut)

        full_comp = signed_angular_dist(rot)
        log_em_context0 = logWrappedAngleStudentTSingleImagePdf(
            mOut0 + full_comp, 0.0, scale_S0, nu_S0
        )
        unnorm_context1 = np.full(n_actions, -1e30)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                g_log_context1 = logWrappedAngleStudentTSingleImagePdf(
                    act + full_comp, 0.0, noise_context, DEFAULT_NU
                )
                log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                j = get_action_index(act)
                unnorm_context1[j] = log_p_action + g_log_context1
        log_em_context1 = logsumexp(unnorm_context1)
        log_alpha = log_pi_pred + np.array([log_em_context0, log_em_context1])
        pi_post = softmax(log_alpha)
        pi = pi_post

        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        Q_dir[0] += alpha * pi_post[1] * (max_r_pos - Q_dir[0])

        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        Q_dir[1] += alpha * pi_post[1] * (max_r_neg - Q_dir[1])

        for k in range(n_mag):
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos

            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg

            Q_mag[k] += alpha * pi_post[1] * (max(r_pos, r_neg) - Q_mag[k])

    logLikelihood = float(np.sum(log_liks)) if len(log_liks) > 0 else 0.0
    totErr_arr = np.array(totErr, dtype=np.float64)
    rmse = np.sqrt(np.mean(totErr_arr**2)) if len(totErr_arr) > 0 else 0.0
    ssRes = np.sum(totErr_arr**2) if len(totErr_arr) > 0 else 0.0
    observed_arr = np.array(observed, dtype=np.float64)
    rSquared = 0.0
    if len(observed_arr) > 0:
        meanObs = np.mean(observed_arr)
        signed_obs = np.array([signed_angular_dist(o - meanObs) for o in observed_arr])
        ssTot = np.sum(signed_obs**2)
        rSquared = 1 - (ssRes / ssTot) if ssTot != 0 else 0.0

    return dict(
        logLikelihood=logLikelihood,
        rmse=rmse,
        rSquared=rSquared,
        errors=totErr_arr,
        mStates=model_explicits_valid if len(totErr_arr) > 0 else [],
        allAims=aims.tolist(),
        pi_preds=pi_preds,
        valid_trials=valid_trials,
        model_explicits=model_explicits_valid,
        all_model_explicits=all_model_explicits,
        human_explicits=human_explicits,
        expected_aims=expected_aims,
        model_predictive_pis=model_predictive_pis,
        model_predictive_policies=model_predictive_policies,
        model_predictive_dirs=model_predictive_dirs,
    )


def processAndPlotSingle(i, data, rawX, negll, plotIdentifier):
    """Diagnostic forward pass for HMM: compute all state/prediction arrays and plot.

    Parameters
    ----------
    i : int
        Participant index
    data : tuple
        (aims, rots, targetPositions, pp, annealing_mode[, scale_S0, nu_S0])
    rawX : ndarray
        Fitted parameters (6,)
    negll : float
        Negative log-likelihood
    plotIdentifier : str
        Base path for plots

    Returns
    -------
    dict
        Diagnostic results including RMSE, R², BIC, state trajectories, etc.
    """
    if len(data) >= 7:
        aims, rots, targetPositions, pp, annealing_mode, scale_S0, nu_S0 = data[:7]
    else:
        aims, rots, targetPositions, pp, annealing_mode = data
        scale_S0, nu_S0 = 3.0, MIN_NU

    diag = run_forward_pass(
        rawX,
        aims,
        rots,
        targetPositions,
        annealing_mode=annealing_mode,
        scale_S0=scale_S0,
        nu_S0=nu_S0,
    )

    numSamp = len(diag['errors'])
    bic = 6 * np.log(numSamp) - 2 * diag['logLikelihood'] if numSamp > 0 else 0
    """
    _plotWorker((
        i, pp, rawX, float(scale_S0), float(nu_S0),
        diag['pi_preds'], diag['model_predictive_pis'],
        diag['model_predictive_policies'], diag['model_predictive_dirs'],
        diag['human_explicits'],
    ))
    """
    return dict(
        rmse=diag['rmse'],
        rSquared=diag['rSquared'],
        bic=bic,
        mStates=diag['mStates'],
        allAims=diag['allAims'],
        pi_preds=diag['pi_preds'],
        valid_trials=diag['valid_trials'],
        model_explicits=diag['model_explicits'],
        all_model_explicits=diag['all_model_explicits'],
        human_explicits=diag['human_explicits'],
        expected_aims=diag['expected_aims'],
        model_predictive_pis=diag['model_predictive_pis'],
        model_predictive_policies=diag['model_predictive_policies'],
        model_predictive_dirs=diag['model_predictive_dirs'],
        errors=diag['errors'],
    )


def _plotWorker(plot_args):
    (
        i,
        pp,
        x,
        scale_S0,
        nu_S0,
        pi_preds,
        model_predictive_pis,
        model_predictive_policies,
        model_predictive_dirs,
        human_explicits,
    ) = plot_args
    noise = x[0]
    n_trials = len(pi_preds)
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))  # Direction probabilities
    if pi_preds:
        pi_preds_arr = np.array(pi_preds).T
        im = axs[0].imshow(pi_preds_arr, aspect='auto', cmap='viridis', extent=[0, n_trials, 0, 2])
        axs[0].set_xlabel('Trial')
        axs[0].set_ylabel('State (0: Naive, 1: Compensate)')
        axs[0].set_title('Model State Probabilities')
        plt.colorbar(im, ax=axs[0], label='Probability')
    if model_predictive_dirs:
        dirs_arr = np.array(model_predictive_dirs)[:, 1]  # Prob neg dir
        axs[1].plot(np.arange(n_trials), dirs_arr, color='blue', linewidth=2)
        axs[1].set_xlabel('Trial')
        axs[1].set_ylabel('P(Negative Direction)')
        axs[1].set_title('Model Sign/Dir Probabilities')
        axs[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    aim_edges = np.arange(-180.0, 181.0, 6.0)
    num_aim_bins = len(aim_edges) - 1
    hist_norm = np.zeros((n_trials, num_aim_bins))
    for t in range(n_trials):
        pi_pred = np.array(model_predictive_pis[t])
        full_policy = np.array(model_predictive_policies[t])  # Full policy over actions
        for k in range(num_aim_bins):
            low = aim_edges[k]
            high = aim_edges[k + 1]
            em0 = _student_t_bin_mass(low, high, 0.0, scale_S0, nu_S0)
            em1_contribs = np.array(
                [
                    _student_t_bin_mass(low, high, float(act), noise, DEFAULT_NU)
                    for act in action_centers
                ],
                dtype=np.float64,
            )
            em1 = np.dot(full_policy, em1_contribs)
            prob_k = pi_pred[0] * em0 + pi_pred[1] * em1
            hist_norm[t, k] = prob_k
    hist_norm = np.ma.masked_where(hist_norm < 1e-6, hist_norm)
    im1 = axs[2].imshow(
        hist_norm.T,
        origin='lower',
        aspect='auto',
        cmap='Greys',
        extent=[0, n_trials, -180, 180],
        interpolation='nearest',
    )
    plt.colorbar(im1, ax=axs[2], label='Probability Mass')
    axs[2].scatter(
        np.arange(n_trials),
        np.array(human_explicits),
        color='red',
        marker='x',
        s=30,
        alpha=0.8,
        label='Human Aim',
    )
    axs[2].axhline(y=0, color='green', linewidth=2)
    axs[2].set_xlabel('Trial')
    axs[2].set_ylabel('Aim (degrees)')
    axs[2].set_title('Model Predictive Density vs Human Aim')
    legend_elements1 = [
        Patch(facecolor='gray', alpha=0.5, label='Model Predictive Density'),
        Line2D(
            [0], [0], marker='x', color='red', linestyle='None', markersize=8, label='Human Aim'
        ),
    ]
    axs[2].legend(handles=legend_elements1, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'tempFigures/{pp}_{i}', dpi=200)
    plt.close()


# annealing_mode: 0 none, 1 linear, 2 state-weighted
class FitShell:
    """HMM fitting shell with CMA-ES infrastructure.

    Parameters
    ----------
    df : DataFrame
        Input data (must have columns: participantNum, phase, blockNum, aim, rotation, targetPosition)
    conVal : str
        Condition value to filter by (if condition != 'none')
    condition : str
        Condition column name to filter by
    fitPhase : str or None
        Phase to fit ('rotation', 'baseline', etc.). If None, fit all trials.
    flipRot : bool
        Whether to flip rotation sign
    annealing_mode : int
        0: no annealing, 1: typical (linear), 2: principled (beta*pi_pred[1])
    numCores : int
        Number of parallel cores for fitting and plotting
    popSizeMultiplier : float
        Scaling factor for CMA-ES restarts (40 * popSizeMultiplier)
    datasetName : str
        Directory name for caching fitted parameters. If left as 'default',
        a stable dataset-specific cache folder is derived automatically.
    quickPlots : bool
        Whether to generate plots during fitting
    fitWashout : bool
        Whether to include washout trials in fitting. Defaults to True to
        preserve the historical HMM behaviour outside Ding-specific fits.
    """

    def __init__(
        self,
        df='none',
        conVal='none',
        condition='none',
        fitPhase='rotation',
        flipRot=False,
        annealing_mode=0,
        numCores=mp.cpu_count() // 2,
        popSizeMultiplier=1,
        datasetName='default',
        quickPlots=False,
        fitWashout=True,
    ):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.flipRot = flipRot
        self.annealing_mode = annealing_mode
        self.numCores = numCores
        self.popSizeMultiplier = popSizeMultiplier
        self.datasetName = datasetName
        self.quickPlots = quickPlots
        self.fitWashout = fitWashout
        self.participantNums = []
        self.xs = []
        self.negLl = []
        self.bics = []
        self.rmses = []
        self.rSquareds = []
        self.mStates = []
        self.allAims = []
        self.pi_preds = []
        self.valid_trials = []
        self.model_explicits = []
        self.all_model_explicits = []
        self.human_explicits = []
        self.expected_aims = []
        self.model_predictive_pis = []
        self.model_predictive_policies = []
        self.model_predictive_dirs = []
        self.baselineScale = []
        self.baselineNu = []
        self.errors = []

    def fitRot(self, numCores=None):
        """Fit HMM to all participants using CMA-ES optimization.

        Parameters
        ----------
        numCores : int or None
            Number of cores to use. If None, use self.numCores.
        """
        if numCores is None:
            numCores = self.numCores

        self.dat = self.df
        if self.condition != 'none':
            pInCond = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(pInCond)]
        uniqP = self.dat['participantNum'].unique()
        self.participantNums = uniqP
        N = len(uniqP)
        if N == 0:
            return
        resolved_dataset_name = _resolve_cache_dir(
            self.datasetName,
            self.dat,
            self.condition,
            self.conVal,
            self.fitPhase,
            self.annealing_mode,
            self.flipRot,
            self.fitWashout,
        )
        if resolved_dataset_name != self.datasetName:
            print(f"Using HMM cache directory: {resolved_dataset_name}")
            self.datasetName = resolved_dataset_name

        self.bics = np.zeros(N)
        self.negLl = np.zeros(N)
        self.rmses = np.zeros(N)
        self.rSquareds = np.zeros(N)
        self.xs = []
        self.errors = []
        self.mStates = [[] for _ in range(N)]
        self.allAims = [[] for _ in range(N)]
        self.pi_preds = [[] for _ in range(N)]
        self.valid_trials = [[] for _ in range(N)]
        self.model_explicits = [[] for _ in range(N)]
        self.all_model_explicits = [[] for _ in range(N)]
        self.human_explicits = [[] for _ in range(N)]
        self.expected_aims = [[] for _ in range(N)]
        self.model_predictive_pis = [[] for _ in range(N)]
        self.model_predictive_policies = [[] for _ in range(N)]
        self.model_predictive_dirs = [[] for _ in range(N)]
        self.baselineScale = np.zeros(N)
        self.baselineNu = np.zeros(N)

        dataList = []
        for i_pp, pp in enumerate(uniqP):
            pDat_full = self.dat[self.dat['participantNum'] == pp].copy()
            pDat_full = _select_first_block(pDat_full)
            scale_S0, nu_S0 = _get_baseline_student_t_from_df(pDat_full)
            self.baselineScale[i_pp] = scale_S0
            self.baselineNu[i_pp] = nu_S0
            phases_full = _get_phase_labels_from_df(pDat_full)
            if self.fitWashout:
                pDat_work = pDat_full.reset_index(drop=True)
                phases_work = phases_full
            else:
                keep_mask = np.array([str(p).lower() != 'washout' for p in phases_full], dtype=bool)
                pDat_work = pDat_full.loc[keep_mask].reset_index(drop=True)
                phases_work = phases_full[keep_mask]
            if self.fitPhase is not None:
                phase_mask = np.array(
                    [str(p).lower() == str(self.fitPhase).lower() for p in phases_work], dtype=bool
                )
                pDat = pDat_work.loc[phase_mask].reset_index(drop=True)
            else:
                pDat = pDat_work
            aims = pDat['aim'].values.astype(np.float64)
            rots = pDat['rotation'].values.astype(np.float64)
            tPos = pDat['targetPosition'].values.astype(np.float64)
            if self.flipRot:
                rots = -rots
            dataList.append((aims, rots, tPos, pp, self.annealing_mode, scale_S0, nu_S0))

        boundsSingle = [
            (1.0, 30.0),  # noise
            (1e-300, 0.05),  # kappa
            (0.1, 100.0),  # noise_context
            (1e-6, 1.0),  # alpha
            (1e-300, 2.0),  # beta
            (1e-300, 1.0),  # beta_dir
        ]

        os.makedirs(self.datasetName, exist_ok=True)

        print(
            f"Starting CMA-ES fitting for {N} participants (annealing_mode={self.annealing_mode})..."
        )
        t_start = time.time()
        with mp.Pool(processes=numCores) as pool:
            results = pool.starmap(
                fitSingle,
                [
                    (dataList[i], boundsSingle, self.popSizeMultiplier, self.datasetName)
                    for i in range(N)
                ],
            )

        indivParams = np.array([r[0] for r in results])
        currentNeglls = np.array([r[2] for r in results])
        self.xs = [list(indivParams[i]) for i in range(N)]
        self.negLl = currentNeglls
        elapsed_fit = time.time() - t_start
        print(f"CMA-ES fitting completed in {elapsed_fit / 60:.1f} minutes.")

        print(f"Running diagnostic passes and generating plots...")
        os.makedirs("tempFigures", exist_ok=True)
        t_plot_start = time.time()
        with mp.Pool(processes=numCores) as pool:
            plotResults = pool.starmap(
                processAndPlotSingle,
                [
                    (i, dataList[i], indivParams[i], currentNeglls[i], self.datasetName)
                    for i in range(N)
                ],
            )

        for i, res in enumerate(plotResults):
            self.bics[i] = res['bic']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']
            self.errors.append(res.get('errors', np.array([])))
            self.mStates[i] = res['mStates']
            self.allAims[i] = res['allAims']
            self.pi_preds[i] = res['pi_preds']
            self.valid_trials[i] = res['valid_trials']
            self.model_explicits[i] = res['model_explicits']
            self.all_model_explicits[i] = res['all_model_explicits']
            self.human_explicits[i] = res['human_explicits']
            self.expected_aims[i] = res['expected_aims']
            self.model_predictive_pis[i] = res['model_predictive_pis']
            self.model_predictive_policies[i] = res['model_predictive_policies']
            self.model_predictive_dirs[i] = res['model_predictive_dirs']

        elapsed_plot = time.time() - t_plot_start
        print(f"Diagnostic passes and plotting completed in {elapsed_plot / 60:.1f} minutes.")
