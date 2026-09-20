import numba as nb
from numba import njit, float64, int64, types
from numba.typed import Dict
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import numpy as np
import scipy.stats
from scipy.special import logsumexp as sp_logsumexp
import multiprocessing
import os
import pickle
import time
import cma

np.random.seed(42)
SEED = 42
EPS = 1e-12
LOGZERO = -1e300
TWO_PI = 2.0 * math.pi

N_HARM = 1
DIM = 2 * N_HARM + 1
N_GROUPS = N_HARM + 1

MAX_TRAIN_TARGETS = 10

ENTROPY_GRID_N = 361
ENTROPY_DX = 360.0 / ENTROPY_GRID_N  # circular domain: N points on [-180, 180)

VIS_COEFF = 0.1  # Default observation scale per degree of perturbation.

# M1 uses fixed heavy tails; M0 degrees of freedom are fitted per participant.
DEFAULT_NU = 5.0
MIN_NU = 10.0  # minimum df to ensure finite kurtosis
MAX_BASELINE_NU = 1000.0  # above this the Student-t is effectively Gaussian here
GAUSSIAN_APPROX_NU = 1000.0
_MAX_BASELINE_LOG_NU_SPAN = math.log(MAX_BASELINE_NU - MIN_NU)
MIN_SCALE = 1.0  # minimum scale in degrees
MODEL_VERSION = 'beta_component_v9'
CACHE_FAILURE_NLL = 1e11

A_HZ = 1.0
B_HZ = 100.0

MAXRUN = 3

# Hypotheses interpolate from pure harmonic to pure DC at equal predictive variance.
N_STRUCT = 2

# Max mixture components: N_STRUCT × (M0 + MAXRUN runs × 2 [self + other])
MAX_COMP = N_STRUCT * (1 + MAXRUN * 2)
DEFAULT_MODEL_CONFIG = {
    'n_harm': int(N_HARM),
    'a_hz': float(A_HZ),
    'b_hz': float(B_HZ),
    'maxrun': int(MAXRUN),
    'n_struct': int(N_STRUCT),
}


def _configure_globals(n_harm=None, a_hz=None, b_hz=None, maxrun=None, n_struct=None):
    """
    Update module-level configuration constants and recompile all affected
    Numba functions.
    """
    global N_HARM, DIM, N_GROUPS, A_HZ, B_HZ, MAXRUN, MAX_COMP, N_STRUCT

    changed = False

    if n_harm is not None and n_harm != N_HARM:
        N_HARM = n_harm
        DIM = 2 * N_HARM + 1
        N_GROUPS = N_HARM + 1
        changed = True
    if a_hz is not None and a_hz != A_HZ:
        A_HZ = a_hz
        changed = True
    if b_hz is not None and b_hz != B_HZ:
        B_HZ = b_hz
        changed = True
    if maxrun is not None and maxrun != MAXRUN:
        MAXRUN = maxrun
        changed = True
    if n_struct is not None and n_struct != N_STRUCT:
        N_STRUCT = n_struct
        changed = True

    MAX_COMP = N_STRUCT * (1 + MAXRUN * 2)

    # Numba captures module constants at compilation time.
    if changed:
        for fn in [
            featureVector,
            buildPriorDiag,
            buildStructPriorDiags,
            precomputeVecs,
            computePriorFSF,
            computeNegLl,
            samplePredictive,
            computeGibbsWeights,
            looGlobalFieldPred,
            safeFSf,
            computeLogZM1,
            temperStudentTParams,
        ]:
            fn.recompile()


def _current_model_config():
    return {
        'n_harm': int(N_HARM),
        'a_hz': float(A_HZ),
        'b_hz': float(B_HZ),
        'maxrun': int(MAXRUN),
        'n_struct': int(N_STRUCT),
    }


def formatFittedParams(xs):
    (
        scale_S0,
        nu_S0,
        logNegLogH,
        logVarTrans,
        logAlpha,
        kappa,
        logPriorOddsStruct,
        logVisCoeff,
        logBeta,
    ) = xs
    logH = -np.exp(np.clip(logNegLogH, -700, 700))
    h = np.exp(np.clip(logH, -700, 0))
    la = np.clip(logAlpha, -700, 700)
    alpha_self = np.log(1.0 + np.exp(-la))
    alpha_other = np.log(1.0 + np.exp(la))
    visCoeff = np.exp(np.clip(logVisCoeff, -700, 700))
    beta = np.exp(np.clip(logBeta, -700, 700))
    bias = "other" if logAlpha > 0 else ("self" if logAlpha < 0 else "balanced")
    return (
        f"scale_S0={scale_S0:.3f}, nu_S0={nu_S0:.2f}, "
        f"logNegLogH={logNegLogH:.2f} (logH={logH:.2f}, h={h:.2e}), "
        f"logVarTrans={logVarTrans:.2f}, "
        f"logAlpha={logAlpha:.4f} (α_self={alpha_self:.4f}, α_other={alpha_other:.4f}, {bias}), "
        f"kappa={kappa:.4f}, "
        f"logPriorOddsStruct={logPriorOddsStruct:.4f}, "
        f"logVisCoeff={logVisCoeff:.4f} (visCoeff={visCoeff:.4f}), "
        f"logBeta={logBeta:.4f} (beta={beta:.4f})"
    )


@njit(cache=False, fastmath=False)
def wrapAngle(x):
    return (x + 180) % 360 - 180


@njit(cache=False, fastmath=False)
def safeExp(x):
    return np.exp(np.minimum(np.maximum(x, -700.0), 700.0))


@njit(cache=False, fastmath=False)
def safeLog(x):
    if x <= 0.0:
        return -1e9
    return math.log(x)


@njit(cache=False, fastmath=False)
def logAddExp(a, b):
    if a == b:
        return a + math.log(2.0)
    maxVal = max(a, b)
    minVal = min(a, b)
    if maxVal - minVal > 700.0:
        return maxVal
    return maxVal + math.log(1.0 + math.exp(minVal - maxVal))


@njit(cache=False, fastmath=False)
def logSumExp(arr):
    n = len(arr)
    if n == 0:
        return -1e9
    maxV = -np.inf
    hasInf = False
    hasNan = False
    for x in arr:
        if np.isnan(x):
            hasNan = True
        elif np.isinf(x) and x > 0:
            hasInf = True
        elif x > maxV:
            maxV = x
    if hasInf:
        return np.inf
    if hasNan:
        return np.nan
    if not math.isfinite(maxV):
        return -1e9
    if maxV < -700.0:
        return maxV
    s = 0.0
    for x in arr:
        if np.isfinite(x):
            s += safeExp(x - maxV)
    if s == 0.0:
        return maxV
    return maxV + safeLog(s)


@njit(cache=False, fastmath=False)
def logGaussianPdf(x, mu, sigma):
    if sigma <= 0.0:
        return -1e9
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


@njit(cache=False, fastmath=False)
def logWrappedGaussianPdf(x, mu, sigma, period=360.0, max_iter=1):
    if sigma <= 0.0:
        return -1e9
    terms_list = nb.typed.List()
    term = logGaussianPdf(x, mu, sigma)
    terms_list.append(term)
    for k in range(1, max_iter + 1):
        terms_list.append(logGaussianPdf(x - mu + period * k, 0.0, sigma))
        terms_list.append(logGaussianPdf(x - mu - period * k, 0.0, sigma))
    arr = np.empty(len(terms_list), dtype=np.float64)
    for i in range(len(terms_list)):
        arr[i] = terms_list[i]
    return logSumExp(arr)


@njit(cache=False, fastmath=False)
def logStudentTPdf(x, mu, sigma, nu):
    """Log-pdf of Student's t with location mu, scale sigma, df nu.
    Parameterised so that sigma is the scale (not std); variance = sigma^2 * nu/(nu-2)."""
    if sigma <= 0.0:
        return -1e9
    if nu >= GAUSSIAN_APPROX_NU:
        return logGaussianPdf(x, mu, sigma)
    z = (x - mu) / sigma
    return (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - math.log(sigma)
        - 0.5 * (nu + 1.0) * math.log(1.0 + z * z / nu)
    )


@njit(cache=False, fastmath=False)
def logWrappedStudentTPdf(x, mu, sigma, nu, period=360.0, max_iter=1):
    """Wrapped Student's t — primary image only (wrap correction < 1e-7)."""
    if sigma <= 0.0:
        return -1e9
    dx = ((x - mu + 180.0) % 360.0) - 180.0
    if nu >= GAUSSIAN_APPROX_NU:
        return logGaussianPdf(dx, 0.0, sigma)
    z = dx / sigma
    return (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - math.log(sigma)
        - 0.5 * (nu + 1.0) * math.log(1.0 + z * z / nu)
    )


@njit(cache=False, fastmath=False)
def log1mexp(x):
    if x > 0.0:
        return LOGZERO
    if x == 0.0:
        return LOGZERO
    if np.isinf(x) and x < 0:
        return 0.0
    cutoff = -math.log(2.0)
    if x > cutoff:
        return math.log(-math.expm1(x))
    else:
        return math.log1p(-math.exp(x))


@njit(cache=False, fastmath=False)
def safeNormaliseExp(log_vals, log_sum):
    if log_sum == LOGZERO:
        return np.zeros_like(log_vals)
    clipped = np.minimum(np.maximum(log_vals - log_sum, -700.0), 0.0)
    return np.exp(clipped)


@njit(cache=False, fastmath=False)
def normalCdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@njit(cache=False, fastmath=False)
def featureVector(theta_rad):
    f = np.empty(DIM)
    f[0] = 1.0
    for h in range(1, N_HARM + 1):
        f[2 * h - 1] = math.sin(h * theta_rad)
        f[2 * h] = math.cos(h * theta_rad)
    return f


@njit(cache=False, fastmath=False)
def buildPriorDiag(logVarTrans):
    s2 = safeExp(logVarTrans)
    sigma_T = math.sqrt(s2)
    RAD2DEG_SQ = (180.0 / math.pi) ** 2

    diag = np.zeros(DIM)
    # DC = 0: under the clean formulation, the base prior is pure harmonic.
    # DC variance is assigned only under H_rot in buildStructPriorDiags.

    N_QUAD = 300
    eps_max = max(6.0 * sigma_T, 3.0)
    d_eps = eps_max / N_QUAD

    for n in range(1, N_HARM + 1):
        integral = 0.0
        for qi in range(1, N_QUAD + 1):
            eps = qi * d_eps
            p_eps = (eps / s2) * math.exp(-eps * eps / (2.0 * s2))
            if eps <= 1.0:
                hn_sq = eps ** (2 * n)
            else:
                eps_neg_n = eps ** (-n)
                hn_sq = (2.0 - eps_neg_n) ** 2
            integral += hn_sq * p_eps * d_eps
        v_n = RAD2DEG_SQ * integral / (2.0 * n**2)
        diag[2 * n - 1] = v_n
        diag[2 * n] = v_n

    return diag


@njit(cache=False, fastmath=False)
def buildStructPriorDiags(logVarTrans):
    """
    Build N_STRUCT prior diagonal arrays by variance-conserving interpolation.
    Returns array of shape (N_STRUCT, DIM).
    """
    base = buildPriorDiag(logVarTrans)

    # Each sin/cos pair contributes one v_n because sin² + cos² = 1.
    harm_pred_sum = 0.0
    for n in range(1, N_HARM + 1):
        harm_pred_sum += base[2 * n - 1]  # v_n (sin and cos entries equal)

    result = np.zeros((N_STRUCT, DIM))
    for m in range(N_STRUCT):
        if N_STRUCT > 1:
            lam = float(m) / float(N_STRUCT - 1)
        else:
            lam = 0.0
        result[m, 0] = lam * harm_pred_sum
        for d in range(1, DIM):
            result[m, d] = (1.0 - lam) * base[d]
    return result


@njit(cache=False, fastmath=False)
def scalarBLRUpdate(c, d, s, y, obsVar):
    q = 1.0 - c * s
    if q < 1e-30:
        q = 1e-30
    denom = q * s + obsVar
    if denom < 1e-20:
        denom = 1e-20
    c_new = c + q * q / denom
    d_new = d + q * (y - d * s) / denom
    return c_new, d_new


@njit(cache=False, fastmath=False)
def scalarBLRMerge(c_a, d_a, c_b, d_b, wA):
    wB = 1.0 - wA
    d_out = wA * d_a + wB * d_b
    c_out = wA * c_a + wB * c_b - wA * wB * (d_a - d_b) * (d_a - d_b)
    return c_out, d_out


@njit(cache=False, fastmath=False)
def rescaleBLRScalars(cRun, dRun, sVals_old, sVals_new, K, runLimit):
    """
    After an IG update changes priorDiag (and hence sVals), recompute the
    per-target (c, d) scalars to give the EXACT Bayesian posterior under
    the new prior.

    The data sufficient statistics are prior-independent:
        S = c / q          (accumulated data precision)
        R = d / q          (accumulated data mean)
    where q = 1 - c * s_old.

    Under the new prior (s_new = f_j' Sigma_0_new f_j), the exact
    posterior is:
        c_new = S / (1 + S * s_new)
        d_new = R / (1 + S * s_new)
    """
    for k in range(runLimit):
        for j in range(K):
            s_old = sVals_old[j]
            s_new = sVals_new[j]
            if s_new < 1e-30 or s_old < 1e-30:
                continue
            q_old = 1.0 - cRun[k, j] * s_old
            if q_old < 1e-30:
                q_old = 1e-30
            # Extract prior-independent data sufficient statistics
            S = cRun[k, j] / q_old
            R = dRun[k, j] / q_old
            # Exact posterior under new prior
            denom = 1.0 + S * s_new
            if denom < 1e-30:
                denom = 1e-30
            cRun[k, j] = S / denom
            dRun[k, j] = R / denom


@njit(cache=False, fastmath=False)
def _solve3x3(A, b):
    """Solve 3x3 symmetric system Ax=b via Cramer's rule. Returns (x, A_inv)."""
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a11, a12 = A[1, 1], A[1, 2]
    a22 = A[2, 2]
    # Symmetric: a10=a01, a20=a02, a21=a12
    det = (
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02)
    )
    if abs(det) < 1e-30:
        x = np.zeros(3)
        Ainv = np.zeros((3, 3))
        return x, Ainv, False
    inv_det = 1.0 / det
    Ainv = np.zeros((3, 3))
    Ainv[0, 0] = (a11 * a22 - a12 * a12) * inv_det
    Ainv[0, 1] = (a02 * a12 - a01 * a22) * inv_det
    Ainv[0, 2] = (a01 * a12 - a02 * a11) * inv_det
    Ainv[1, 0] = Ainv[0, 1]
    Ainv[1, 1] = (a00 * a22 - a02 * a02) * inv_det
    Ainv[1, 2] = (a02 * a01 - a00 * a12) * inv_det
    Ainv[2, 0] = Ainv[0, 2]
    Ainv[2, 1] = Ainv[1, 2]
    Ainv[2, 2] = (a00 * a11 - a01 * a01) * inv_det
    x = np.zeros(3)
    for r in range(3):
        for c in range(3):
            x[r] += Ainv[r, c] * b[c]
    return x, Ainv, True


@njit(cache=False, fastmath=False)
def _solve_spd(A, b):
    """Solve a symmetric positive-definite system via Cholesky factorisation.

    Returns
    -------
    x : (n,) solution to Ax=b
    Ainv : (n, n) inverse of A
    ok : bool indicating whether the decomposition succeeded
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    # Cholesky: A = L L^T
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-20:
                    return np.zeros(n), np.zeros((n, n)), False
                L[i, j] = math.sqrt(s)
            else:
                if abs(L[j, j]) < 1e-30:
                    return np.zeros(n), np.zeros((n, n)), False
                L[i, j] = s / L[j, j]

    # Forward solve: L y = b
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        if abs(L[i, i]) < 1e-30:
            return np.zeros(n), np.zeros((n, n)), False
        y[i] = s / L[i, i]

    # Back solve: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k, i] * x[k]
        if abs(L[i, i]) < 1e-30:
            return np.zeros(n), np.zeros((n, n)), False
        x[i] = s / L[i, i]

    # Invert via solves against basis vectors.
    Ainv = np.zeros((n, n))
    y_col = np.zeros(n)
    x_col = np.zeros(n)
    for col in range(n):
        for i in range(n):
            s = 1.0 if i == col else 0.0
            for k in range(i):
                s -= L[i, k] * y_col[k]
            y_col[i] = s / L[i, i]
        for i in range(n - 1, -1, -1):
            s = y_col[i]
            for k in range(i + 1, n):
                s -= L[k, i] * x_col[k]
            x_col[i] = s / L[i, i]
        for i in range(n):
            Ainv[i, col] = x_col[i]

    # Numerical symmetry clean-up.
    for i in range(n):
        for j in range(i):
            avg = 0.5 * (Ainv[i, j] + Ainv[j, i])
            Ainv[i, j] = avg
            Ainv[j, i] = avg

    return x, Ainv, True


@njit(cache=False, fastmath=False)
def looGlobalFieldPred(fVecs, allPriorDiag_m, dPred, cPred, sVals, gibbsW_row, crossDot, i_idx, K):
    """Compute LOO global field prediction at target i.

    Uses the FULL feature space (DC + harmonics) so that H_rot (pure DC)
    can propagate cross-target information through the DC component, and
    H_trans (pure harmonic) through the harmonic components.  The prior
    precision naturally gates which dimensions are active: dimensions with
    zero prior variance get prior_prec → 1e30, locking them at 0.

    Parameters
    ----------
    fVecs : (K, DIM) feature vectors [1, cos(θ), sin(θ), ...]
    allPriorDiag_m : (DIM,) prior diagonal for hypothesis m
    dPred, cPred : (K,) per-target BLR posterior scalars
    sVals : (K,) self-dot products f_j^T diag(prior) f_j
    gibbsW_row : (K,) Gibbs weights G(i, j) for fixed i
    crossDot : (K, K) cross-dot products
    i_idx : target index to predict at
    K : number of targets

    Returns
    -------
    pred_other : LOO field prediction at target i
    var_other : LOO field predictive variance at target i
    """
    prior_prec_diag = np.zeros(DIM)
    for d in range(DIM):
        pd = allPriorDiag_m[d]
        if pd > 1e-30:
            prior_prec_diag[d] = 1.0 / pd
        else:
            prior_prec_diag[d] = 1e30  # dimension locked by prior

    A = np.zeros((DIM, DIM))
    for d in range(DIM):
        A[d, d] = prior_prec_diag[d]
    b = np.zeros(DIM)

    # Normalize Gibbs weights over observed j != i.
    # Only observed targets (cPred[j] > 0) contribute.
    wSum = 0.0
    for j in range(K):
        if j != i_idx and cPred[j] > 0.0:
            wSum += gibbsW_row[j]
    if wSum < 1e-30:
        # No observed targets besides i — fall back to prior
        return 0.0, sVals[i_idx]

    for j in range(K):
        if j == i_idx:
            continue
        if cPred[j] <= 0.0:
            continue
        w_j = gibbsW_row[j] / wSum
        if w_j < 1e-30:
            continue

        delta_j = dPred[j] * sVals[j]

        q_j = 1.0 - cPred[j] * sVals[j]
        if q_j < 1e-10:
            q_j = 1e-10
        eff_var_j = q_j * sVals[j]
        if eff_var_j < 1e-10:
            eff_var_j = 1e-10

        precision_j = w_j / eff_var_j

        for d1 in range(DIM):
            for d2 in range(DIM):
                A[d1, d2] += precision_j * fVecs[j, d1] * fVecs[j, d2]
            b[d1] += precision_j * fVecs[j, d1] * delta_j

    if DIM == 3:
        beta, Ainv, ok = _solve3x3(A, b)
        if not ok:
            return 0.0, sVals[i_idx]
    else:
        beta, Ainv, ok = _solve_spd(A, b)
        if not ok:
            return 0.0, sVals[i_idx]

    fi = fVecs[i_idx]
    pred_other = 0.0
    for d in range(DIM):
        pred_other += fi[d] * beta[d]

    var_other = 0.0
    for d1 in range(DIM):
        for d2 in range(DIM):
            var_other += fi[d1] * Ainv[d1, d2] * fi[d2]
    return pred_other, max(var_other, 1e-10)


@njit(cache=False, fastmath=False)
def precomputeVecs(fVecs, priorDiag, K):
    vVecs = np.zeros((K, DIM))
    for j in range(K):
        for d in range(DIM):
            vVecs[j, d] = priorDiag[d] * fVecs[j, d]

    crossDot = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            s = 0.0
            for d in range(DIM):
                s += fVecs[i, d] * vVecs[j, d]
            crossDot[i, j] = s

    sVals = np.zeros(K)
    for j in range(K):
        sVals[j] = crossDot[j, j]

    return vVecs, crossDot, sVals


@njit(cache=False, fastmath=False)
def computeGibbsWeights(crossDot, sVals, K, kappa):
    gibbsW = np.zeros((K, K))
    for i in range(K):
        pv_i = max(sVals[i], 1e-30)
        for j in range(K):
            pv_j = max(sVals[j], 1e-30)
            r2 = crossDot[i, j] * crossDot[i, j] / (pv_i * pv_j)
            gibbsW[i, j] = safeExp(kappa * r2)
    return gibbsW


@njit(cache=False, fastmath=False)
def computePriorFSF(priorDiag, f):
    s = 0.0
    for d in range(DIM):
        s += f[d] * priorDiag[d] * f[d]
    return s


@njit(cache=False, fastmath=False)
def safeFSf(cd_ii, c_j, cd_ij):
    """Predictive variance: cd_ii - c_j * cd_ij².

    The result is bounded in [0, cd_ii] by Cauchy-Schwarz and the
    BLR constraint c_j * sVals[j] < 1.
    """
    return max(cd_ii - c_j * cd_ij * cd_ij, 0.0)


# Dense grids around component means resolve narrow peaks; a coarse grid covers overlap.
_ZM1_N_LOCAL = 21  # dense points per component (odd, centred on mean)
_ZM1_N_COARSE = 18  # coarse background points


@njit(cache=False, fastmath=False)
def computeLogZM1(m1_mean, m1_scale, m1_invscale, m1_base, nM1, beta, logP_M1, _half_nup1, _inv_nu):
    """Compute log Z_M1 = log ∫ f_M1(a)^β da via component-seeded quadrature."""
    sqrt_beta = math.sqrt(beta)
    max_pts = _ZM1_N_LOCAL * nM1 + _ZM1_N_COARSE + 10

    grid_x = np.empty(max_pts)
    n_pts = 0
    for c in range(nM1):
        mu_c = m1_mean[c]
        sigma_eff = m1_scale[c] / sqrt_beta
        half_width = max(6.0 * sigma_eff, 1.0)
        dx_local = 2.0 * half_width / (_ZM1_N_LOCAL - 1)
        for j in range(_ZM1_N_LOCAL):
            x = mu_c - half_width + j * dx_local
            x = ((x + 180.0) % 360.0) - 180.0
            if n_pts < max_pts:
                grid_x[n_pts] = x
                n_pts += 1

    for j in range(_ZM1_N_COARSE):
        x = -180.0 + j * (360.0 / _ZM1_N_COARSE)
        if n_pts < max_pts:
            grid_x[n_pts] = x
            n_pts += 1

    for i in range(1, n_pts):
        key = grid_x[i]
        j = i - 1
        while j >= 0 and grid_x[j] > key:
            grid_x[j + 1] = grid_x[j]
            j -= 1
        grid_x[j + 1] = key

    # 4. Deduplicate (within 0.05°)
    dedup_x = np.empty(n_pts)
    dedup_x[0] = grid_x[0]
    n_dedup = 1
    for i in range(1, n_pts):
        if grid_x[i] - dedup_x[n_dedup - 1] > 0.05:
            dedup_x[n_dedup] = grid_x[i]
            n_dedup += 1

    log_g = np.empty(n_dedup)
    log_g_max = -1e30
    for i in range(n_dedup):
        x = dedup_x[i]
        logB = LOGZERO
        for c in range(nM1):
            dx = ((x - m1_mean[c] + 180.0) % 360.0) - 180.0
            z = dx * m1_invscale[c]
            lp = m1_base[c] - _half_nup1 * math.log(1.0 + z * z * _inv_nu)
            logB = logAddExp(logB, lp)
        log_g[i] = beta * (logB - logP_M1)
        if log_g[i] > log_g_max:
            log_g_max = log_g[i]

    # 6. Non-uniform trapezoidal rule (shifted for stability)
    integral = 0.0
    for i in range(n_dedup - 1):
        dx = dedup_x[i + 1] - dedup_x[i]
        g_left = safeExp(log_g[i] - log_g_max)
        g_right = safeExp(log_g[i + 1] - log_g_max)
        integral += 0.5 * dx * (g_left + g_right)

    # Circular closure: gap from last point to first + 360°
    dx_wrap = (dedup_x[0] + 360.0) - dedup_x[n_dedup - 1]
    if dx_wrap > 0.0 and dx_wrap < 360.0:
        g_left = safeExp(log_g[n_dedup - 1] - log_g_max)
        g_right = safeExp(log_g[0] - log_g_max)
        integral += 0.5 * dx_wrap * (g_left + g_right)

    if integral > 0.0:
        return log_g_max + math.log(integral)
    return LOGZERO


@njit(cache=False, fastmath=False)
def temperStudentTParams(sigma, nu, beta):
    """Exact normalized power transform for an unwrapped Student-t kernel.

    If p(x) is Student-t(sigma, nu), then p(x)^beta / Z is again Student-t with
        nu_beta = beta * (nu + 1) - 1
        sigma_beta^2 = sigma^2 * nu / nu_beta

    We apply the same transform to the wrapped readout kernels used in BHT.
    """
    if sigma <= 0.0:
        return 1e-10, max(nu, 1e-6)
    if beta <= 0.0:
        return sigma, max(nu, 1e-6)
    nu_beta = beta * (nu + 1.0) - 1.0
    if nu_beta <= 1e-6:
        nu_beta = 1e-6
    sigma_beta = sigma * math.sqrt(max(nu, 1e-6) / nu_beta)
    return max(sigma_beta, 1e-10), nu_beta


@njit(cache=False, fastmath=False)
def findTargetIdx(destTargets, nDest, target):
    for j in range(nDest):
        if abs(destTargets[j] - target) < 0.01:
            return j
    return -1


def fitBaselineStudentT(bl_aims):
    """Fit Student's t (location=0, scale, df) to baseline aim data via MLE.

    Returns (scale, nu) with scale >= MIN_SCALE, nu >= MIN_NU.
    Location is fixed at 0 (no directional bias expected at baseline).
    """
    from scipy.optimize import minimize as sp_minimize

    bl = bl_aims[~np.isnan(bl_aims)]
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
            ll = -0.5 * z * z - np.log(scale) - 0.5 * np.log(2.0 * np.pi)
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
    for nu_init in [MIN_NU + 0.1, MIN_NU + 20.0, MIN_NU + 90.0]:
        x0 = [np.log(mad_scale), np.log(max(nu_init - MIN_NU, 1e-6))]
        try:
            res = sp_minimize(
                negll,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                sc = max(np.exp(np.clip(res.x[0], -700.0, 700.0)), MIN_SCALE)
                nu = MIN_NU + np.exp(np.clip(res.x[1], -700.0, _MAX_BASELINE_LOG_NU_SPAN))
                best_params = (sc, nu)
        except Exception:
            continue
    return best_params


@njit(cache=False, fastmath=False)
def computeNegLl(
    params,
    allAims,
    mask,
    hasFeedback,
    trials,
    isRotation,
    rotations,
    targets,
    uniqueThetas,
    scale_S0,
    nu_S0,
    destTargets,
    nDestTargets,
    computeEntropy=False,
):
    logNegLogH, logVarTrans, logAlpha, kappa, logPriorOddsStruct, logVisCoeff, logBeta = params
    logH = -safeExp(logNegLogH)
    visCoeff = safeExp(logVisCoeff)
    beta = safeExp(logBeta)
    _half_nup1 = 0.5 * (DEFAULT_NU + 1.0)  # (ν+1)/2
    _inv_nu = 1.0 / DEFAULT_NU
    _logNormNu = (
        math.lgamma(_half_nup1)
        - math.lgamma(0.5 * DEFAULT_NU)
        - 0.5 * math.log(DEFAULT_NU * math.pi)
    )
    _do_temper = abs(beta - 1.0) > 1e-10  # skip Z_M1 quadrature when beta ≈ 1
    var_S0 = scale_S0**2
    sigmaAim2 = var_S0
    numTrials = len(trials)
    K = nDestTargets
    deg2rad = math.pi / 180.0
    # Softplus split: logAlpha > 0 → other-biased, logAlpha < 0 → self-biased
    if logAlpha > 20.0:
        _alpha_self = math.exp(-logAlpha)
        _alpha_other = logAlpha
    elif logAlpha < -20.0:
        _alpha_self = -logAlpha
        _alpha_other = math.exp(logAlpha)
    else:
        _alpha_self = math.log(1.0 + math.exp(-logAlpha))
        _alpha_other = math.log(1.0 + math.exp(logAlpha))
    M = N_STRUCT  # number of structure hypotheses
    allPriorDiag = buildStructPriorDiags(logVarTrans)

    fVecs = np.zeros((K, DIM))
    for i in range(K):
        fi = featureVector(destTargets[i] * deg2rad)
        for d in range(DIM):
            fVecs[i, d] = fi[d]

    crossDot_all = np.zeros((M, K, K))
    sVals_all = np.zeros((M, K))
    gibbsW_all = np.zeros((M, K, K))
    for m in range(M):
        _, cd_m, sv_m = precomputeVecs(fVecs, allPriorDiag[m], K)
        gw_m = computeGibbsWeights(cd_m, sv_m, K, kappa)
        for i in range(K):
            sVals_all[m, i] = sv_m[i]
            for j in range(K):
                crossDot_all[m, i, j] = cd_m[i, j]
                gibbsW_all[m, i, j] = gw_m[i, j]

    # Structure log-prior
    # logPriorOddsStruct > 0  →  harmonic (mTrans) favoured
    # logPriorOddsStruct < 0  →  DC (mRot) favoured
    logStructW = np.zeros(M)
    for m in range(M):
        if M > 1:
            lam = float(m) / float(M - 1)
        else:
            lam = 0.0
        logStructW[m] = logPriorOddsStruct * (1.0 - lam)
    logStructZ = logSumExp(logStructW)
    for m in range(M):
        logStructW[m] -= logStructZ

    logQ0_h = np.zeros(M)  # all start in M0
    logR_h = np.full((M, MAXRUN), LOGZERO)

    cRun_h = np.zeros((M, MAXRUN, K))
    dRun_h = np.zeros((M, MAXRUN, K))

    # Responsibility counts are indexed by hypothesis, run, then target.
    # Softplus(logAlpha) and softplus(-logAlpha) supply the persistent prior counts.
    selfCount_h = np.zeros((M, MAXRUN, K))
    otherCount_h = np.zeros((M, MAXRUN, K))

    # The outer hazard is fitted once; only the inner hazard learns online.
    logH_fixed = min(logH, 0.0)  # clamp so H <= 1
    log1mH_fixed = log1mexp(logH_fixed)  # log(1 - H), exact in log space
    aHz_h = np.full(M, A_HZ)
    bHz_h = np.full(M, B_HZ)

    cPred_h = np.zeros((M, MAXRUN, K))
    dPred_h = np.zeros((M, MAXRUN, K))
    selfCountPred_h = np.zeros((M, MAXRUN, K))
    otherCountPred_h = np.zeros((M, MAXRUN, K))

    q0s = np.full(numTrials, 1.0)
    runLimits = np.zeros(numTrials, dtype=types.int64)
    pKArrs = np.full((numTrials, MAXRUN), 0.0)
    pChanges = np.full(numTrials, 0.0)
    predEntropy = np.zeros(numTrials)

    postLogQ0 = np.zeros(numTrials)
    postLogR = np.full((numTrials, MAXRUN), LOGZERO)
    aHzStore = np.full(numTrials, A_HZ)
    bHzStore = np.full(numTrials, B_HZ)

    preLogQ0 = np.zeros(numTrials)
    preLogR = np.full((numTrials, MAXRUN), LOGZERO)

    predStructLogW = np.zeros((numTrials, M))

    MAX_K = MAX_TRAIN_TARGETS
    predCompLogProb = np.full((numTrials, MAX_COMP), LOGZERO)
    predCompMean = np.zeros((numTrials, MAX_COMP))
    predCompScale = np.zeros((numTrials, MAX_COMP))
    predCompNu = np.full((numTrials, MAX_COMP), DEFAULT_NU)
    predNComp = np.zeros(numTrials, dtype=types.int64)

    predSelfOtherW = np.zeros((numTrials, MAXRUN, 2))  # [w_self, w_other]

    predCPred = np.zeros((numTrials, MAXRUN, MAX_K))
    predDPred = np.zeros((numTrials, MAXRUN, MAX_K))
    predSelfCount = np.zeros((numTrials, MAXRUN, MAX_K))

    predCrossDot = np.zeros((numTrials, MAX_K, MAX_K))
    predSVals = np.zeros((numTrials, MAX_K))
    predGibbsW = np.zeros((numTrials, MAX_K, MAX_K))

    predObsVar = np.zeros(numTrials)
    predTgtIdx = np.full(numTrials, -1, dtype=types.int64)

    diagD_h = np.zeros((numTrials, M, MAXRUN, K))  # BLR posterior mean d
    diagC_h = np.zeros((numTrials, M, MAXRUN, K))  # BLR precision c
    diagLogR_h = np.full((numTrials, M, MAXRUN), LOGZERO)  # run-length log-probs
    diagLogQ0_h = np.zeros((numTrials, M))  # M0 log-prob per hyp
    diagLogMargPerHyp = np.full((numTrials, M), LOGZERO)  # per-hyp marginal lik
    diagLogMargBOCPD = np.full((numTrials, M, MAXRUN), LOGZERO)  # BOCPD marginals

    _ent_grid_x = np.empty(ENTROPY_GRID_N)
    for _gi in range(ENTROPY_GRID_N):
        _ent_grid_x[_gi] = -180.0 + _gi * ENTROPY_DX

    _bl_logpdf_ent = np.empty(ENTROPY_GRID_N)
    for _gi in range(ENTROPY_GRID_N):
        _bl_logpdf_ent[_gi] = logWrappedStudentTPdf(_ent_grid_x[_gi], 0.0, scale_S0, nu_S0)

    # Reusable per-trial work arrays (avoid repeated allocation)
    _m1_logw = np.full(MAX_COMP, LOGZERO)
    _m1_mean = np.zeros(MAX_COMP)
    _m1_scale = np.zeros(MAX_COMP)
    _m1_logscale = np.zeros(MAX_COMP)
    _m1_invscale = np.zeros(MAX_COMP)
    _m1_base = np.zeros(MAX_COMP)  # m1_logw + _logNormNu - m1_logscale

    predLogZM1 = np.zeros(numTrials)
    predLogPM1 = np.full(numTrials, LOGZERO)

    logLikelihood = 0.0

    for idx in range(numTrials):
        trial = trials[idx]
        aim = allAims[trial]
        deltaObs = rotations[idx] if isRotation[trial] else 0.0
        effectiveDeltaObs = wrapAngle(deltaObs)
        absObs = math.fabs(effectiveDeltaObs)
        f = featureVector(targets[trial] * deg2rad)
        theta_deg = targets[trial]
        tgtIdx = findTargetIdx(destTargets, nDestTargets, theta_deg)

        visVar = (visCoeff * absObs) * (visCoeff * absObs)

        # Invalid inputs skip state updates but retain a row in the diagnostic arrays.
        _nanObs = not math.isfinite(effectiveDeltaObs)
        _nanTgt = not math.isfinite(theta_deg)
        _nanTrial = _nanObs or _nanTgt

        for m in range(M):
            diagLogQ0_h[idx, m] = logQ0_h[m]
            for k in range(MAXRUN):
                diagLogR_h[idx, m, k] = logR_h[m, k]
                for j in range(K):
                    diagC_h[idx, m, k, j] = cRun_h[m, k, j]
                    diagD_h[idx, m, k, j] = dRun_h[m, k, j]

        for m in range(M):
            predStructLogW[idx, m] = logStructW[m]

        # Store pre-update state (structure-averaged M1 probability)
        # Use hypothesis 0 for run-length tracking (representative)
        preLogQ0[idx] = logQ0_h[0]
        for k_s in range(MAXRUN):
            preLogR[idx, k_s] = logR_h[0, k_s]
        predObsVar[idx] = visVar if tgtIdx >= 0 else sigmaAim2
        predTgtIdx[idx] = tgtIdx

        maxRunLimit = 1
        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN
            if rl_m > maxRunLimit:
                maxRunLimit = rl_m
        runLimits[idx] = maxRunLimit

        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN

            # New runs start with symmetric unit self/other counts.
            for j in range(K):
                cPred_h[m, 0, j] = 0.0
                dPred_h[m, 0, j] = 0.0
                selfCountPred_h[m, 0, j] = 1.0
                otherCountPred_h[m, 0, j] = 1.0

            for k in range(1, rl_m):
                for j in range(K):
                    cPred_h[m, k, j] = cRun_h[m, k - 1, j]
                    dPred_h[m, k, j] = dRun_h[m, k - 1, j]
                    selfCountPred_h[m, k, j] = selfCount_h[m, k - 1, j]
                    otherCountPred_h[m, k, j] = otherCount_h[m, k - 1, j]

            # Merge at MAXRUN
            if rl_m == MAXRUN and rl_m > 1:
                logMass_sum = logAddExp(logR_h[m, rl_m - 2], logR_h[m, rl_m - 1])
                wShort = safeExp(logR_h[m, rl_m - 2] - logMass_sum)
                wLong = 1.0 - wShort
                for j in range(K):
                    c_mg, d_mg = scalarBLRMerge(
                        cRun_h[m, rl_m - 2, j],
                        dRun_h[m, rl_m - 2, j],
                        cRun_h[m, rl_m - 1, j],
                        dRun_h[m, rl_m - 1, j],
                        wShort,
                    )
                    cPred_h[m, rl_m - 1, j] = c_mg
                    dPred_h[m, rl_m - 1, j] = d_mg
                    selfCountPred_h[m, rl_m - 1, j] = (
                        wShort * selfCount_h[m, rl_m - 2, j] + wLong * selfCount_h[m, rl_m - 1, j]
                    )
                    otherCountPred_h[m, rl_m - 1, j] = (
                        wShort * otherCount_h[m, rl_m - 2, j] + wLong * otherCount_h[m, rl_m - 1, j]
                    )

        # Store per-run predictive BLR & self/other state (hypothesis 0)
        if computeEntropy:
            rl_0 = 1
            for k in range(1, MAXRUN):
                if logR_h[0, k] > -700.0 or logR_h[0, k - 1] > -700.0:
                    rl_0 = k + 1
            if rl_0 > MAXRUN:
                rl_0 = MAXRUN
            for k in range(rl_0):
                for j in range(K):
                    predCPred[idx, k, j] = cPred_h[0, k, j]
                    predDPred[idx, k, j] = dPred_h[0, k, j]
                    predSelfCount[idx, k, j] = selfCountPred_h[0, k, j]

        # Temper each M1 Student-t component via p(a)^beta / Z; leave M0 unchanged.
        comp_idx = 0

        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN

            logM0_m = logStructW[m] + logQ0_h[m]
            predCompLogProb[idx, comp_idx] = logM0_m
            predCompMean[idx, comp_idx] = 0.0
            predCompScale[idx, comp_idx] = scale_S0
            predCompNu[idx, comp_idx] = nu_S0
            comp_idx += 1

            if tgtIdx >= 0:
                i_idx = tgtIdx
                for k in range(rl_m):
                    pred_other, var_other_raw = looGlobalFieldPred(
                        fVecs,
                        allPriorDiag[m],
                        dPred_h[m, k],
                        cPred_h[m, k],
                        sVals_all[m],
                        gibbsW_all[m, i_idx],
                        crossDot_all[m],
                        i_idx,
                        K,
                    )
                    scale_other = math.sqrt(var_other_raw + sigmaAim2)

                    # Self prediction — hierarchical prior when unobserved
                    if cPred_h[m, k, i_idx] > 0.0:
                        pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                        fSf_self = safeFSf(
                            crossDot_all[m, i_idx, i_idx],
                            cPred_h[m, k, i_idx],
                            crossDot_all[m, i_idx, i_idx],
                        )
                    else:
                        pred_self = pred_other
                        fSf_self = var_other_raw
                    scale_self = math.sqrt(fSf_self + sigmaAim2)

                    n_self = selfCountPred_h[m, k, i_idx]
                    wGSum = 0.0
                    n_other = 0.0
                    for j_c in range(K):
                        if j_c != i_idx:
                            wGSum += gibbsW_all[m, i_idx, j_c]
                    for j_c in range(K):
                        if j_c != i_idx and wGSum > 1e-30:
                            wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                            n_other += wG_j * otherCountPred_h[m, k, j_c]
                    denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                    w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                    w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                    logW_self = logStructW[m] + logR_h[m, k] + safeLog(max(w_self, 1e-30))
                    logW_other = logStructW[m] + logR_h[m, k] + safeLog(max(w_other, 1e-30))
                    scale_self_beta, nu_self_beta = temperStudentTParams(
                        scale_self, DEFAULT_NU, beta
                    )
                    scale_other_beta, nu_other_beta = temperStudentTParams(
                        scale_other, DEFAULT_NU, beta
                    )

                    predCompLogProb[idx, comp_idx] = logW_self
                    predCompMean[idx, comp_idx] = -pred_self
                    predCompScale[idx, comp_idx] = scale_self_beta
                    predCompNu[idx, comp_idx] = nu_self_beta
                    comp_idx += 1

                    predCompLogProb[idx, comp_idx] = logW_other
                    predCompMean[idx, comp_idx] = -pred_other
                    predCompScale[idx, comp_idx] = scale_other_beta
                    predCompNu[idx, comp_idx] = nu_other_beta
                    comp_idx += 1

                    if computeEntropy and m == 0:
                        predSelfOtherW[idx, k, 0] = w_self
                        predSelfOtherW[idx, k, 1] = w_other
            else:
                # Unknown target: use prior-based M1 components (mean=0)
                priorFSF_m = computePriorFSF(allPriorDiag[m], f)
                scale_prior = math.sqrt(priorFSF_m + sigmaAim2)
                for k in range(rl_m):
                    logW_prior = logStructW[m] + logR_h[m, k]
                    scale_prior_beta, nu_prior_beta = temperStudentTParams(
                        scale_prior, DEFAULT_NU, beta
                    )

                    predCompLogProb[idx, comp_idx] = logW_prior
                    predCompMean[idx, comp_idx] = 0.0
                    predCompScale[idx, comp_idx] = scale_prior_beta
                    predCompNu[idx, comp_idx] = nu_prior_beta
                    comp_idx += 1

        predNComp[idx] = comp_idx

        logAimPdf = LOGZERO
        for c in range(comp_idx):
            logAimPdf = logAddExp(
                logAimPdf,
                predCompLogProb[idx, c]
                + logWrappedStudentTPdf(
                    aim,
                    predCompMean[idx, c],
                    predCompScale[idx, c],
                    predCompNu[idx, c],
                ),
            )

        if computeEntropy:
            nComp = comp_idx
            logNorm = logSumExp(predCompLogProb[idx, :nComp])
            entropy_val = 0.0
            for gi in range(ENTROPY_GRID_N):
                x_g = _ent_grid_x[gi]
                logP_terms = np.full(nComp, LOGZERO)
                for c in range(nComp):
                    logP_terms[c] = (
                        predCompLogProb[idx, c]
                        - logNorm
                        + logWrappedStudentTPdf(
                            x_g,
                            predCompMean[idx, c],
                            predCompScale[idx, c],
                            predCompNu[idx, c],
                        )
                    )
                logP_x = logSumExp(logP_terms)
                p_x = safeExp(logP_x)
                if p_x > 1e-30:
                    entropy_val -= p_x * logP_x * ENTROPY_DX
            predEntropy[idx] = entropy_val

        if mask[trial] and not _nanTgt:
            logLikelihood += logAimPdf

        logQ0_avg_terms = np.full(M, LOGZERO)
        for m in range(M):
            logQ0_avg_terms[m] = logStructW[m] + logQ0_h[m]
        q0_avg = safeExp(logSumExp(logQ0_avg_terms))
        q0s[idx] = q0_avg

        # Legacy diagnostics expose the harmonic hypothesis (index 0).
        for ii in range(K):
            predSVals[idx, ii] = sVals_all[0, ii]
            for jj in range(K):
                predCrossDot[idx, ii, jj] = crossDot_all[0, ii, jj]
                predGibbsW[idx, ii, jj] = gibbsW_all[0, ii, jj]

        if hasFeedback[idx] and not _nanTrial:
            # M0 perturbation marginal uses scale_S0 (baseline aim variance).
            logMargS0 = logWrappedStudentTPdf(effectiveDeltaObs, 0.0, scale_S0, nu_S0)

            logMargPerHyp = np.full(M, LOGZERO)

            for m in range(M):
                logH_curr_m = logH_fixed
                log1mH_curr_m = log1mH_fixed
                logPChangeInt_m = safeLog(aHz_h[m]) - safeLog(aHz_h[m] + bHz_h[m])
                logPNoChangeInt_m = log1mexp(logPChangeInt_m)

                rl_m = 1
                for k in range(1, MAXRUN):
                    if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                        rl_m = k + 1
                if rl_m > MAXRUN:
                    rl_m = MAXRUN

                logMarg_k_m = np.full(MAXRUN, LOGZERO)
                # Self-only evidence prevents cross-target correlations from triggering changepoints.
                logMargBOCPD_k_m = np.full(MAXRUN, LOGZERO)
                if tgtIdx >= 0:
                    i_idx = tgtIdx
                    for k in range(rl_m):
                        pred_other, var_other_raw = looGlobalFieldPred(
                            fVecs,
                            allPriorDiag[m],
                            dPred_h[m, k],
                            cPred_h[m, k],
                            sVals_all[m],
                            gibbsW_all[m, i_idx],
                            crossDot_all[m],
                            i_idx,
                            K,
                        )
                        scale_other = math.sqrt(var_other_raw + visVar)

                        # Self prediction — hierarchical prior when unobserved
                        if cPred_h[m, k, i_idx] > 0.0:
                            pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                            fSf_self = safeFSf(
                                crossDot_all[m, i_idx, i_idx],
                                cPred_h[m, k, i_idx],
                                crossDot_all[m, i_idx, i_idx],
                            )
                        else:
                            pred_self = pred_other
                            fSf_self = var_other_raw
                        scale_self = math.sqrt(fSf_self + visVar)

                        n_self = selfCountPred_h[m, k, i_idx]
                        wGSum = 0.0
                        n_other = 0.0
                        for j_c in range(K):
                            if j_c != i_idx:
                                wGSum += gibbsW_all[m, i_idx, j_c]
                        for j_c in range(K):
                            if j_c != i_idx and wGSum > 1e-30:
                                wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                                n_other += wG_j * otherCountPred_h[m, k, j_c]
                        denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                        w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                        w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                        logMarg_self = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_self, scale_self, DEFAULT_NU
                        )
                        logMarg_other = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_other, scale_other, DEFAULT_NU
                        )

                        if w_self > 1e-30 and w_other > 1e-30:
                            logMarg_k_m[k] = logAddExp(
                                safeLog(w_self) + logMarg_self, safeLog(w_other) + logMarg_other
                            )
                        elif w_self > 1e-30:
                            logMarg_k_m[k] = safeLog(w_self) + logMarg_self
                        else:
                            logMarg_k_m[k] = safeLog(w_other) + logMarg_other

                        logMargBOCPD_k_m[k] = logMarg_self
                else:
                    priorFSF_m = computePriorFSF(allPriorDiag[m], f)
                    for k in range(rl_m):
                        logMarg_k_m[k] = logWrappedStudentTPdf(
                            effectiveDeltaObs, 0.0, math.sqrt(priorFSF_m + visVar), DEFAULT_NU
                        )
                        logMargBOCPD_k_m[k] = logMarg_k_m[k]

                logMarg_M1_terms = np.full(MAXRUN, LOGZERO)
                for k in range(rl_m):
                    if logR_h[m, k] > -700.0:
                        logMarg_M1_terms[k] = logR_h[m, k] + logMarg_k_m[k]
                logMargM1 = logSumExp(logMarg_M1_terms)
                logMargPerHyp[m] = logAddExp(logQ0_h[m] + logMargS0, logMargM1)

                for k in range(rl_m):
                    diagLogMargBOCPD[idx, m, k] = logMargBOCPD_k_m[k]

                # BOCPD mass update for hypothesis m
                # Uses on-diagonal marginals to decouple changepoint detection
                logSumR_old_m = logSumExp(logR_h[m, :])
                logTotalM1_m = logSumR_old_m
                logMassM1toM0_m = logTotalM1_m + logH_curr_m
                logQ0_new_m = logAddExp(logQ0_h[m] + log1mH_curr_m, logMassM1toM0_m) + logMargS0

                logMassFromM0_m = logQ0_h[m] + logH_curr_m
                logM1_cp_terms_m = np.full(MAXRUN, LOGZERO)
                for k in range(MAXRUN):
                    if logR_h[m, k] > -700.0:
                        logM1_cp_terms_m[k] = logR_h[m, k] + logPChangeInt_m
                logMassFromM1cp_m = logSumExp(logM1_cp_terms_m)
                logMassIntoCP_m = logAddExp(logMassFromM0_m, logMassFromM1cp_m)

                logR_new_m = np.full(MAXRUN, LOGZERO)
                logR_new_m[0] = logMassIntoCP_m + logMargBOCPD_k_m[0]
                for k in range(1, rl_m):
                    if logR_h[m, k - 1] > -700.0:
                        logR_new_m[k] = logR_h[m, k - 1] + logPNoChangeInt_m + logMargBOCPD_k_m[k]
                if rl_m == MAXRUN and rl_m > 1:
                    logMassLumped_m = logAddExp(logR_h[m, rl_m - 2], logR_h[m, rl_m - 1])
                    logR_new_m[rl_m - 1] = (
                        logMassLumped_m + logPNoChangeInt_m + logMargBOCPD_k_m[rl_m - 1]
                    )

                logSumR_new_m = logSumExp(logR_new_m)
                logZ_all_m = logAddExp(logQ0_new_m, logSumR_new_m)
                logQ0_h[m] = logQ0_new_m - logZ_all_m
                for k in range(MAXRUN):
                    logR_h[m, k] = logR_new_m[k] - logZ_all_m

                q1_now_m = safeExp(log1mexp(logQ0_h[m]))
                if q1_now_m > EPS:
                    logSumR_hz_m = logSumExp(logR_h[m, :])
                    if logSumR_hz_m > -700.0:
                        soft_change_raw_m = safeExp(logR_h[m, 0] - logSumR_hz_m)
                    else:
                        soft_change_raw_m = 0.0
                    aHz_h[m] += q1_now_m * soft_change_raw_m
                    bHz_h[m] += q1_now_m * (1.0 - soft_change_raw_m)

                if tgtIdx >= 0:
                    i_idx = tgtIdx
                    for k in range(rl_m):
                        resp = safeExp(logR_h[m, k])
                        if resp < 1e-30:
                            for j in range(K):
                                cRun_h[m, k, j] = cPred_h[m, k, j]
                                dRun_h[m, k, j] = dPred_h[m, k, j]
                                selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                                otherCount_h[m, k, j] = otherCountPred_h[m, k, j]
                            continue

                        c_new, d_new = scalarBLRUpdate(
                            cPred_h[m, k, i_idx],
                            dPred_h[m, k, i_idx],
                            sVals_all[m, i_idx],
                            effectiveDeltaObs,
                            visVar,
                        )
                        cRun_h[m, k, i_idx] = c_new
                        dRun_h[m, k, i_idx] = d_new
                        for j in range(K):
                            if j != i_idx:
                                cRun_h[m, k, j] = cPred_h[m, k, j]
                                dRun_h[m, k, j] = dPred_h[m, k, j]

                        pred_other, var_other_raw = looGlobalFieldPred(
                            fVecs,
                            allPriorDiag[m],
                            dPred_h[m, k],
                            cPred_h[m, k],
                            sVals_all[m],
                            gibbsW_all[m, i_idx],
                            crossDot_all[m],
                            i_idx,
                            K,
                        )
                        scale_other = math.sqrt(var_other_raw + visVar)

                        # Self prediction — hierarchical prior when unobserved
                        if cPred_h[m, k, i_idx] > 0.0:
                            pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                            fSf_self = safeFSf(
                                crossDot_all[m, i_idx, i_idx],
                                cPred_h[m, k, i_idx],
                                crossDot_all[m, i_idx, i_idx],
                            )
                        else:
                            pred_self = pred_other
                            fSf_self = var_other_raw
                        scale_self = math.sqrt(fSf_self + visVar)

                        n_self = selfCountPred_h[m, k, i_idx]
                        wGSum = 0.0
                        n_other = 0.0
                        for j_c in range(K):
                            if j_c != i_idx:
                                wGSum += gibbsW_all[m, i_idx, j_c]
                        for j_c in range(K):
                            if j_c != i_idx and wGSum > 1e-30:
                                wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                                n_other += wG_j * otherCountPred_h[m, k, j_c]
                        denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                        w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                        w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                        logLik_self = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_self, scale_self, DEFAULT_NU
                        )
                        logLik_other = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_other, scale_other, DEFAULT_NU
                        )

                        logResp_self = safeLog(max(w_self, 1e-30)) + logLik_self
                        logResp_other = safeLog(max(w_other, 1e-30)) + logLik_other
                        logRespNorm = logAddExp(logResp_self, logResp_other)
                        resp_self = safeExp(logResp_self - logRespNorm)

                        for j in range(K):
                            selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                            otherCount_h[m, k, j] = otherCountPred_h[m, k, j]
                        selfCount_h[m, k, i_idx] = selfCountPred_h[m, k, i_idx] + resp_self
                        otherCount_h[m, k, i_idx] = otherCountPred_h[m, k, i_idx] + (
                            1.0 - resp_self
                        )

                else:
                    for k in range(rl_m):
                        for j in range(K):
                            cRun_h[m, k, j] = cPred_h[m, k, j]
                            dRun_h[m, k, j] = dPred_h[m, k, j]
                            selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                            otherCount_h[m, k, j] = otherCountPred_h[m, k, j]

            # Structure evidence survives run resets, allowing savings across perturbations.
            for m in range(M):
                logStructW[m] += logMargPerHyp[m]
            logStructZ_t = logSumExp(logStructW)
            for m in range(M):
                logStructW[m] -= logStructZ_t

            for m in range(M):
                diagLogMargPerHyp[idx, m] = logMargPerHyp[m]

        # Legacy change diagnostics use hypothesis 0.
        pChanges[idx] = safeExp(safeLog(aHz_h[0]) - safeLog(aHz_h[0] + bHz_h[0]))

        logQ0_wavg_terms = np.full(M, LOGZERO)
        for m in range(M):
            logQ0_wavg_terms[m] = logStructW[m] + logQ0_h[m]
        logQ0_avg = logSumExp(logQ0_wavg_terms)
        logQ1_avg = log1mexp(logQ0_avg)
        q1_avg = safeExp(logQ1_avg)

        logR_avg = np.full(MAXRUN, LOGZERO)
        for k in range(maxRunLimit):
            logR_avg_terms = np.full(M, LOGZERO)
            for m in range(M):
                logR_avg_terms[m] = logStructW[m] + logR_h[m, k]
            logR_avg[k] = logSumExp(logR_avg_terms)
        logSumR_avg = logSumExp(logR_avg)
        if q1_avg > EPS:
            pKArr = safeNormaliseExp(logR_avg[:maxRunLimit], logSumR_avg)
        else:
            pKArr = np.zeros(maxRunLimit)
            if maxRunLimit > 0:
                pKArr[0] = 1.0
        pKArrs[idx, :maxRunLimit] = pKArr

        postLogQ0[idx] = logQ0_avg
        for k in range(MAXRUN):
            postLogR[idx, k] = logR_avg[k]
        aHzStore[idx] = aHz_h[0]
        bHzStore[idx] = bHz_h[0]

    negll = -logLikelihood if math.isfinite(logLikelihood) else 1e12
    if negll < -1e-6:
        negll = 1e12
    return (
        negll,
        q0s,
        runLimits,
        pKArrs,
        pChanges,
        predEntropy,
        postLogQ0,
        postLogR,
        aHzStore,
        bHzStore,
        preLogQ0,
        preLogR,
        predCompLogProb,
        predCompMean,
        predCompScale,
        predCompNu,
        predNComp,
        predSelfOtherW,
        predCPred,
        predDPred,
        predSelfCount,
        predCrossDot,
        predSVals,
        predGibbsW,
        predObsVar,
        predTgtIdx,
        predStructLogW,
        diagD_h,
        diagC_h,
        diagLogR_h,
        diagLogQ0_h,
        diagLogMargPerHyp,
        diagLogMargBOCPD,
        predLogZM1,
        predLogPM1,
    )


# Keep sampling and likelihood state updates aligned; both use component-wise tempering.
@njit(cache=False, fastmath=False)
def samplePredictive(
    params,
    hasFeedback,
    trials,
    isRotation,
    rotations,
    targets,
    scale_S0,
    nu_S0,
    destTargets,
    nDestTargets,
    numSamples,
    rngSeed,
):
    logNegLogH, logVarTrans, logAlpha, kappa, logPriorOddsStruct, logVisCoeff, logBeta = params
    logH = -safeExp(logNegLogH)
    visCoeff = safeExp(logVisCoeff)
    beta = safeExp(logBeta)
    _half_nup1 = 0.5 * (DEFAULT_NU + 1.0)
    _inv_nu = 1.0 / DEFAULT_NU
    _logNormNu = (
        math.lgamma(_half_nup1)
        - math.lgamma(0.5 * DEFAULT_NU)
        - 0.5 * math.log(DEFAULT_NU * math.pi)
    )
    _do_temper = abs(beta - 1.0) > 1e-10

    var_S0 = scale_S0**2
    sigmaAim2 = var_S0
    numTrials = len(trials)
    K = nDestTargets
    deg2rad = math.pi / 180.0
    # Softplus split (same as computeNegLl)
    if logAlpha > 20.0:
        _alpha_self = math.exp(-logAlpha)
        _alpha_other = logAlpha
    elif logAlpha < -20.0:
        _alpha_self = -logAlpha
        _alpha_other = math.exp(logAlpha)
    else:
        _alpha_self = math.log(1.0 + math.exp(-logAlpha))
        _alpha_other = math.log(1.0 + math.exp(logAlpha))
    M = N_STRUCT

    allPriorDiag = buildStructPriorDiags(logVarTrans)

    fVecs = np.zeros((K, DIM))
    for i in range(K):
        fi = featureVector(destTargets[i] * deg2rad)
        for d in range(DIM):
            fVecs[i, d] = fi[d]

    crossDot_all = np.zeros((M, K, K))
    sVals_all = np.zeros((M, K))
    gibbsW_all = np.zeros((M, K, K))
    for m in range(M):
        _, cd_m, sv_m = precomputeVecs(fVecs, allPriorDiag[m], K)
        gw_m = computeGibbsWeights(cd_m, sv_m, K, kappa)
        for i in range(K):
            sVals_all[m, i] = sv_m[i]
            for j in range(K):
                crossDot_all[m, i, j] = cd_m[i, j]
                gibbsW_all[m, i, j] = gw_m[i, j]

    logStructW = np.zeros(M)
    for m in range(M):
        if M > 1:
            lam = float(m) / float(M - 1)
        else:
            lam = 0.0
        logStructW[m] = logPriorOddsStruct * (1.0 - lam)
    logStructZ = logSumExp(logStructW)
    for m in range(M):
        logStructW[m] -= logStructZ

    np.random.seed(rngSeed)

    _ent_grid_x = np.empty(ENTROPY_GRID_N)
    for _gi in range(ENTROPY_GRID_N):
        _ent_grid_x[_gi] = -180.0 + _gi * ENTROPY_DX
    _bl_logpdf_ent = np.empty(ENTROPY_GRID_N)
    for _gi in range(ENTROPY_GRID_N):
        _bl_logpdf_ent[_gi] = logWrappedStudentTPdf(_ent_grid_x[_gi], 0.0, scale_S0, nu_S0)

    _m1_logw = np.full(MAX_COMP, LOGZERO)
    _m1_mean = np.zeros(MAX_COMP)
    _m1_scale = np.zeros(MAX_COMP)
    _m1_logscale = np.zeros(MAX_COMP)
    _m1_invscale = np.zeros(MAX_COMP)
    _m1_base = np.zeros(MAX_COMP)
    _logPi_grid = np.empty(ENTROPY_GRID_N)
    _cumProb = np.zeros(ENTROPY_GRID_N)
    _m0_logw = np.empty(M)  # M0 weights per hypothesis (overwritten each trial)

    logQ0_h = np.zeros(M)
    logR_h = np.full((M, MAXRUN), LOGZERO)
    cRun_h = np.zeros((M, MAXRUN, K))
    dRun_h = np.zeros((M, MAXRUN, K))
    selfCount_h = np.zeros((M, MAXRUN, K))
    otherCount_h = np.zeros((M, MAXRUN, K))

    logH_fixed = min(logH, 0.0)
    log1mH_fixed = log1mexp(logH_fixed)
    aHz_h = np.full(M, A_HZ)
    bHz_h = np.full(M, B_HZ)

    cPred_h = np.zeros((M, MAXRUN, K))
    dPred_h = np.zeros((M, MAXRUN, K))
    selfCountPred_h = np.zeros((M, MAXRUN, K))
    otherCountPred_h = np.zeros((M, MAXRUN, K))

    samples = np.zeros((numTrials, numSamples))

    for idx in range(numTrials):
        trial = trials[idx]
        deltaObs = rotations[idx] if isRotation[trial] else 0.0
        effectiveDeltaObs = wrapAngle(deltaObs)
        absObs = math.fabs(effectiveDeltaObs)
        f = featureVector(targets[trial] * deg2rad)
        theta_deg = targets[trial]
        tgtIdx = findTargetIdx(destTargets, nDestTargets, theta_deg)

        visVar = (visCoeff * absObs) * (visCoeff * absObs)

        # Detect NaN in critical inputs (mirrors computeNegLl guards)
        _nanObs = not math.isfinite(effectiveDeltaObs)
        _nanTgt = not math.isfinite(theta_deg)
        _nanTrial = _nanObs or _nanTgt

        if _nanTgt:
            # Cannot build target-specific predictive; sample from baseline
            for s in range(numSamples):
                z_norm = np.random.normal(0.0, 1.0)
                g = np.random.standard_gamma(nu_S0 / 2.0) * 2.0
                g = max(g, 1e-30)
                t_sample = z_norm * math.sqrt(nu_S0 / g)
                samples[idx, s] = wrapAngle(scale_S0 * t_sample)
            continue

        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN

            for j in range(K):
                cPred_h[m, 0, j] = 0.0
                dPred_h[m, 0, j] = 0.0
                selfCountPred_h[m, 0, j] = 1.0
                otherCountPred_h[m, 0, j] = 1.0

            for k in range(1, rl_m):
                for j in range(K):
                    cPred_h[m, k, j] = cRun_h[m, k - 1, j]
                    dPred_h[m, k, j] = dRun_h[m, k - 1, j]
                    selfCountPred_h[m, k, j] = selfCount_h[m, k - 1, j]
                    otherCountPred_h[m, k, j] = otherCount_h[m, k - 1, j]

            if rl_m == MAXRUN and rl_m > 1:
                logMass_sum = logAddExp(logR_h[m, rl_m - 2], logR_h[m, rl_m - 1])
                wShort = safeExp(logR_h[m, rl_m - 2] - logMass_sum)
                wLong = 1.0 - wShort
                for j in range(K):
                    c_mg, d_mg = scalarBLRMerge(
                        cRun_h[m, rl_m - 2, j],
                        dRun_h[m, rl_m - 2, j],
                        cRun_h[m, rl_m - 1, j],
                        dRun_h[m, rl_m - 1, j],
                        wShort,
                    )
                    cPred_h[m, rl_m - 1, j] = c_mg
                    dPred_h[m, rl_m - 1, j] = d_mg
                    selfCountPred_h[m, rl_m - 1, j] = (
                        wShort * selfCount_h[m, rl_m - 2, j] + wLong * selfCount_h[m, rl_m - 1, j]
                    )
                    otherCountPred_h[m, rl_m - 1, j] = (
                        wShort * otherCount_h[m, rl_m - 2, j] + wLong * otherCount_h[m, rl_m - 1, j]
                    )

        total_comp = 0
        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN
            total_comp += 1 + rl_m * 2

        compLogProb = np.full(total_comp, LOGZERO)
        compMean = np.zeros(total_comp)
        compScale = np.zeros(total_comp)
        compNu = np.full(total_comp, DEFAULT_NU)

        ci = 0
        for m in range(M):
            rl_m = 1
            for k in range(1, MAXRUN):
                if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                    rl_m = k + 1
            if rl_m > MAXRUN:
                rl_m = MAXRUN

            compLogProb[ci] = logStructW[m] + logQ0_h[m]
            compMean[ci] = 0.0
            compScale[ci] = scale_S0
            compNu[ci] = nu_S0
            ci += 1

            if tgtIdx >= 0:
                i_idx = tgtIdx
                for k in range(rl_m):
                    logRunMass = logR_h[m, k]

                    pred_other, var_other_raw = looGlobalFieldPred(
                        fVecs,
                        allPriorDiag[m],
                        dPred_h[m, k],
                        cPred_h[m, k],
                        sVals_all[m],
                        gibbsW_all[m, i_idx],
                        crossDot_all[m],
                        i_idx,
                        K,
                    )
                    scale_other = math.sqrt(var_other_raw + sigmaAim2)

                    if cPred_h[m, k, i_idx] > 0.0:
                        pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                        fSf_self = safeFSf(
                            crossDot_all[m, i_idx, i_idx],
                            cPred_h[m, k, i_idx],
                            crossDot_all[m, i_idx, i_idx],
                        )
                    else:
                        pred_self = pred_other
                        fSf_self = var_other_raw
                    scale_self = math.sqrt(fSf_self + sigmaAim2)

                    n_self = selfCountPred_h[m, k, i_idx]
                    wGSum = 0.0
                    n_other = 0.0
                    for j_c in range(K):
                        if j_c != i_idx:
                            wGSum += gibbsW_all[m, i_idx, j_c]
                    for j_c in range(K):
                        if j_c != i_idx and wGSum > 1e-30:
                            wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                            n_other += wG_j * otherCountPred_h[m, k, j_c]
                    denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                    w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                    w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                    compLogProb[ci] = logStructW[m] + logRunMass + safeLog(max(w_self, 1e-30))
                    compMean[ci] = -pred_self
                    scale_self_beta, nu_self_beta = temperStudentTParams(
                        scale_self, DEFAULT_NU, beta
                    )
                    compScale[ci] = scale_self_beta
                    compNu[ci] = nu_self_beta
                    ci += 1

                    compLogProb[ci] = logStructW[m] + logRunMass + safeLog(max(w_other, 1e-30))
                    compMean[ci] = -pred_other
                    scale_other_beta, nu_other_beta = temperStudentTParams(
                        scale_other, DEFAULT_NU, beta
                    )
                    compScale[ci] = scale_other_beta
                    compNu[ci] = nu_other_beta
                    ci += 1
            else:
                priorFSF_m = computePriorFSF(allPriorDiag[m], f)
                for k in range(rl_m):
                    logRunMass = logR_h[m, k]
                    scale_prior = math.sqrt(priorFSF_m + sigmaAim2)
                    scale_prior_beta, nu_prior_beta = temperStudentTParams(
                        scale_prior, DEFAULT_NU, beta
                    )
                    compLogProb[ci] = logStructW[m] + logRunMass
                    compMean[ci] = 0.0
                    compScale[ci] = scale_prior_beta
                    compNu[ci] = nu_prior_beta
                    ci += 1

        nComp = ci
        logTotal = logSumExp(compLogProb[:nComp])
        cumProb = np.zeros(nComp)
        running = 0.0
        for c in range(nComp):
            running += safeExp(compLogProb[c] - logTotal)
            cumProb[c] = running
        cumProb[nComp - 1] = 1.0

        for s in range(numSamples):
            u = np.random.random()
            chosen = 0
            while chosen < nComp - 1 and cumProb[chosen] < u:
                chosen += 1

            nu_sample = compNu[chosen]
            z_norm = np.random.normal(0.0, 1.0)
            g = np.random.standard_gamma(nu_sample / 2.0) * 2.0
            g = max(g, 1e-30)
            t_sample = z_norm * math.sqrt(nu_sample / g)
            aim_s = compMean[chosen] + max(compScale[chosen], 1e-10) * t_sample
            samples[idx, s] = wrapAngle(aim_s)

        # STATE UPDATE (mirrors computeNegLl)
        # Skipped when rotation is NaN (target NaN already exited above)
        if hasFeedback[idx] and not _nanTrial:
            logMargS0 = logWrappedStudentTPdf(effectiveDeltaObs, 0.0, scale_S0, nu_S0)
            logMargPerHyp = np.full(M, LOGZERO)

            for m in range(M):
                logH_curr_m = logH_fixed
                log1mH_curr_m = log1mH_fixed
                logPChangeInt_m = safeLog(aHz_h[m]) - safeLog(aHz_h[m] + bHz_h[m])
                logPNoChangeInt_m = log1mexp(logPChangeInt_m)

                rl_m = 1
                for k in range(1, MAXRUN):
                    if logR_h[m, k] > -700.0 or logR_h[m, k - 1] > -700.0:
                        rl_m = k + 1
                if rl_m > MAXRUN:
                    rl_m = MAXRUN

                logMarg_k_m = np.full(MAXRUN, LOGZERO)
                # Self-only marginals (for BOCPD)
                logMargBOCPD_k_m = np.full(MAXRUN, LOGZERO)
                if tgtIdx >= 0:
                    i_idx = tgtIdx
                    for k in range(rl_m):
                        pred_other, var_other_raw = looGlobalFieldPred(
                            fVecs,
                            allPriorDiag[m],
                            dPred_h[m, k],
                            cPred_h[m, k],
                            sVals_all[m],
                            gibbsW_all[m, i_idx],
                            crossDot_all[m],
                            i_idx,
                            K,
                        )
                        scale_other = math.sqrt(var_other_raw + visVar)

                        # Self prediction — hierarchical prior when unobserved
                        if cPred_h[m, k, i_idx] > 0.0:
                            pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                            fSf_self = safeFSf(
                                crossDot_all[m, i_idx, i_idx],
                                cPred_h[m, k, i_idx],
                                crossDot_all[m, i_idx, i_idx],
                            )
                        else:
                            pred_self = pred_other
                            fSf_self = var_other_raw
                        scale_self = math.sqrt(fSf_self + visVar)

                        n_self = selfCountPred_h[m, k, i_idx]
                        wGSum = 0.0
                        n_other = 0.0
                        for j_c in range(K):
                            if j_c != i_idx:
                                wGSum += gibbsW_all[m, i_idx, j_c]
                        for j_c in range(K):
                            if j_c != i_idx and wGSum > 1e-30:
                                wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                                n_other += wG_j * otherCountPred_h[m, k, j_c]
                        denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                        w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                        w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                        logMarg_self = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_self, scale_self, DEFAULT_NU
                        )
                        logMarg_other = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_other, scale_other, DEFAULT_NU
                        )

                        if w_self > 1e-30 and w_other > 1e-30:
                            logMarg_k_m[k] = logAddExp(
                                safeLog(w_self) + logMarg_self, safeLog(w_other) + logMarg_other
                            )
                        elif w_self > 1e-30:
                            logMarg_k_m[k] = safeLog(w_self) + logMarg_self
                        else:
                            logMarg_k_m[k] = safeLog(w_other) + logMarg_other

                        logMargBOCPD_k_m[k] = logMarg_self
                else:
                    priorFSF_m = computePriorFSF(allPriorDiag[m], f)
                    for k in range(rl_m):
                        logMarg_k_m[k] = logWrappedStudentTPdf(
                            effectiveDeltaObs, 0.0, math.sqrt(priorFSF_m + visVar), DEFAULT_NU
                        )
                        logMargBOCPD_k_m[k] = logMarg_k_m[k]

                logMarg_M1_terms = np.full(MAXRUN, LOGZERO)
                for k in range(rl_m):
                    if logR_h[m, k] > -700.0:
                        logMarg_M1_terms[k] = logR_h[m, k] + logMarg_k_m[k]
                logMargM1 = logSumExp(logMarg_M1_terms)
                logMargPerHyp[m] = logAddExp(logQ0_h[m] + logMargS0, logMargM1)

                # BOCPD mass update (uses on-diagonal marginals)
                logSumR_old_m = logSumExp(logR_h[m, :])
                logTotalM1_m = logSumR_old_m
                logMassM1toM0_m = logTotalM1_m + logH_curr_m
                logQ0_new_m = logAddExp(logQ0_h[m] + log1mH_curr_m, logMassM1toM0_m) + logMargS0

                logMassFromM0_m = logQ0_h[m] + logH_curr_m
                logM1_cp_terms_m = np.full(MAXRUN, LOGZERO)
                for k in range(MAXRUN):
                    if logR_h[m, k] > -700.0:
                        logM1_cp_terms_m[k] = logR_h[m, k] + logPChangeInt_m
                logMassFromM1cp_m = logSumExp(logM1_cp_terms_m)
                logMassIntoCP_m = logAddExp(logMassFromM0_m, logMassFromM1cp_m)

                logR_new_m = np.full(MAXRUN, LOGZERO)
                logR_new_m[0] = logMassIntoCP_m + logMargBOCPD_k_m[0]
                for k in range(1, rl_m):
                    if logR_h[m, k - 1] > -700.0:
                        logR_new_m[k] = logR_h[m, k - 1] + logPNoChangeInt_m + logMargBOCPD_k_m[k]
                if rl_m == MAXRUN and rl_m > 1:
                    logMassLumped_m = logAddExp(logR_h[m, rl_m - 2], logR_h[m, rl_m - 1])
                    logR_new_m[rl_m - 1] = (
                        logMassLumped_m + logPNoChangeInt_m + logMargBOCPD_k_m[rl_m - 1]
                    )

                logSumR_new_m = logSumExp(logR_new_m)
                logZ_all_m = logAddExp(logQ0_new_m, logSumR_new_m)
                logQ0_h[m] = logQ0_new_m - logZ_all_m
                for k in range(MAXRUN):
                    logR_h[m, k] = logR_new_m[k] - logZ_all_m

                q1_now_m = safeExp(log1mexp(logQ0_h[m]))
                if q1_now_m > EPS:
                    logSumR_hz_m = logSumExp(logR_h[m, :])
                    if logSumR_hz_m > -700.0:
                        soft_change_raw_m = safeExp(logR_h[m, 0] - logSumR_hz_m)
                    else:
                        soft_change_raw_m = 0.0
                    aHz_h[m] += q1_now_m * soft_change_raw_m
                    bHz_h[m] += q1_now_m * (1.0 - soft_change_raw_m)

                if tgtIdx >= 0:
                    i_idx = tgtIdx
                    for k in range(rl_m):
                        resp = safeExp(logR_h[m, k])
                        if resp < 1e-30:
                            for j in range(K):
                                cRun_h[m, k, j] = cPred_h[m, k, j]
                                dRun_h[m, k, j] = dPred_h[m, k, j]
                                selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                                otherCount_h[m, k, j] = otherCountPred_h[m, k, j]
                            continue

                        c_new, d_new = scalarBLRUpdate(
                            cPred_h[m, k, i_idx],
                            dPred_h[m, k, i_idx],
                            sVals_all[m, i_idx],
                            effectiveDeltaObs,
                            visVar,
                        )
                        cRun_h[m, k, i_idx] = c_new
                        dRun_h[m, k, i_idx] = d_new
                        for j in range(K):
                            if j != i_idx:
                                cRun_h[m, k, j] = cPred_h[m, k, j]
                                dRun_h[m, k, j] = dPred_h[m, k, j]

                        pred_other, var_other_raw = looGlobalFieldPred(
                            fVecs,
                            allPriorDiag[m],
                            dPred_h[m, k],
                            cPred_h[m, k],
                            sVals_all[m],
                            gibbsW_all[m, i_idx],
                            crossDot_all[m],
                            i_idx,
                            K,
                        )
                        scale_other = math.sqrt(var_other_raw + visVar)

                        # Self prediction — hierarchical prior when unobserved
                        if cPred_h[m, k, i_idx] > 0.0:
                            pred_self = dPred_h[m, k, i_idx] * crossDot_all[m, i_idx, i_idx]
                            fSf_self = safeFSf(
                                crossDot_all[m, i_idx, i_idx],
                                cPred_h[m, k, i_idx],
                                crossDot_all[m, i_idx, i_idx],
                            )
                        else:
                            pred_self = pred_other
                            fSf_self = var_other_raw
                        scale_self = math.sqrt(fSf_self + visVar)

                        n_self = selfCountPred_h[m, k, i_idx]
                        wGSum = 0.0
                        n_other = 0.0
                        for j_c in range(K):
                            if j_c != i_idx:
                                wGSum += gibbsW_all[m, i_idx, j_c]
                        for j_c in range(K):
                            if j_c != i_idx and wGSum > 1e-30:
                                wG_j = gibbsW_all[m, i_idx, j_c] / wGSum
                                n_other += wG_j * otherCountPred_h[m, k, j_c]
                        denom = (n_self + _alpha_self) + (n_other + _alpha_other)
                        w_self = (n_self + _alpha_self) / denom if denom > 0.0 else 0.5
                        w_other = (n_other + _alpha_other) / denom if denom > 0.0 else 0.5

                        logLik_self = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_self, scale_self, DEFAULT_NU
                        )
                        logLik_other = logWrappedStudentTPdf(
                            effectiveDeltaObs, pred_other, scale_other, DEFAULT_NU
                        )

                        logResp_self = safeLog(max(w_self, 1e-30)) + logLik_self
                        logResp_other = safeLog(max(w_other, 1e-30)) + logLik_other
                        logRespNorm = logAddExp(logResp_self, logResp_other)
                        resp_self = safeExp(logResp_self - logRespNorm)

                        for j in range(K):
                            selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                            otherCount_h[m, k, j] = otherCountPred_h[m, k, j]
                        selfCount_h[m, k, i_idx] = selfCountPred_h[m, k, i_idx] + resp_self
                        otherCount_h[m, k, i_idx] = otherCountPred_h[m, k, i_idx] + (
                            1.0 - resp_self
                        )

                else:
                    for k in range(rl_m):
                        for j in range(K):
                            cRun_h[m, k, j] = cPred_h[m, k, j]
                            dRun_h[m, k, j] = dPred_h[m, k, j]
                            selfCount_h[m, k, j] = selfCountPred_h[m, k, j]
                            otherCount_h[m, k, j] = otherCountPred_h[m, k, j]

            # Preserve structure evidence across run resets, as in computeNegLl.
            for m in range(M):
                logStructW[m] += logMargPerHyp[m]
            logStructZ_t = logSumExp(logStructW)
            for m in range(M):
                logStructW[m] -= logStructZ_t

    return samples


def _featureVector_py(theta_rad):
    f = np.zeros(DIM)
    f[0] = 1.0
    for h in range(1, N_HARM + 1):
        f[2 * h - 1] = np.sin(h * theta_rad)
        f[2 * h] = np.cos(h * theta_rad)
    return f


def _unifiedPred_py(theta_rad, c_val, d_val, priorDiag, v_vec):
    f = _featureVector_py(theta_rad)
    pred = d_val * float(f @ v_vec)
    fSf = max(float(f @ (priorDiag * f)) - c_val * float(f @ v_vec) ** 2, 0.0)
    return pred, fSf


def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        return 0.0
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0
    return 1 - (ssRes / ssTot)


def parseConditionString(condStr):
    parts = condStr.split('_')
    inner_outer = parts[0]
    numTargets = int(parts[1].replace('T', ''))
    rotMag = int(parts[2]) if len(parts) > 2 else 0
    return inner_outer, numTargets, rotMag


def _rotationPhaseMask(pDat):
    phases = derivePhase(pDat)
    return phases == 'rotation'


def identifyGeneralisationTargets(pDat):
    rotMask = pDat['rotation'] != 0
    noFbMask = pDat['Cursor FB'].str.lower().str.strip() == 'no_fb'
    vals = pDat.loc[rotMask & noFbMask, 'targetPosition'].unique()
    return set(v for v in vals if not np.isnan(v))


def identifyTrainingTargets(pDat, genTargets):
    rotPhaseMask = _rotationPhaseMask(pDat)
    fbMask = pDat['Cursor FB'].str.lower().str.strip() != 'no_fb'
    vals = pDat.loc[rotPhaseMask & fbMask, 'targetPosition'].unique()
    allRotTargets = set(v for v in vals if not np.isnan(v))
    return sorted(allRotTargets - genTargets)


def buildTrialStatus(pDat, genTargets):
    status = []
    for _, row in pDat.iterrows():
        if (
            row['rotation'] != 0
            and str(row['Cursor FB']).lower().strip() == 'no_fb'
            and row['targetPosition'] in genTargets
        ):
            status.append('generalisation')
        else:
            status.append('normal')
    return status


def buildHasFeedback(pDat):
    fb = pDat['Cursor FB'].astype(str).str.lower().str.strip().values
    return np.array([v != 'no_fb' for v in fb], dtype=bool)


def derivePhase(pDat):
    rotations = pDat['rotation'].values
    n = len(rotations)
    first_rot = -1
    last_rot = -1
    for i in range(n):
        if rotations[i] != 0:
            if first_rot < 0:
                first_rot = i
            last_rot = i
    phases = []
    for i in range(n):
        if first_rot < 0:
            phases.append('baseline')
        elif i < first_rot:
            phases.append('baseline')
        elif i <= last_rot:
            phases.append('rotation')
        else:
            phases.append('washout')
    return np.array(phases)


def plotCombined(
    modelExplicit,
    mOutsSingle,
    humanExplicit,
    allAims,
    trials,
    number,
    plotIdentifier,
    fittedParams,
    targets,
    compMags,
    phases,
    hasFeedback,
    trialStatus,
    destTargets,
    numSamples=200,
    scale_S0=None,
    nu_S0=10.0,
):
    (
        scale_S0_val,
        nu_S0_val,
        logNegLogH,
        logVarTrans,
        logAlpha,
        kappa,
        logPriorOddsStruct,
        logVisCoeff,
        logBeta,
    ) = fittedParams
    params = np.array(
        [logNegLogH, logVarTrans, logAlpha, kappa, logPriorOddsStruct, logVisCoeff, logBeta]
    )
    numTrials = len(trials)
    isRotation = np.array([phases[trial].lower() == 'rotation' for trial in trials], dtype=bool)
    destTargArr = np.array(destTargets, dtype=np.float64)

    samples = samplePredictive(
        params,
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        scale_S0_val,
        nu_S0_val,
        destTargArr,
        len(destTargets),
        numSamples,
        rngSeed=42,
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    trialBins = np.arange(min(trials) - 0.5, max(trials) + 1.5, 1)
    aimBins = np.arange(-180, 183, 3)

    trialsList = np.repeat(trials, numSamples)
    aimsList = samples.flatten()

    hist, xedges, yedges = np.histogram2d(trialsList, aimsList, bins=(trialBins, aimBins))
    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-10)
    hist = np.ma.masked_where(hist == 0, hist)
    ax.imshow(
        hist.T,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[min(trials), max(trials), -180, 180],
        interpolation='nearest',
    )

    trialArr = np.array(trials)
    humanArr = np.array(humanExplicit)
    genMask = np.array([s == 'generalisation' for s in trialStatus], dtype=bool)
    normMask = ~genMask
    if np.any(normMask):
        ax.scatter(
            trialArr[normMask],
            humanArr[normMask],
            color='#9B30FF',
            marker='x',
            s=50,
            alpha=1,
            zorder=5,
        )
    if np.any(genMask):
        ax.scatter(
            trialArr[genMask],
            humanArr[genMask],
            color='#00FF00',
            marker='x',
            s=50,
            alpha=1,
            zorder=5,
        )
    ax.hlines(y=0, xmin=0, xmax=len(trials), linewidth=0.5, color='grey', alpha=0.5, ls='--')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Degrees')
    ax.set_title(f'Model Predictive Density vs Human for Participant {number}')
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color='grey', label='Model Predictive Density', linewidth=5, alpha=0.5
            ),
            Line2D(
                [0],
                [0],
                marker='x',
                color='#9B30FF',
                label='Human (trained)',
                markersize=6,
                linestyle='None',
            ),
            Line2D(
                [0],
                [0],
                marker='x',
                color='#00FF00',
                label='Human (gen)',
                markersize=6,
                linestyle='None',
            ),
        ]
    )
    plt.savefig(plotIdentifier + str(number) + "_combined.png", dpi=200)
    plt.close()


class Objective:
    def __init__(
        self,
        allAims,
        mask,
        hasFeedback,
        trials,
        phases,
        rotations,
        targets,
        bounds,
        scale_S0,
        nu_S0,
        destTargets,
    ):
        self.allAims = allAims
        self.mask = mask
        self.hasFeedback = hasFeedback
        self.trials = trials
        self.isRotation = np.array([p.lower() == 'rotation' for p in phases], dtype=bool)
        self.rotations = rotations
        self.targets = targets
        self.uniqueThetas = np.unique(targets)
        self.bounds = np.array(bounds)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]
        self.scale_S0 = scale_S0
        self.nu_S0 = nu_S0
        self.destTargets = np.array(destTargets, dtype=np.float64)
        self.nDestTargets = len(destTargets)

    def denormalize(self, pN):
        return np.clip(self.lower + pN * (self.upper - self.lower), self.lower, self.upper)

    def __call__(self, pN):
        return computeNegLl(
            self.denormalize(pN),
            self.allAims,
            self.mask,
            self.hasFeedback,
            self.trials,
            self.isRotation,
            self.rotations,
            self.targets,
            self.uniqueThetas,
            self.scale_S0,
            self.nu_S0,
            self.destTargets,
            self.nDestTargets,
        )[0]


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


def fitSingle(data, boundsSingle, popSizeMultiplier, datasetName, hyper_overrides=None):
    if hyper_overrides is not None:
        _configure_globals(**hyper_overrides)
    (
        allAims,
        mask,
        trials,
        heightCap,
        compMags,
        pp,
        rotPp,
        phases,
        targetPositions,
        uniqueTargets,
        plotIdentifier,
        popSizeMultiplier,
        humanExplicit,
        scale_S0,
        nu_S0,
        hasFeedback,
        trialStatus,
        destTargets,
    ) = data
    objFunc = Objective(
        allAims,
        mask,
        hasFeedback,
        trials,
        phases,
        compMags,
        targetPositions,
        boundsSingle,
        scale_S0,
        nu_S0,
        destTargets,
    )
    numSamples = np.sum(mask)
    if numSamples == 0:
        return np.zeros(len(boundsSingle)), 0.0, 0.0
    nParams = len(boundsSingle)
    current_model_config = _current_model_config()
    save_path = f"{datasetName}/{pp}.pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            saved = pickle.load(f)
        saved_model_config = saved.get('model_config')
        config_matches = saved_model_config == current_model_config or (
            saved_model_config is None and current_model_config == DEFAULT_MODEL_CONFIG
        )
        if (
            saved.get('model_version') == MODEL_VERSION
            and config_matches
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
                    allAims,
                    mask,
                    hasFeedback,
                    objFunc.trials,
                    objFunc.isRotation,
                    compMags,
                    targetPositions,
                    uniqueTargets,
                    scale_S0,
                    nu_S0,
                    objFunc.destTargets,
                    objFunc.nDestTargets,
                )[0]
                if np.isfinite(current_negll) and current_negll < CACHE_FAILURE_NLL:
                    if abs(current_negll - cached_negll) > 1e-9:
                        with open(save_path, 'wb') as f:
                            pickle.dump(
                                {
                                    'model_version': MODEL_VERSION,
                                    'model_config': current_model_config,
                                    'bestX': cached_best_x,
                                    'bestValue': current_negll,
                                    'negll': current_negll,
                                },
                                f,
                            )
                    return cached_best_x, current_negll, current_negll
            print(f"{pp} cached fit unusable; refitting", flush=True)

    t_start = time.time()

    nRestarts = int(40 * popSizeMultiplier)
    nPolish = max(5, nRestarts // 8)
    maxfevals_per_run = 16000

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
                'model_version': MODEL_VERSION,
                'model_config': current_model_config,
                'bestX': bestX_real,
                'bestValue': bestValue,
                'negll': bestValue,
            },
            f,
        )
    return bestX_real, bestValue, bestValue


def processAndPlotSingle(
    i,
    dataItem,
    rawX,
    negll,
    numSamples,
    plotIdentifier,
    hyper_overrides=None,
    make_quick_plots=True,
):
    if hyper_overrides is not None:
        _configure_globals(**hyper_overrides)
    (
        allAims,
        mask,
        trials,
        heightCap,
        compMags,
        pp,
        conVal,
        phases,
        targets,
        uniqueTargets,
        _,
        popSizeMultiplier,
        humanExplicit,
        scale_S0,
        nu_S0,
        hasFeedback,
        trialStatus,
        destTargets,
    ) = dataItem
    xs = [scale_S0, nu_S0] + list(rawX)
    params = np.array(list(rawX))
    numTrials = len(trials)
    isRotation = np.array([phases[trial].lower() == 'rotation' for trial in trials], dtype=bool)
    destTargArr = np.array(destTargets, dtype=np.float64)
    result = computeNegLl(
        params,
        np.zeros(numTrials),
        np.zeros(numTrials, dtype=bool),
        hasFeedback,
        trials,
        isRotation,
        compMags,
        targets,
        np.unique(targets),
        scale_S0,
        nu_S0,
        destTargArr,
        len(destTargets),
        computeEntropy=True,
    )
    (
        _,
        q0s,
        runLimits,
        pKArrs,
        pChanges_arr,
        predEntropyArr,
        postLogQ0,
        postLogR,
        aHzStore,
        bHzStore,
        preLogQ0,
        preLogR,
        predCompLogProb,
        predCompMean,
        predCompScale,
        predCompNu,
        predNComp,
        predSelfOtherW,
        predCPred,
        predDPred,
        predSelfCount,
        predCrossDot,
        predSVals,
        predGibbsW,
        predObsVar,
        predTgtIdx,
        predStructLogW,
        diagD_h,
        diagC_h,
        diagLogR_h,
        diagLogQ0_h,
        diagLogMargPerHyp,
        diagLogMargBOCPD,
        predLogZM1,
        predLogPM1,
    ) = result

    modelExplicit = np.zeros(numTrials)
    if make_quick_plots:
        plotCombined(
            modelExplicit,
            modelExplicit,
            humanExplicit,
            allAims,
            trials,
            pp,
            plotIdentifier,
            xs,
            targets,
            compMags,
            phases,
            hasFeedback,
            trialStatus,
            destTargets,
            numSamples=200,
            scale_S0=scale_S0,
            nu_S0=nu_S0,
        )

    validAims = allAims[mask]
    nParams = 7
    bicI = nParams * np.log(len(validAims)) + 2 * negll if len(validAims) > 0 else np.inf
    rSquared = computeRSquared(validAims, modelExplicit[mask])
    rmseVal = (
        np.sqrt(np.mean((validAims - modelExplicit[mask]) ** 2)) if len(validAims) > 0 else np.inf
    )

    return (
        rmseVal,
        rSquared,
        bicI,
        modelExplicit.tolist(),
        q0s,
        runLimits,
        pKArrs,
        pChanges_arr.tolist(),
        predEntropyArr,
        postLogQ0,
        postLogR,
        aHzStore,
        bHzStore,
        preLogQ0,
        preLogR,
        predCompLogProb,
        predCompMean,
        predCompScale,
        predCompNu,
        predNComp,
        predSelfOtherW,
        predCPred,
        predDPred,
        predSelfCount,
        predCrossDot,
        predSVals,
        predGibbsW,
        predObsVar,
        predTgtIdx,
        predStructLogW,
        diagLogQ0_h,
        diagLogMargPerHyp,
        predLogZM1,
        predLogPM1,
    )


class FitShell:
    def __init__(
        self,
        df,
        conVal='none',
        condition='none',
        fitPhase='rotation',
        heightCap=180,
        plotIdentifier='',
        numCores=multiprocessing.cpu_count() // 2,
        popSizeMultiplier=1,
        datasetName='default',
        fitWashout=False,
        # Model hyper-parameters (None → keep current module defaults)
        n_harm=None,
        a_hz=None,
        b_hz=None,
        maxrun=None,
        n_struct=None,
        make_quick_plots=True,
    ):
        self.fitWashout = fitWashout
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.numCores = numCores
        self.plotIdentifier = plotIdentifier
        self.popSizeMultiplier = popSizeMultiplier
        self.datasetName = datasetName
        self.make_quick_plots = make_quick_plots
        self.participantInfo = {}
        self.trialStatuses = {}
        self.genTargets = {}
        self.trainTargets = {}
        self.destTargets = {}
        self.predState = {}
        self._hyper_overrides = dict(
            n_harm=n_harm,
            a_hz=a_hz,
            b_hz=b_hz,
            maxrun=maxrun,
            n_struct=n_struct,
        )

    def fitRot(self, numCores=multiprocessing.cpu_count() // 2):
        _configure_globals(**self._hyper_overrides)
        self.modelConfig = _current_model_config()

        if self.condition != 'none' and self.conVal != 'none':
            if isinstance(self.conVal, (int, float)):
                pInCond = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            else:
                pInCond = self.df[self.df[self.condition].isin(self.conVal)][
                    'participantNum'
                ].unique()
            self.dat = self.df[self.df['participantNum'].isin(pInCond)]
        uniqP = self.dat['participantNum'].unique()
        self.participantNums = uniqP
        N = len(uniqP)
        if N == 0:
            return

        self.bics = np.zeros(N)
        self.rmses = np.zeros(N)
        self.rSquareds = np.zeros(N)
        self.negLl = np.zeros(N)
        self.mStates = [[] for _ in range(N)]
        self.allAims = [[] for _ in range(N)]
        self.xs = []
        self.q0s = [[] for _ in range(N)]
        self.predEntropyAll = [[] for _ in range(N)]

        dataList = []
        for pp in uniqP:
            pDat_full = self.df[self.df['participantNum'] == pp].copy()
            condStr = pDat_full['condition'].iloc[0]
            io, nT, _ = parseConditionString(condStr)
            self.participantInfo[pp] = {'inner_outer': io, 'numTargets': nT, 'condition': condStr}
            phases_full = derivePhase(pDat_full)
            hE = pDat_full['aim'].values
            bl = hE[phases_full == 'baseline']
            bl = bl[~np.isnan(bl)]
            scale_S0, nu_S0 = fitBaselineStudentT(bl)
            gT = identifyGeneralisationTargets(pDat_full)
            self.genTargets[pp] = gT
            tT = identifyTrainingTargets(pDat_full, gT)
            self.trainTargets[pp] = tT
            dT = sorted(set(tT) | gT)
            self.destTargets[pp] = dT
            nwM = (
                np.ones(len(phases_full), dtype=bool)
                if self.fitWashout
                else (phases_full != 'washout')
            )
            pDat = pDat_full[nwM].reset_index(drop=True)
            phases = phases_full[nwM]
            allAims = pDat['aim'].values
            hFb = buildHasFeedback(pDat)
            tS = buildTrialStatus(pDat, gT)
            self.trialStatuses[pp] = tS
            cM = pDat['rotation'].values
            rP = cM[cM != 0][0] if np.any(cM != 0) else 0
            tP = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uT = np.unique(tP[~np.isnan(tP)])
            trials = np.arange(len(pDat))
            dataList.append(
                (
                    allAims,
                    mask,
                    trials,
                    self.heightCap,
                    cM,
                    pp,
                    rP,
                    phases,
                    tP,
                    uT,
                    self.plotIdentifier,
                    self.popSizeMultiplier,
                    allAims,
                    scale_S0,
                    nu_S0,
                    hFb,
                    tS,
                    dT,
                )
            )

        boundsSingle = [
            (-5.0, 8),  # logNegLogH  (θ: logH = -exp(θ); θ=-5→H≈0.993, θ=8.6→H≈exp(-5400)≈0)
            (-5.0, 100.0),  # logVarTrans
            (-40.0, 40.0),  # logAlpha
            (-20.0, 80.0),  # kappa
            (-5.0, 600.0),  # logPriorOddsStruct
            (-5.0, 0.0),  # logVisCoeff  (exp(-5)≈0.007, exp(0)=1.0; sensible range ~0.01-1.0)
            (-3, 3.0),  # logBeta (beta in [, 20.1])
        ]
        os.makedirs(self.datasetName, exist_ok=True)
        with multiprocessing.Pool(processes=self.numCores) as pool:
            results = pool.starmap(
                fitSingle,
                [
                    (
                        dataList[i],
                        boundsSingle,
                        self.popSizeMultiplier,
                        self.datasetName,
                        self._hyper_overrides,
                    )
                    for i in range(N)
                ],
            )
        indivParams = np.array([r[0] for r in results])
        currentNeglls = np.array([r[2] for r in results])
        self.allAims = [d[0].tolist() for d in dataList]
        self.xs = []
        for i in range(N):
            self.xs.append([dataList[i][13], dataList[i][14]] + list(indivParams[i]))
        self.negLl = currentNeglls

        with multiprocessing.Pool(processes=self.numCores) as pool:
            plotResults = pool.starmap(
                processAndPlotSingle,
                [
                    (
                        i,
                        dataList[i],
                        indivParams[i],
                        currentNeglls[i],
                        np.sum(dataList[i][1]),
                        self.plotIdentifier,
                        self._hyper_overrides,
                        self.make_quick_plots,
                    )
                    for i in range(N)
                ],
            )

        for i, res in enumerate(plotResults):
            (
                rmse,
                r2,
                bic,
                mO,
                q0,
                rl,
                pk,
                pc,
                pEnt,
                postLogQ0,
                postLogR,
                aHzStore,
                bHzStore,
                preLogQ0,
                preLogR,
                predCompLogProb,
                predCompMean,
                predCompScale,
                predCompNu,
                predNComp,
                predSelfOtherW,
                predCPred,
                predDPred,
                predSelfCount,
                predCrossDot,
                predSVals,
                predGibbsW,
                predObsVar,
                predTgtIdx,
                predStructLogW,
                diagLogQ0_h,
                diagLogMargPerHyp,
                predLogZM1,
                predLogPM1,
            ) = res
            self.rmses[i] = rmse
            self.bics[i] = bic
            self.rSquareds[i] = r2
            self.mStates[i] = mO
            self.q0s[i] = q0
            self.predEntropyAll[i] = pEnt
            pp = uniqP[i]
            self.predState[pp] = {
                'predEntropy': pEnt,
                'q0s': q0,
                'runLimits': rl,
                'pKArrs': pk,
                'pChanges': pc,
                'postLogQ0': postLogQ0,
                'postLogR': postLogR,
                'aHz': aHzStore,
                'bHz': bHzStore,
                'preLogQ0': preLogQ0,
                'preLogR': preLogR,
                'predCompLogProb': predCompLogProb,
                'predCompMean': predCompMean,
                'predCompScale': predCompScale,
                'predCompNu': predCompNu,
                'predNComp': predNComp,
                'predSelfOtherW': predSelfOtherW,
                'predCPred': predCPred,
                'predDPred': predDPred,
                'predSelfCount': predSelfCount,
                'predCrossDot': predCrossDot,
                'predSVals': predSVals,
                'predGibbsW': predGibbsW,
                'predObsVar': predObsVar,
                'predTgtIdx': predTgtIdx,
                'predStructLogW': predStructLogW,
                'diagLogQ0_h': diagLogQ0_h,
                'diagLogMargPerHyp': diagLogMargPerHyp,
                'predLogZM1': predLogZM1,
                'predLogPM1': predLogPM1,
            }

        with open(os.path.join(self.datasetName, '_participant_metadata.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'modelConfig': self.modelConfig,
                    'participantInfo': self.participantInfo,
                    'trialStatuses': self.trialStatuses,
                    'genTargets': {pp: list(v) for pp, v in self.genTargets.items()},
                    'trainTargets': self.trainTargets,
                    'destTargets': self.destTargets,
                },
                f,
            )
        for pp in uniqP:
            pred_save_path = os.path.join(self.datasetName, f'{pp}_predstate.pkl')
            with open(pred_save_path, 'wb') as f:
                pickle.dump(
                    {
                        'modelConfig': self.modelConfig,
                        'xs': self.xs[list(uniqP).index(pp)],
                        'negll': self.negLl[list(uniqP).index(pp)],
                        'predState': self.predState[pp],
                        'destTargets': self.destTargets[pp],
                        'trainTargets': self.trainTargets[pp],
                        'genTargets': list(self.genTargets[pp]),
                        'trialStatuses': self.trialStatuses[pp],
                    },
                    f,
                )
