"""Convert the savings experiment's MATLAB table to trial-level fitting data.

A state-space model fitted to probes estimates implicit adaptation on all
trials. Explicit aim is hand_theta - implicit; feedback rotation includes
implicit adaptation. Washout and probe aims are excluded from likelihood scoring.

Each subject has 720 trials: baseline probes (40), baseline feedback (40),
rotation (200), washout probes (40), washout feedback (160), re-exposure (200),
and final washout probes (40).
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import struct
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def load_exp1_mat(mat_path):
    """
    Load Exp1.mat which contains a MATLAB table stored as an MCOS object.
    The actual numeric data lives in __function_workspace__ as 20 double arrays
    of shape (17280, 1).
    """
    mat = sio.loadmat(mat_path)
    ws = mat['__function_workspace__'].flatten()
    buf = ws.tobytes()

    # Find all miMATRIX elements of size 138288 (= 17280 doubles + mat headers)
    array_offsets = []
    pos = 0
    while pos < len(buf) - 8:
        dtype_val, nbytes = struct.unpack_from('<II', buf, pos)
        if dtype_val == 14 and nbytes == 138288:
            array_offsets.append(pos)
            pos += nbytes + 8
        else:
            pos += 1

    assert len(array_offsets) == 20, f"Expected 20 data columns, found {len(array_offsets)}"

    arrays = []
    for off in array_offsets:
        # Skip the miMATRIX tag, flags, dimensions and empty name before the numeric data tag.
        data_tag_off = off + 8 + 16 + 16 + 8
        vals = np.frombuffer(buf, dtype='<f8', count=17280, offset=data_tag_off + 8)
        arrays.append(vals.copy())

    # Column mapping (CN absent; 20 columns verified by value ranges)
    col_names = [
        'SN',
        'TN',
        'CCW',
        'ti',
        'stage',
        'fbi',
        'ri',
        'clampi',
        'breaks',
        'hand_theta',
        'hand_theta_maxv',
        'hand_theta_maxradv',
        'handMaxRadExt',
        'raw_ep_hand_ang',
        'hand_theta_50',
        'MT',
        'RT',
        'ST',
        'radvelmax',
        'maxRadDist',
    ]

    df = pd.DataFrame({name: arr for name, arr in zip(col_names, arrays)})
    return df


def assign_phases(df):
    """Assign phase labels based on stage number."""
    phase = pd.Series('baseline', index=df.index)
    phase[df['stage'].isin([3, 6])] = 'rotation'
    phase[df['stage'].isin([4, 5, 7])] = 'washout'
    return phase


def _simulate_implicit(A, B, ri):
    """Forward-simulate the deterministic implicit SSM."""
    T = len(ri)
    x = np.zeros(T)
    for t in range(T - 1):
        x[t + 1] = (A + B) * x[t] + B * ri[t]
    return x


def compute_implicit_per_subject(subj_df):
    """
    Fit A, B of the single-process SSM to probe observations (MLE / least
    squares), then return x_t on every trial.

    SSM:  x_{t+1} = A * x_t + B * (ri_t + x_t)
    Obs on probes:  y_t ≈ x_t
    """
    ri = subj_df['ri'].values.copy()
    ht = subj_df['hand_theta'].values.copy()
    fbi = subj_df['fbi'].values
    is_probe = (fbi == 0) & ~np.isnan(ht)

    probe_idx = np.where(is_probe)[0]
    probe_y = ht[is_probe]

    if len(probe_idx) < 4:
        return np.zeros(len(subj_df))

    def neg_loglik(params):
        A, B = params
        x = _simulate_implicit(A, B, ri)
        resid = probe_y - x[probe_idx]
        # MLE with Gaussian obs noise: minimise SSE (variance cancels)
        return np.sum(resid**2)

    best_val = np.inf
    best_x = None
    # Negative B corrects against the signed error ri + x_t.
    bounds = [
        (0.0, 1.0),  # A: retention
        (-0.5, 0.0),  # B: corrective learning rate
    ]

    rng = np.random.RandomState(42)
    starts = [
        [0.99, -0.05],
        [0.95, -0.10],
        [0.98, -0.02],
        [0.97, -0.15],
        [0.999, -0.03],
        [0.90, -0.20],
    ]
    for _ in range(14):
        starts.append([rng.uniform(0.8, 1.0), rng.uniform(-0.3, -0.01)])

    for x0 in starts:
        try:
            res = minimize(
                neg_loglik,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        A, B = 0.98, 0.05
    else:
        A, B = best_x

    return _simulate_implicit(A, B, ri)


def convert_exp1(mat_path, output_csv_path=None):
    """
    Full pipeline: load → compute implicit → derive aim & rotation → format.
    """
    print("Loading Exp1.mat ...")
    raw = load_exp1_mat(mat_path)
    print(
        f"  {len(raw)} rows, {raw['SN'].nunique()} subjects, {int(raw['TN'].max())} trials/subject"
    )

    raw['phase'] = assign_phases(raw)

    print("Computing implicit adaptation from probe trials ...")
    implicit_all = np.zeros(len(raw))
    for sn in sorted(raw['SN'].unique()):
        mask = raw['SN'] == sn
        implicit_all[mask] = compute_implicit_per_subject(raw[mask])
    raw['implicit'] = implicit_all

    raw['aim'] = raw['hand_theta'] - raw['implicit']

    # Exclude washout and probes from aim scoring while retaining trials for state updates.
    raw.loc[raw['phase'] == 'washout', 'aim'] = np.nan
    raw.loc[raw['fbi'] == 0, 'aim'] = np.nan

    # Feedback includes the implicit aftereffect; probes keep raw rotation for phase detection.
    raw['rotation'] = raw['ri'].copy()
    fb_mask = raw['fbi'] == 1
    rot_or_wash_fb = fb_mask & (raw['phase'].isin(['rotation', 'washout']))
    raw.loc[rot_or_wash_fb, 'rotation'] = (
        raw.loc[rot_or_wash_fb, 'ri'] + raw.loc[rot_or_wash_fb, 'implicit']
    )

    out = pd.DataFrame()
    out['participantNum'] = raw['SN'].astype(int)
    out['aim'] = raw['aim']
    out['rotation'] = raw['rotation']
    out['targetPosition'] = raw['ti']
    out['Cursor FB'] = np.where(raw['fbi'] == 1, 'cursor', 'no_fb')
    out['blockNum'] = (raw['stage'] - 1).astype(int)  # 0-indexed

    # Condition string for parseConditionString: "{io}_{nT}T_{rotMag}"
    n_targets = int(raw.groupby('SN')['ti'].nunique().mode().iloc[0])
    out['condition'] = f"inner_{n_targets}T_45"

    out['hand_theta'] = raw['hand_theta']
    out['implicit'] = raw['implicit']
    out['rawRotation'] = raw['ri']
    out['phase'] = raw['phase']
    out['fbi'] = raw['fbi'].astype(int)
    out['TN'] = raw['TN'].astype(int)
    out['CCW'] = raw['CCW'].astype(int)
    out['stage'] = raw['stage'].astype(int)

    out = out.sort_values(['participantNum', 'TN']).reset_index(drop=True)

    if output_csv_path:
        out.to_csv(output_csv_path, index=False)
        print(f"Saved to {output_csv_path}")

    print(
        f"\nOutput: {out.shape[0]} rows × {out.shape[1]} cols, "
        f"{out['participantNum'].nunique()} subjects"
    )
    for sn in sorted(out['participantNum'].unique())[:2]:
        s = out[out['participantNum'] == sn]
        print(f"  Subject {sn}:")
        for ph in ['baseline', 'rotation', 'washout']:
            pm = s['phase'] == ph
            print(
                f"    {ph:10s}: n={pm.sum():4d}, "
                f"aim={s.loc[pm, 'aim'].mean():+.2f}, "
                f"rot={s.loc[pm, 'rotation'].mean():+.2f}, "
                f"impl={s.loc[pm, 'implicit'].mean():+.2f}"
            )

    return out


if __name__ == '__main__':
    mat_path = 'Exp1.mat'
    csv_path = 'Exp1.csv'
    convert_exp1(mat_path, csv_path)
