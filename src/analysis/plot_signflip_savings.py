"""
Plot p(signFlipError) timeseries for Savings dataset (Exp1).

Shows human strategic aims (aim = total - implicit) and BHT model predictions,
with Block 1 (A1) and Block 2 (A2) overlaid on the same x-axis (rotation cycle).

Sign convention: each participant's data is flipped by CCW so that positive =
correct compensation direction.  A sign-flip error is simply aim < 0 after
this normalisation.  Since every participant sees a single rotation sign,
no running train_sign accumulation is needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from scipy.special import logsumexp
from scipy.stats import t as student_t
import seaborn as sns
import os

savingsFits = np.load('BHTAvrahamSavings.npy', allow_pickle=True)
bht = savingsFits[0] if hasattr(savingsFits, '__len__') else savingsFits.item()
df_full = pd.read_csv('Exp1.csv')


def bht_p_sign_flip(predstate, trial_idx, flip):
    """P(model aim < 0) after sign-normalisation by flip."""
    n_comp = int(predstate['predNComp'][trial_idx])
    if n_comp <= 0:
        return np.nan
    log_w = predstate['predCompLogProb'][trial_idx, :n_comp]
    weights = np.exp(log_w - logsumexp(log_w))
    means = flip * predstate['predCompMean'][trial_idx, :n_comp]
    scale = predstate['predCompScale'][trial_idx, :n_comp]
    nu = predstate['predCompNu'][trial_idx, :n_comp]
    # CDF at 0: probability mass below zero (wrong sign)
    return float(np.sum(weights * student_t.cdf(-means / scale, df=nu)))


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


rows = []
for pp_idx, pp in enumerate(bht.participantNums):
    pDat = df_full[df_full['participantNum'] == pp].sort_values('TN').reset_index(drop=True)
    predstate = bht.predState[pp]
    n_trials = min(len(pDat), len(predstate['predNComp']))

    ccw = pDat['CCW'].iloc[0]
    flip = -1.0 if ccw == 1 else 1.0

    aims = pDat['aim'].values[:n_trials]
    rotations = pDat['rotation'].values[:n_trials]
    stages = pDat['stage'].values[:n_trials]
    fbis = pDat['fbi'].values[:n_trials]

    for t in range(n_trials):
        if stages[t] not in (3, 6):
            continue
        if fbis[t] != 1 or rotations[t] == 0 or np.isnan(aims[t]):
            continue

        aim_signed = aims[t] * flip
        human_flip = float(aim_signed < 0)

        bht_flip_p = bht_p_sign_flip(predstate, t, flip)
        if np.isnan(bht_flip_p):
            continue

        # Rotation cycle within block: 4 training targets per cycle
        stage_start = np.where(stages == stages[t])[0][0]
        fb_within_stage = np.sum((stages[stage_start:t] == stages[t]) & (fbis[stage_start:t] == 1))
        rot_cycle = fb_within_stage // 4

        rows.append(
            {
                'pp': pp,
                'trial': t,
                'block': 'A1' if stages[t] == 3 else 'A2',
                'rot_cycle': int(rot_cycle),
                'human_flip': human_flip,
                'bht_flip_p': bht_flip_p,
            }
        )

flip_df = pd.DataFrame(rows)
print(f"Built flip table: {len(flip_df)} rows, {flip_df['pp'].nunique()} participants")
print(f"  A1: {(flip_df['block'] == 'A1').sum()}, A2: {(flip_df['block'] == 'A2').sum()}")


def agg_by_cycle(df, col, block):
    sub = df[df['block'] == block]
    pp_cycle = sub.groupby(['rot_cycle', 'pp'])[col].mean().reset_index()
    stats = pp_cycle.groupby('rot_cycle')[col].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    return stats['rot_cycle'].values, stats['mean'].values, stats['ci'].values


fig, ax = plt.subplots(figsize=(10, 6.25))

colors = {
    'human_A1': '#C8A2C8',
    'human_A2': '#4B0082',
    'bht_A1': '#90D4A0',
    'bht_A2': '#1b7a3d',
}

stats_lines = []
for block, hc, bc in [('A1', 'human_A1', 'bht_A1'), ('A2', 'human_A2', 'bht_A2')]:
    cyc_h, m_h, ci_h = agg_by_cycle(flip_df, 'human_flip', block)
    ax.fill_between(cyc_h, m_h - ci_h, m_h + ci_h, color=colors[hc], alpha=0.2)
    ax.scatter(cyc_h, m_h, s=30, color=colors[hc], alpha=0.8, edgecolors='none', zorder=3)

    cyc_b, m_b, ci_b = agg_by_cycle(flip_df, 'bht_flip_p', block)
    ax.fill_between(cyc_b, m_b - ci_b, m_b + ci_b, color=colors[bc], alpha=0.15)
    ax.plot(cyc_b, m_b, color=colors[bc], lw=2.5, alpha=0.9, zorder=3)

    bht_r2 = compute_scalar_r2(m_h, m_b)
    bht_rmse = compute_rmse(m_h, m_b)
    stats_lines.append(
        f'{block}: BHT $R^2$='
        + ('nan' if np.isnan(bht_r2) else f'{bht_r2:.3f}')
        + ', RMSE='
        + ('nan' if np.isnan(bht_rmse) else f'{bht_rmse:.3f}')
    )

ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Rotation Cycle (4 feedback trials)', fontsize=12)
ax.set_ylabel('p(sign-flip error)', fontsize=12)
ax.set_title(
    f'Sign-Flip Error: Human Strategic Aims vs BHT (N={flip_df["pp"].nunique()})', fontsize=13
)
ax.set_ylim(-0.02, 0.65)
ax.set_xlim(-0.5, max(flip_df['rot_cycle'].max(), 40) + 0.5)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.text(
    0.02,
    0.98,
    '\n'.join(stats_lines),
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.8', alpha=0.9),
)

ax.legend(
    handles=[
        mlines.Line2D(
            [], [], marker='o', color=colors['human_A1'], markersize=6, ls='None', label='Human A1'
        ),
        mlines.Line2D(
            [], [], marker='o', color=colors['human_A2'], markersize=6, ls='None', label='Human A2'
        ),
        mlines.Line2D([], [], color=colors['bht_A1'], lw=2.5, label='BHT A1'),
        mlines.Line2D([], [], color=colors['bht_A2'], lw=2.5, label='BHT A2'),
        mlines.Line2D([], [], color='gray', lw=0.8, ls='--', label='Chance'),
    ],
    fontsize=9,
    loc='upper right',
    framealpha=0.8,
)

sns.despine(ax=ax)
plt.tight_layout()

save_path = 'BLRFigures/SavingsFigures/signflip_savings_human_vs_bht.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close(fig)
