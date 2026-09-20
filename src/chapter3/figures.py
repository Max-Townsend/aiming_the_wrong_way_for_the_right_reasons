"""Chapter 3 figure recipes using the shared plotting routines."""

import importlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import BayesHypothesisTesting as BHT
import runpy
from .plotting import (
    _apply_fit_model_config,
    generate_bht_phase_plots,
    generate_bht_plots,
    plot_ding_companion,
    plot_savings_companion,
    plot_savings_individual_companion,
    plot_wildcard_figure,
)
from .statistics import wildcard_slope_statistics, wildcard_aim_statistics


def _setup(quick=False):
    np.random.seed(42)
    Path("tempFigures").mkdir(exist_ok=True)
    folder = "CheckFigures/" if quick else "BLRFigures/"
    Path(folder).mkdir(exist_ok=True)
    return folder


def figure32(quick=False):
    targetFolder = _setup(quick)
    dingFitsForPlots = np.load("dingBLRDing.npy", allow_pickle=True)
    hmmDingFitsForPlots = np.load("HMMDing.npy", allow_pickle=True)
    if quick:
        for fit in dingFitsForPlots.item().values():
            fit.participantNums = fit.participantNums[:1]
    import blr_energy_plots as bep

    importlib.reload(bep)

    ENERGY_ANALYSIS_SAMPLES = 64 if quick else 10000
    ENERGY_ANALYSIS_MAX_WORKERS = 6
    ENERGY_ANALYSIS_PARALLEL = not quick
    dingBaseFolder = targetFolder + 'DingFigures/'
    Path(dingBaseFolder).mkdir(exist_ok=True, parents=True)
    dingSubFolder = dingBaseFolder + 'indiplots/'
    Path(dingSubFolder).mkdir(exist_ok=True)
    dingPhaseFolder = dingBaseFolder + 'phaseplots/'
    Path(dingPhaseFolder).mkdir(exist_ok=True)
    generate_bht_plots(
        dingFitsForPlots,
        save_dir=dingSubFolder,
        max_participants=1 if quick else 600,
        num_samples=32 if quick else 500,
        debug=True,
        per_participant_flip=True,
        ding_aggregate=True,
        exclude_washout=True,
    )
    plt.close("all")
    import ding_mechanistic_comparison as dmc

    importlib.reload(dmc)
    DING_RESULTS = dmc.main(
        save_dir=dingBaseFolder + "mechanistic_comparison",
        n_samples=64 if quick else 10000,
        max_workers=6,
        parallel=not quick,
    )
    plt.close("all")
    dingEnergyFolder = dingBaseFolder + 'energy_analysis/'
    Path(dingEnergyFolder).mkdir(exist_ok=True, parents=True)
    if hmmDingFitsForPlots is not None:
        DING_ENERGY_RESULTS = bep.run_ding_energy_analysis(
            bht_fit_dict=dingFitsForPlots,
            hmm_fit=hmmDingFitsForPlots,
            save_dir=dingEnergyFolder,
            n_samples=ENERGY_ANALYSIS_SAMPLES,
            max_workers=ENERGY_ANALYSIS_MAX_WORKERS,
            parallel=ENERGY_ANALYSIS_PARALLEL,
        )
    else:
        print('Skipping Ding energy analysis: HMMDing fits not found.')
    plt.close("all")
    generate_bht_phase_plots(
        dingFitsForPlots,
        save_dir=dingPhaseFolder,
        n_bins=150,
        num_samples=32 if quick else 2000,
        debug=True,
        per_participant_flip=True,
        ding_aggregate=True,
        ylimMax=0.03,
        hmm_fits=hmmDingFitsForPlots,
    )
    plt.close("all")
    dingCompFolder = dingBaseFolder + 'companion/'
    Path(dingCompFolder).mkdir(exist_ok=True, parents=True)
    plot_ding_companion(
        dingFitsForPlots,
        save_dir=dingCompFolder,
        debug=True,
        ding_aggregate=True,
        exclude_washout=True,
    )
    plt.close("all")
    ding_group_plots(dingFitsForPlots, dingBaseFolder)
    plt.close("all")
    geometry(quick=quick)


def ding_group_plots(dingFitsForPlots, dingBaseFolder):
    from scipy.special import logsumexp
    from scipy.stats import sem, gaussian_kde
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    import dingBLR

    ding2TFolder = dingBaseFolder + 'group_2T/'
    Path(ding2TFolder).mkdir(exist_ok=True, parents=True)
    ding8TFolder = dingBaseFolder + 'group_8T/'
    Path(ding8TFolder).mkdir(exist_ok=True, parents=True)

    dingDict = (
        dingFitsForPlots.item() if isinstance(dingFitsForPlots, np.ndarray) else dingFitsForPlots
    )

    def _ding_group_heatmap_with_model(dingDict, task_size, save_folder):
        """
        Human aim density heatmap (magma_r) with BHT model predictive
        contours (cyan with dark stroke) overlaid.

        Uses exact predictive densities from predState for the model overlay,
        which preserves multimodality better than sample-based KDE.

        Layout: 2×2 (Inner/Outer × Train/Gen) + P(rotation) bottom row
                (single-column, aligned under left panel).
        """
        from scipy.stats import t as t_dist

        inner_keys = [k for k in dingDict if f'Inner_{task_size}' in k]
        outer_keys = [k for k in dingDict if f'Outer_{task_size}' in k]
        N_TRIALS = 120
        ROT_ONSET = 24
        AIM_LO, AIM_HI = -181, 181
        aim_centres = np.linspace(AIM_LO, AIM_HI, 200)
        contour_levels = [0.2, 0.4, 0.6, 0.8]
        contour_colors = ['#004C4C', '#007F7F', '#00BABA', '#00FFFF']
        contour_widths = [1.5, 1.5, 1.5, 1.5]

        def _trial_model_pdf(ps, trial_idx, flip):
            comp_logp = np.asarray(ps['predCompLogProb'])
            comp_mean = np.asarray(ps['predCompMean'])
            comp_scale = np.asarray(ps['predCompScale'])
            comp_nu = np.asarray(ps.get('predCompNu', np.full(comp_scale.shape, 5.0)))
            n_comp_arr = np.asarray(ps['predNComp']).astype(int)

            if trial_idx >= len(n_comp_arr):
                return None
            nc = int(n_comp_arr[trial_idx])
            if nc <= 0:
                return None

            lp = comp_logp[trial_idx, :nc]
            mn = comp_mean[trial_idx, :nc] * flip
            sc = comp_scale[trial_idx, :nc]
            nu = comp_nu[trial_idx, :nc]
            valid = (
                np.isfinite(lp)
                & np.isfinite(mn)
                & np.isfinite(sc)
                & np.isfinite(nu)
                & (sc > 0)
                & (nu > 0)
            )
            if not np.any(valid):
                return None

            lp_v = lp[valid]
            mn_v = mn[valid]
            sc_v = sc[valid]
            nu_v = nu[valid]
            wts = np.exp(lp_v - logsumexp(lp_v))
            pdf_matrix = t_dist.pdf(
                aim_centres[:, None], df=nu_v[None, :], loc=mn_v[None, :], scale=sc_v[None, :]
            )
            pdf_vals = pdf_matrix @ wts
            if not np.any(np.isfinite(pdf_vals)):
                return None
            peak = np.nanmax(pdf_vals)
            if not np.isfinite(peak) or peak <= 0:
                return None
            return pdf_vals

        def _collect_group(keys):
            train_h, gen_h, train_m, gen_m = {}, {}, {}, {}
            n_pp = 0
            for key in keys:
                bht = dingDict[key]
                _apply_fit_model_config(bht)
                for pIdx, pp in enumerate(bht.participantNums):
                    ps = bht.predState.get(pp)
                    if ps is None:
                        continue
                    pdf = bht.df[bht.df['participantNum'] == pp].copy()
                    if len(pdf) == 0:
                        continue
                    pDat_full = pdf.reset_index(drop=True)
                    phases_full = np.array(BHT.derivePhase(pDat_full))
                    n_common = min(
                        len(pDat_full),
                        len(phases_full),
                        len(bht.allAims[pIdx]),
                        len(bht.trialStatuses[pp]),
                        len(ps['predNComp']),
                    )
                    if n_common <= 0:
                        continue
                    nwM = np.array(
                        [str(ph).lower() != 'washout' for ph in phases_full[:n_common]], dtype=bool
                    )
                    if not np.any(nwM):
                        continue
                    rots = pDat_full['rotation'].values[:n_common][nwM]
                    nz = rots[rots != 0]
                    if len(nz) == 0:
                        continue
                    flip = -np.sign(nz[0])
                    aims = np.array(bht.allAims[pIdx], dtype=float)[:n_common][nwM] * flip
                    ts = np.array(bht.trialStatuses[pp], dtype=object)[:n_common][nwM]

                    max_trials = min(N_TRIALS, len(ts), len(aims))
                    n_pp += 1
                    for i in range(max_trials):
                        trial_pdf = _trial_model_pdf(ps, i, flip)
                        if ts[i] == 'generalisation':
                            gen_h.setdefault(i, []).append(aims[i])
                            if trial_pdf is not None:
                                gen_m.setdefault(i, []).append(trial_pdf)
                        else:
                            train_h.setdefault(i, []).append(aims[i])
                            if trial_pdf is not None:
                                train_m.setdefault(i, []).append(trial_pdf)
            return train_h, gen_h, train_m, gen_m, n_pp

        def _kde_image(td, tr):
            trials = sorted([t for t in td if tr[0] <= t <= tr[1]])
            img = np.full((len(aim_centres), len(trials)), np.nan)
            for col, t in enumerate(trials):
                vals = np.clip(np.array(td[t])[np.isfinite(td[t])], AIM_LO + 5, AIM_HI - 5)
                if len(vals) < 5:
                    continue
                try:
                    d = gaussian_kde(vals, bw_method=0.25)(aim_centres)
                    peak = np.nanmax(d)
                    if np.isfinite(peak) and peak > 0:
                        img[:, col] = d / peak
                except Exception:
                    pass
            return img, np.array(trials)

        def _pdf_image(td, tr):
            trials = sorted([t for t in td if tr[0] <= t <= tr[1]])
            img = np.full((len(aim_centres), len(trials)), np.nan)
            for col, t in enumerate(trials):
                pdfs = [np.asarray(p, dtype=float) for p in td[t] if p is not None]
                if len(pdfs) == 0:
                    continue
                arr = np.vstack(pdfs)
                mean_pdf = np.nanmean(arr, axis=0)
                peak = np.nanmax(mean_pdf)
                if np.isfinite(peak) and peak > 0:
                    img[:, col] = mean_pdf / peak
            return img, np.array(trials)

        def _mean_by_trial(td, tr):
            trials = sorted([t for t in td if tr[0] <= t <= tr[1]])
            return (np.array(trials), np.array([np.nanmean(td[t]) for t in trials]))

        def _collect_prot(keys):
            all_p = []
            for key in keys:
                bht = dingDict[key]
                _apply_fit_model_config(bht)
                for pIdx, pp in enumerate(bht.participantNums):
                    ps = bht.predState.get(pp)
                    if ps is None:
                        continue
                    pDat_full = bht.df[bht.df['participantNum'] == pp].reset_index(drop=True)
                    phases_full = np.array(BHT.derivePhase(pDat_full))
                    n_common = min(len(phases_full), len(ps['predStructLogW']), len(pDat_full))
                    if n_common <= 0:
                        continue
                    nwM = np.array(
                        [str(ph).lower() != 'washout' for ph in phases_full[:n_common]], dtype=bool
                    )
                    if not np.any(nwM):
                        continue
                    r = pDat_full['rotation'].values[:n_common][nwM]
                    if not np.any(r != 0):
                        continue
                    sl = np.array(ps['predStructLogW'])[:n_common][nwM]
                    p_rot = np.exp(sl[:, 1] - logsumexp(sl, axis=1))
                    padded = np.full(N_TRIALS, np.nan)
                    use_n = min(N_TRIALS, len(p_rot))
                    padded[:use_n] = p_rot[:use_n]
                    all_p.append(padded)
            return np.array(all_p)

        print(f'Collecting {task_size}...')
        in_th, in_gh, in_tm, in_gm, n_in = _collect_group(inner_keys)
        out_th, out_gh, out_tm, out_gm, n_out = _collect_group(outer_keys)
        in_prot, out_prot = _collect_prot(inner_keys), _collect_prot(outer_keys)

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.45], hspace=0.28, wspace=0.12)
        cmap = plt.cm.magma_r.copy()
        cmap.set_bad('white')
        cmap.set_under('white')
        stroke = [pe.withStroke(linewidth=2.0, foreground='black')]

        panel_specs = [
            (0, 0, in_th, in_tm, f'Inner {task_size} — Training (n={n_in})'),
            (0, 1, in_gh, in_gm, f'Inner {task_size} — Generalisation'),
            (1, 0, out_th, out_tm, f'Outer {task_size} — Training (n={n_out})'),
            (1, 1, out_gh, out_gm, f'Outer {task_size} — Generalisation'),
        ]

        last_pcm = None  # store a pcolormesh mappable for colorbar
        last_cs = None  # store a contour set for colorbar

        for row, col, h_td, m_td, title in panel_specs:
            ax = fig.add_subplot(gs[row, col])

            img, tpos = _kde_image(h_td, (0, N_TRIALS - 1))
            ts = tpos - ROT_ONSET
            if len(tpos) > 0 and np.any(np.isfinite(img)):
                masked = np.ma.masked_where(~np.isfinite(img) | (img <= 1 / 140), img)
                pcm = ax.pcolormesh(
                    ts,
                    aim_centres,
                    masked,
                    cmap=cmap,
                    shading='nearest',
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                last_pcm = pcm

            # Exact model contours from predState mixture densities
            mi, mt = _pdf_image(m_td, (0, N_TRIALS - 1))
            mts = mt - ROT_ONSET
            if len(mt) > 1 and np.any(np.isfinite(mi)):
                mc = np.nan_to_num(mi, 0.0)
                mc = np.ma.masked_where(mc < 0.08, mc)
                cs = ax.contour(
                    mts,
                    aim_centres,
                    mc,
                    levels=contour_levels,
                    colors=contour_colors,
                    linewidths=contour_widths,
                )
                last_cs = cs
                for child in ax.get_children():
                    if isinstance(child, LineCollection):
                        child.set_path_effects(stroke)

            mmt, mmv = _mean_by_trial(h_td, (0, N_TRIALS - 1))

            for y in [0, 60, -60]:
                ax.axhline(y, color='white', lw=0.5, ls='--', alpha=0.6)
            ax.axvline(-0.5, color='white', lw=1.8, ls=':', alpha=0.5)
            ax.set_xlim(-ROT_ONSET, N_TRIALS - ROT_ONSET)
            ax.set_ylim(-181, 181)
            ax.set_title(title, fontsize=18)
            ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
            ax.tick_params(labelsize=14)
            if col == 0:
                ax.set_ylabel('Aim (°)', fontsize=16)
            if row < 1:
                ax.set_xticklabels([])
            sns.despine(ax=ax)

        ax_p = fig.add_subplot(gs[2, 0])
        trial_ax = np.arange(N_TRIALS) - ROT_ONSET
        c_in, c_out = '#2166AC', '#B2182B'

        def _mean_ci(arr):
            return (np.nanmean(arr, 0), 1.96 * sem(arr, axis=0, nan_policy='omit'))

        for prot, c, lbl in [
            (in_prot, c_in, f'Inner {task_size}'),
            (out_prot, c_out, f'Outer {task_size}'),
        ]:
            m, ci = _mean_ci(prot)
            ax_p.plot(trial_ax, m, color=c, lw=1.5, label=lbl)
            ax_p.fill_between(trial_ax, m - ci, m + ci, color=c, alpha=0.15)

        ax_p.axhline(0.5, color='gray', lw=0.5, ls='--', alpha=0.5)
        for yy in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            ax_p.axhline(yy, color='gray', lw=0.4, ls='-', alpha=0.3)
        ax_p.axvline(-0.5, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax_p.set_xlabel('Trial (0 = perturbation onset)', fontsize=16)
        ax_p.set_ylabel('P(rotation)', fontsize=16)
        ax_p.set_ylim(-0.05, 1.05)
        ax_p.set_xlim(-ROT_ONSET, N_TRIALS - ROT_ONSET)
        ax_p.tick_params(labelsize=14)
        ax_p.legend(fontsize=14, loc='center right', framealpha=0.7)
        sns.despine(ax=ax_p)

        sp = save_folder + f'Ding_{task_size}_inner_vs_outer_model_overlay'
        fig.savefig(sp + '.svg', bbox_inches='tight')
        fig.savefig(sp + '.png', dpi=200, bbox_inches='tight')
        plt.show()

        from matplotlib.colors import BoundaryNorm, ListedColormap

        if last_pcm is not None:
            fig_cb1, ax_cb1 = plt.subplots(figsize=(0.6, 4))
            cb1 = fig_cb1.colorbar(last_pcm, cax=ax_cb1, orientation='vertical')
            cb1.set_label('Human aim density (norm.)', fontsize=14)
            cb1.ax.tick_params(labelsize=12)
            cb1_path = save_folder + f'Ding_{task_size}_colorbar_human'
            fig_cb1.savefig(cb1_path + '.svg', bbox_inches='tight')
            fig_cb1.savefig(cb1_path + '.png', dpi=200, bbox_inches='tight')
            plt.show()
            print(f'Saved to {cb1_path}')

        fig_cb2, ax_cb2 = plt.subplots(figsize=(0.6, 4))
        boundaries = [0.1, 0.3, 0.5, 0.7, 0.9]
        cb_cmap = ListedColormap(contour_colors)
        norm = BoundaryNorm(boundaries, cb_cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cb_cmap, norm=norm)
        sm.set_array([])
        cb2 = fig_cb2.colorbar(sm, cax=ax_cb2, orientation='vertical', ticks=contour_levels)
        cb2.set_label('Model contour level (norm.)', fontsize=14)
        cb2.ax.tick_params(labelsize=12, which='both')
        cb2.ax.minorticks_off()
        cb2_path = save_folder + f'Ding_{task_size}_colorbar_model'
        fig_cb2.savefig(cb2_path + '.svg', bbox_inches='tight')
        fig_cb2.savefig(cb2_path + '.png', dpi=200, bbox_inches='tight')
        plt.show()
        print(f'Saved to {cb2_path}')

    _ding_group_heatmap_with_model(dingDict, '2T', ding2TFolder)
    _ding_group_heatmap_with_model(dingDict, '8T', ding8TFolder)


def geometry(quick=False):
    targetFolder = _setup(quick)
    dingBaseFolder = targetFolder + 'DingFigures/'
    Path(dingBaseFolder).mkdir(exist_ok=True, parents=True)
    dingSubFolder = dingBaseFolder + 'indiplots/'
    Path(dingSubFolder).mkdir(exist_ok=True)
    dingPhaseFolder = dingBaseFolder + 'phaseplots/'
    Path(dingPhaseFolder).mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    def _draw_target_geometry(
        ax,
        training_angles_deg,
        gen_angle_deg,
        title,
        label_training=True,
        label_gen=True,
        radius=1.0,
        train_label_pos=None,
        gen_label_pos=None,
        marker_size=5,
        gen_marker_size=None,
    ):
        """Draw a single target-geometry sub-panel.

        Angles use compass convention: 0 = top (12 o'clock), CW positive.
        Start position is at the centre of the workspace.
        A light gray reference circle is always drawn.
        """
        if gen_marker_size is None:
            gen_marker_size = marker_size

        ax.set_aspect('equal')
        pad = 0.75
        ax.set_xlim(-radius - pad, radius + pad)
        ax.set_ylim(-radius - pad - 0.15, radius + pad)
        ax.axis('off')
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=6)

        circle = plt.Circle((0, 0), radius, fill=False, edgecolor='0.78', linewidth=0.7)
        ax.add_patch(circle)

        ax.plot(0, 0, 'o', color='0.55', markersize=4, zorder=4)
        ax.text(
            0,
            -0.16,
            'Start\nPosition',
            ha='center',
            va='top',
            fontsize=5.5,
            color='0.4',
            linespacing=0.85,
        )

        def _xy(angle_deg):
            rad = np.radians(angle_deg)
            return radius * np.sin(rad), radius * np.cos(rad)

        blue = '#2166AC'
        orange = '#D95F02'

        for ang in training_angles_deg:
            x, y = _xy(ang)
            ax.plot(x, y, 'o', color=blue, markersize=marker_size, zorder=6)

        gx, gy = _xy(gen_angle_deg)
        ax.plot(gx, gy, 'o', color=orange, markersize=gen_marker_size, zorder=6)

        if label_training and train_label_pos is not None:
            ys = [_xy(a)[1] for a in training_angles_deg]
            ref_ang = training_angles_deg[np.argmax(ys)]
            tx, ty = _xy(ref_ang)
            ax.annotate(
                'Training\nTarget',
                xy=(tx, ty),
                xytext=train_label_pos,
                fontsize=5.5,
                color=blue,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=blue, lw=0.8, shrinkA=0, shrinkB=2),
                ha='center',
                va='bottom',
                linespacing=0.85,
            )

        if label_gen and gen_label_pos is not None:
            ax.annotate(
                'Generalization\nTarget',
                xy=(gx, gy),
                xytext=gen_label_pos,
                fontsize=5.5,
                color=orange,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=orange, lw=0.8, shrinkA=0, shrinkB=2),
                ha='center',
                va='top',
                linespacing=0.85,
            )

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))

    # Outer (constrained): 2 training targets 140° apart → ±70° from top
    exp1_outer = np.array([-70.0, 70.0])
    # Inner (unconstrained): 2 training targets 20° apart → ±10° from top
    exp1_inner = np.array([-10.0, 10.0])
    # Gen target: opposite midpoint → 180° (bottom)

    _draw_target_geometry(
        axes[0],
        exp1_outer,
        180.0,
        'Outer (Constrained)',
        label_training=True,
        label_gen=True,
        train_label_pos=(-1.1, 0.75),
        gen_label_pos=(-1.0, -1.25),
    )
    _draw_target_geometry(
        axes[1], exp1_inner, 180.0, 'Inner (Unconstrained)', label_training=False, label_gen=False
    )

    # Outer targets form two 5° clusters, each 110° from the generalisation target.
    exp2_outer = np.concatenate(
        [
            np.array([-70.0, -65.0, -60.0, -55.0]),  # left cluster
            np.array([55.0, 60.0, 65.0, 70.0]),  # right cluster
        ]
    )
    # Inner (unconstrained): 8 targets at 20° spacing → -70° to 70°
    exp2_inner = np.linspace(-70, 70, 8)

    # Smaller markers keep the 5° target spacing visible.
    _draw_target_geometry(
        axes[2],
        exp2_outer,
        180.0,
        'Outer (Constrained)',
        label_training=True,
        label_gen=True,
        marker_size=3.5,
        gen_marker_size=4.5,
        train_label_pos=(-0.6, 1.2),
        gen_label_pos=(-1.0, -1.25),
    )
    _draw_target_geometry(axes[3], exp2_inner, 180.0, 'Inner (Unconstrained)', marker_size=4.5)

    fig.subplots_adjust(left=0.04, right=0.97, top=0.80, bottom=0.02, wspace=0.20)

    mid01 = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    mid23 = (axes[2].get_position().x0 + axes[3].get_position().x1) / 2
    fig.text(
        mid01,
        0.96,
        'Experiment 1 Target Geometry',
        ha='center',
        va='top',
        fontsize=9.5,
        fontweight='bold',
    )
    fig.text(
        mid23,
        0.96,
        'Experiment 2 Target Geometry',
        ha='center',
        va='top',
        fontsize=9.5,
        fontweight='bold',
    )

    fig.text(
        axes[0].get_position().x0 - 0.015,
        0.97,
        'C',
        fontsize=13,
        fontweight='bold',
        va='top',
        ha='right',
    )
    fig.text(
        axes[2].get_position().x0 - 0.015,
        0.97,
        'D',
        fontsize=13,
        fontweight='bold',
        va='top',
        ha='right',
    )

    savePath = dingBaseFolder + 'Ding_task_geometry_CD.svg'
    fig.savefig(savePath, dpi=300, bbox_inches='tight')
    fig.savefig(savePath.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved to {savePath}')


def figure33():
    quick = False
    targetFolder = _setup()
    savingsFits = np.load("BHTAvrahamSavings.npy", allow_pickle=True)
    import blr_energy_plots as bep

    importlib.reload(bep)

    ENERGY_ANALYSIS_SAMPLES = 64 if quick else 10000
    ENERGY_ANALYSIS_MAX_WORKERS = 6
    ENERGY_ANALYSIS_PARALLEL = not quick
    savingsBaseFolder = targetFolder + 'SavingsFigures/'
    Path(savingsBaseFolder).mkdir(exist_ok=True, parents=True)
    savingsSubFolder = savingsBaseFolder + 'indiplots/'
    Path(savingsSubFolder).mkdir(exist_ok=True)
    savingsPhaseFolder = savingsBaseFolder + 'phaseplots/'
    Path(savingsPhaseFolder).mkdir(exist_ok=True)
    plt.close("all")
    generate_bht_plots(
        savingsFits,
        save_dir=savingsSubFolder,
        max_participants=600,
        num_samples=500,
        debug=True,
        block_boundary_lines=True,
        include_implicit=True,
        per_participant_flip=True,
    )
    plt.close("all")
    plot_savings_companion(
        savingsFits.item(),
        savingsFits.item().df,
        save_path=savingsBaseFolder + 'savings_combined_BCD.png',
    )
    plt.close("all")
    savingsCompFolder = savingsBaseFolder + 'companion/'
    Path(savingsCompFolder).mkdir(exist_ok=True, parents=True)
    plot_savings_individual_companion(savingsFits, save_dir=savingsCompFolder, debug=True)
    plt.close("all")
    runpy.run_path("plot_signflip_savings.py", run_name="__main__")
    plt.close("all")
    hmmSavingsFits = np.load('HMMSavings.npy', allow_pickle=True)
    savingsEnergyFolder = savingsBaseFolder + 'energy_analysis/'
    Path(savingsEnergyFolder).mkdir(exist_ok=True, parents=True)
    SAVINGS_ENERGY_RESULTS = bep.run_single_group_energy_analysis(
        dataset_label='Savings',
        bht_fit=savingsFits,
        hmm_fit=hmmSavingsFits,
        save_dir=savingsEnergyFolder,
        n_samples=ENERGY_ANALYSIS_SAMPLES,
        max_workers=ENERGY_ANALYSIS_MAX_WORKERS,
        parallel=ENERGY_ANALYSIS_PARALLEL,
    )
    plt.close("all")


def figure34(quick=False):
    targetFolder = _setup(quick)
    WildCardFits = np.load("BHTWildcard.npy", allow_pickle=True)
    if quick:
        WildCardFits.item().participantNums = WildCardFits.item().participantNums[:1]
    import blr_energy_plots as bep

    importlib.reload(bep)

    ENERGY_ANALYSIS_SAMPLES = 64 if quick else 10000
    ENERGY_ANALYSIS_MAX_WORKERS = 6
    ENERGY_ANALYSIS_PARALLEL = not quick
    wildcardBaseFolder = targetFolder + 'WildCardFigures/'
    Path(wildcardBaseFolder).mkdir(exist_ok=True, parents=True)
    wildcardSubFolder = wildcardBaseFolder + 'indiplots/'
    Path(wildcardSubFolder).mkdir(exist_ok=True)
    wildcardPhaseFolder = wildcardBaseFolder + 'phaseplots/'
    Path(wildcardPhaseFolder).mkdir(exist_ok=True)
    generate_bht_plots(
        WildCardFits,
        save_dir=wildcardSubFolder,
        max_participants=1 if quick else 600,
        num_samples=32 if quick else 500,
        debug=True,
        multi_rotation_colors=True,
    )
    plt.close("all")
    plot_wildcard_figure(
        'wildCardTask.csv', 'BHTWildcard.npy', save_path=wildcardBaseFolder + 'wildcard_F2.png'
    )
    plt.close("all")
    hmmWildcardFits = np.load('HMMWildcard.npy', allow_pickle=True)
    wildcardEnergyFolder = wildcardBaseFolder + 'energy_analysis/'
    Path(wildcardEnergyFolder).mkdir(exist_ok=True, parents=True)
    WILDCARD_ENERGY_RESULTS = bep.run_single_group_energy_analysis(
        dataset_label='Wildcard',
        bht_fit=WildCardFits,
        hmm_fit=hmmWildcardFits,
        save_dir=wildcardEnergyFolder,
        n_samples=ENERGY_ANALYSIS_SAMPLES,
        max_workers=ENERGY_ANALYSIS_MAX_WORKERS,
        parallel=ENERGY_ANALYSIS_PARALLEL,
    )
    plt.close("all")
    import wildcard_pen12_window16_multipanel as wc_multi

    importlib.reload(wc_multi)

    wildcardHeterogeneityFolder = Path(wildcardBaseFolder) / 'heterogeneity_trueother'
    wildcardHeterogeneityFolder.mkdir(exist_ok=True, parents=True)

    wildcardMultiFig, wildcardMultiCsv = wc_multi.generate_wildcard_trueother_multipanel(
        window_cycles=16,
        out_fig=wildcardHeterogeneityFolder / 'wildcard_trueother_discrimination_multipanel.png',
        out_csv=wildcardHeterogeneityFolder
        / 'wildcard_trueother_discrimination_multipanel_points.csv',
        show_profile_spaghetti=False,
        panel_d_mode='model_profile',
        show_panel_b=False,
    )

    print(f"Saved {wildcardMultiFig}")
    print(f"Saved {wildcardMultiCsv}")
    plt.close("all")
    wildcard_slope_statistics()
    wildcard_aim_statistics()
