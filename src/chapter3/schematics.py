"""Draw the model and target-geometry schematics for Figure 3.1."""

import matplotlib.pyplot as plt


def translation_prediction():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.gridspec import GridSpec
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.dpi': 300,
        }
    )

    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)

    R = 1.0
    RAD2DEG = 180.0 / np.pi
    n_targets = 8
    target_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

    obs_target_angle = 0.0
    rotation_deg = 15.0
    fb_angle = obs_target_angle + np.radians(rotation_deg)

    obs_x = R * np.cos(obs_target_angle)
    obs_y = R * np.sin(obs_target_angle)
    fb_x = R * np.cos(fb_angle)
    fb_y = R * np.sin(fb_angle)

    t_dx = fb_x - obs_x
    t_dy = fb_y - obs_y

    # Cursor minus target position gives a noisy 2D translation observation.
    sigma2_T = 0.03  # prior variance on tx, ty (dimensionless)
    sigma2_obs = 0.005  # observation noise variance on cursor position (per axis)

    Sigma_prior = np.diag([sigma2_T, sigma2_T])
    mu_prior = np.array([0.0, 0.0])

    T_obs = np.array([fb_x - obs_x, fb_y - obs_y])

    # 2D Kalman update with H = I₂
    R_noise = np.diag([sigma2_obs, sigma2_obs])
    S = Sigma_prior + R_noise
    K = Sigma_prior @ np.linalg.inv(S)

    mu_post = mu_prior + K @ (T_obs - mu_prior)
    Sigma_post = (np.eye(2) - K) @ Sigma_prior

    t_dx = mu_post[0]
    t_dy = mu_post[1]
    Sigma_T = Sigma_post

    print(f"T_observed: ({T_obs[0]:.4f}, {T_obs[1]:.4f})")
    print(f"Posterior:  tx={t_dx:.4f}, ty={t_dy:.4f}")
    print(
        f"Sigma_T = [[{Sigma_T[0, 0]:.6f}, {Sigma_T[0, 1]:.6f}], [{Sigma_T[1, 0]:.6f}, {Sigma_T[1, 1]:.6f}]]"
    )
    print(f"std_tx={np.sqrt(Sigma_T[0, 0]):.4f}, std_ty={np.sqrt(Sigma_T[1, 1]):.4f}")

    def cov_ellipse_params(Sigma, n_std=2.0):
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.maximum(vals, 1e-10)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width = 2 * n_std * np.sqrt(vals[0])
        height = 2 * n_std * np.sqrt(vals[1])
        return width, height, angle

    def predict_angular(theta, t_dx, t_dy):
        u = R * np.cos(theta) + t_dx
        v = R * np.sin(theta) + t_dy
        raw = np.degrees(np.arctan2(v, u) - theta)
        return (raw + 180) % 360 - 180

    def angular_jacobian(theta, t_dx, t_dy):
        """Returns H as a 2-vector: d(angular_pred)/d(tx, ty) in deg per unit."""
        u = R * np.cos(theta) + t_dx
        v = R * np.sin(theta) + t_dy
        D2 = u**2 + v**2
        if D2 < 1e-30:
            D2 = 1e-30
        return np.array([-v / D2 * RAD2DEG, u / D2 * RAD2DEG])

    def draw_curved_arrow(ax, theta_start, pert_deg, color, lw=1.8, alpha=1.0, r=R, n_pts=30):
        if abs(pert_deg) < 0.3:
            return
        pert_rad = np.radians(pert_deg)
        theta_end = theta_start + pert_rad
        angles = np.linspace(theta_start, theta_end, n_pts)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle='butt', zorder=15)
        head_len = 0.06
        head_width = 0.04
        tip_x, tip_y = r * np.cos(theta_end), r * np.sin(theta_end)
        back_angle = theta_end - np.sign(pert_deg) * head_len / r
        base_x, base_y = r * np.cos(back_angle), r * np.sin(back_angle)
        norm_x, norm_y = np.cos(theta_end), np.sin(theta_end)
        tri_xs = [tip_x, base_x + head_width * norm_x, base_x - head_width * norm_x]
        tri_ys = [tip_y, base_y + head_width * norm_y, base_y - head_width * norm_y]
        ax.fill(tri_xs, tri_ys, color=color, alpha=alpha, zorder=15)

    def draw_gaussian_on_circle(
        ax,
        theta_target,
        mean_pert_deg,
        std_deg,
        color,
        r_base=R,
        density_scale=0.35,
        n_pts=200,
        alpha=0.25,
    ):
        sweep_range = 3.0 * std_deg
        pert_vals = np.linspace(mean_pert_deg - sweep_range, mean_pert_deg + sweep_range, n_pts)
        density = np.exp(-0.5 * ((pert_vals - mean_pert_deg) / std_deg) ** 2)
        density /= np.max(density)
        angles = theta_target + np.radians(pert_vals)
        r_outer = r_base + density * density_scale
        xs_outer = r_outer * np.cos(angles)
        ys_outer = r_outer * np.sin(angles)
        xs_inner = r_base * np.cos(angles)
        ys_inner = r_base * np.sin(angles)
        xs_fill = np.concatenate([xs_outer, xs_inner[::-1]])
        ys_fill = np.concatenate([ys_outer, ys_inner[::-1]])
        ax.fill(xs_fill, ys_fill, color=color, alpha=alpha, zorder=10)
        ax.plot(xs_outer, ys_outer, color=color, lw=0.8, alpha=alpha + 0.3, zorder=11)

    pred_means = []
    pred_stds = []
    jacobians = []
    for theta in target_angles:
        mu = predict_angular(theta, t_dx, t_dy)
        H = angular_jacobian(theta, t_dx, t_dy)
        var = H @ Sigma_T @ H
        pred_means.append(mu)
        pred_stds.append(np.sqrt(max(var, 0.01)))
        jacobians.append(H)
    pred_means = np.array(pred_means)
    pred_stds = np.array(pred_stds)

    color_mean = '#B2182B'
    color_density = '#D6604D'

    fig = plt.figure(figsize=(17, 7.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.08, width_ratios=[1, 1, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_aspect('equal')
    ax_a.set_xlim(-2.2, 2.2)
    ax_a.set_ylim(-2.0, 2.4)

    ax_a.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax_a.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

    for i, theta in enumerate(target_angles):
        tx, ty = R * np.cos(theta), R * np.sin(theta)
        is_observed = abs(theta - obs_target_angle) < 0.01

        ax_a.plot(
            tx,
            ty,
            'o',
            color='0.15' if is_observed else '0.3',
            ms=12 if is_observed else 10,
            zorder=7,
        )

        fx, fy = tx + t_dx, ty + t_dy

        ax_a.annotate(
            '',
            xy=(fx, fy),
            xytext=(tx, ty),
            arrowprops=dict(
                arrowstyle='->',
                color='#B2182B',
                lw=2.0 if is_observed else 1.3,
                alpha=1.0 if is_observed else 0.7,
                mutation_scale=10,
                shrinkA=5,
                shrinkB=2,
            ),
            zorder=8,
        )

        ax_a.plot(fx, fy, 'o', color='#E66100', ms=5, zorder=12)

        w2, h2, a2 = cov_ellipse_params(Sigma_T, n_std=2.0)
        ell2 = Ellipse(
            (fx, fy),
            w2,
            h2,
            angle=a2,
            facecolor='#D6604D',
            edgecolor='#B2182B',
            alpha=0.15,
            lw=0.8,
            zorder=9,
        )
        ax_a.add_patch(ell2)

        w1, h1, a1 = cov_ellipse_params(Sigma_T, n_std=1.0)
        ell1 = Ellipse(
            (fx, fy),
            w1,
            h1,
            angle=a1,
            facecolor='#D6604D',
            edgecolor='#B2182B',
            alpha=0.2,
            lw=0.6,
            ls='--',
            zorder=9,
        )
        ax_a.add_patch(ell1)

        if is_observed:
            ax_a.annotate(
                'observed\ntarget',
                xy=(tx + 0.06, ty - 0.06),
                fontsize=7.5,
                color='0.15',
                ha='left',
                va='top',
            )

    ax_a.annotate(
        '',
        xy=(t_dx, t_dy),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='0.5', lw=1.2, linestyle='--', mutation_scale=10),
    )
    ax_a.text(
        t_dx - 0.15,
        t_dy / 2 + 0.02,
        '$\\hat{T}$',
        fontsize=12,
        color='0.5',
        fontstyle='italic',
        ha='right',
    )

    ax_a.set_title(
        'A   Cartesian translation + uncertainty', loc='left', fontweight='bold', fontsize=11
    )

    legend_a = [
        Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='0.15', ms=7, label='Observed target'
        ),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.3', ms=7, label='Other targets'),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='#E66100',
            ms=5,
            label='Predicted position',
        ),
        Line2D([0], [0], color='#B2182B', lw=1.5, label='Inferred $\\hat{T}$'),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='#D6604D',
            ms=12,
            alpha=0.3,
            label='Uncertainty (1σ, 2σ)',
        ),
    ]
    ax_a.legend(
        handles=legend_a, fontsize=6, frameon=False, loc='lower left', bbox_to_anchor=(-0.05, -0.08)
    )
    ax_a.axis('off')

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_aspect('equal')
    ax_b.set_xlim(-2.4, 2.4)
    ax_b.set_ylim(-2.4, 2.6)

    ax_b.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax_b.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

    for i, theta in enumerate(target_angles):
        tx, ty = R * np.cos(theta), R * np.sin(theta)
        is_observed = abs(theta - obs_target_angle) < 0.01

        ax_b.plot(
            tx,
            ty,
            'o',
            color='0.15' if is_observed else '0.3',
            ms=12 if is_observed else 10,
            zorder=20,
        )

        mu = pred_means[i]
        std = pred_stds[i]

        draw_gaussian_on_circle(
            ax_b, theta, mu, std, color_density, r_base=R, density_scale=0.40, alpha=0.20
        )

        draw_curved_arrow(
            ax_b,
            theta,
            mu,
            color_mean,
            lw=1.8 if is_observed else 1.3,
            alpha=0.9 if is_observed else 0.7,
            r=R * 1.03,
        )

        label_r = R + 0.60
        lx = label_r * np.cos(theta)
        ly = label_r * np.sin(theta)
        ax_b.text(
            lx,
            ly,
            f'{mu:+.1f}° ± {std:.1f}°',
            fontsize=6.5,
            color=color_mean,
            ha='center',
            va='center',
            fontweight='bold',
            alpha=0.85,
        )

    ax_b.set_title('B   Angular projection (arctan)', loc='left', fontweight='bold', fontsize=11)

    legend_b = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.3', ms=7, label='Target'),
        Line2D([0], [0], color=color_mean, lw=2, alpha=0.8, label='Mean prediction'),
        Line2D([0], [0], color=color_density, lw=6, alpha=0.3, label='Predictive density (±3σ)'),
    ]
    ax_b.legend(
        handles=legend_b, fontsize=6, frameon=False, loc='lower left', bbox_to_anchor=(-0.05, -0.08)
    )
    ax_b.axis('off')

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_aspect('equal')

    obs_mu = pred_means[0]
    obs_std = pred_stds[0]

    sweep = max(4.0 * obs_std, 2.0 * rotation_deg)
    arc_angles = np.linspace(
        obs_target_angle - np.radians(sweep), obs_target_angle + np.radians(sweep), 300
    )
    ax_c.plot(R * np.cos(arc_angles), R * np.sin(arc_angles), color='0.75', lw=1.2, ls='--')

    ax_c.plot(obs_x, obs_y, 'o', color='0.15', ms=14, zorder=20)
    ax_c.annotate(
        'target (0°)',
        xy=(obs_x, obs_y),
        xytext=(obs_x - 0.02, obs_y - 0.06),
        fontsize=8,
        color='0.3',
        ha='center',
        va='top',
    )

    draw_gaussian_on_circle(
        ax_c,
        obs_target_angle,
        obs_mu,
        obs_std,
        color_density,
        r_base=R,
        density_scale=0.20,
        alpha=0.30,
        n_pts=300,
    )

    draw_curved_arrow(ax_c, obs_target_angle, obs_mu, color_mean, lw=2.5, alpha=0.9, r=R * 1.015)

    for n_sig, ls, lw_tick in [(1, '--', 1.2), (2, ':', 0.9)]:
        for sign in [-1, 1]:
            tick_pert = obs_mu + sign * n_sig * obs_std
            tick_angle = obs_target_angle + np.radians(tick_pert)
            tick_r_inner = R * 0.95
            tick_r_outer = R * 1.05
            ax_c.plot(
                [tick_r_inner * np.cos(tick_angle), tick_r_outer * np.cos(tick_angle)],
                [tick_r_inner * np.sin(tick_angle), tick_r_outer * np.sin(tick_angle)],
                color=color_mean,
                lw=lw_tick,
                ls=ls,
                alpha=0.5,
            )

    for n_sig in [1, 2]:
        tick_pert = obs_mu + n_sig * obs_std
        tick_angle = obs_target_angle + np.radians(tick_pert)
        lx = R * 0.88 * np.cos(tick_angle)
        ly = R * 0.88 * np.sin(tick_angle)
        ax_c.text(
            lx, ly, f'+{n_sig}σ', fontsize=7, color=color_mean, ha='center', va='center', alpha=0.5
        )

        tick_pert_neg = obs_mu - n_sig * obs_std
        tick_angle_neg = obs_target_angle + np.radians(tick_pert_neg)
        lx_n = R * 0.88 * np.cos(tick_angle_neg)
        ly_n = R * 0.88 * np.sin(tick_angle_neg)
        ax_c.text(
            lx_n,
            ly_n,
            f'−{n_sig}σ',
            fontsize=7,
            color=color_mean,
            ha='center',
            va='center',
            alpha=0.5,
        )

    ax_c.text(
        0.03,
        0.03,
        f'μ = {obs_mu:+.1f}°\nσ = {obs_std:.1f}°\nvar = $H(\\theta)^\\top \\Sigma_T\\, H(\\theta)$',
        transform=ax_c.transAxes,
        fontsize=8,
        color=color_mean,
        va='bottom',
        ha='left',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.8', pad=3, boxstyle='round,pad=0.3'),
    )

    zoom_half = 0.35
    cx = R * np.cos(obs_target_angle + np.radians(rotation_deg / 2))
    cy = R * np.sin(obs_target_angle + np.radians(rotation_deg / 2))
    ax_c.set_xlim(cx - zoom_half, cx + zoom_half)
    ax_c.set_ylim(cy - zoom_half, cy + zoom_half)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('0.5')

    ax_c.set_title('C   Zoom: observed target', loc='left', fontweight='bold', fontsize=11)

    out_path = os.path.join(out_dir, 'gaussian_translation_prediction.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved to {out_path}")


def fourier_fields():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.gridspec import GridSpec
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.dpi': 300,
        }
    )

    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)

    R = 1.0
    n_targets = 8
    target_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    dc_rotation = 15

    obs_target_angle = 0.0
    fb_angle_rad = obs_target_angle + np.radians(dc_rotation)

    obs_x = R * np.cos(obs_target_angle)
    obs_y = R * np.sin(obs_target_angle)
    fb_x_obs = R * np.cos(fb_angle_rad)
    fb_y_obs = R * np.sin(fb_angle_rad)

    t_dx = fb_x_obs - obs_x
    t_dy = fb_y_obs - obs_y
    T_mag = np.sqrt(t_dx**2 + t_dy**2)
    T_dir = np.arctan2(t_dy, t_dx)

    print(f"T = ({t_dx:.3f}, {t_dy:.3f}), |T| = {T_mag:.3f}, dir = {np.degrees(T_dir):.1f}°")

    def translation_perturbation(theta, t_dx, t_dy):
        tx = R * np.cos(theta) + t_dx
        ty = R * np.sin(theta) + t_dy
        return np.degrees(np.arctan2(ty, tx) - theta)

    def wrap_deg(x):
        return (x + 180) % 360 - 180

    h1_pert = np.array([wrap_deg(translation_perturbation(th, t_dx, t_dy)) for th in target_angles])

    def draw_curved_arrow(ax, theta_start, pert_deg, color, lw=1.8, alpha=1.0, r=R, n_pts=30):
        if abs(pert_deg) < 0.3:
            return
        pert_rad = np.radians(pert_deg)
        theta_end = theta_start + pert_rad
        angles = np.linspace(theta_start, theta_end, n_pts)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle='butt', zorder=15)
        head_len = 0.06
        head_width = 0.04
        tip_x, tip_y = r * np.cos(theta_end), r * np.sin(theta_end)
        back_angle = theta_end - np.sign(pert_deg) * head_len / r
        base_x, base_y = r * np.cos(back_angle), r * np.sin(back_angle)
        norm_x, norm_y = np.cos(theta_end), np.sin(theta_end)
        tri_xs = [tip_x, base_x + head_width * norm_x, base_x - head_width * norm_x]
        tri_ys = [tip_y, base_y + head_width * norm_y, base_y - head_width * norm_y]
        ax.fill(tri_xs, tri_ys, color=color, alpha=alpha, zorder=15)

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.6)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_new = fig.add_subplot(gs[0, 2:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_d = fig.add_subplot(gs[1, 3:6])

    ax = ax_a
    ax.set_aspect('equal')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)
    for theta in target_angles:
        is_obs = abs(theta - obs_target_angle) < 0.01
        ax.plot(
            R * np.cos(theta),
            R * np.sin(theta),
            'o',
            color='0.15' if is_obs else '0.72',
            ms=10,
            zorder=7,
        )
        draw_curved_arrow(
            ax, theta, dc_rotation, '#2166AC', r=R * 1.1, alpha=1.0 if is_obs else 0.55
        )
    ax.plot(
        R * np.cos(fb_angle_rad), R * np.sin(fb_angle_rad), 'o', color='#E66100', ms=6, zorder=15
    )
    ax.set_title('A   True rotation', loc='left', fontweight='bold')
    ax.text(
        0, -1.52, 'Uniform angular shift\nat all targets', ha='center', fontsize=8, color='0.45'
    )
    ax.axis('off')

    ax = ax_new
    ax.set_aspect('equal')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

    for i, theta in enumerate(target_angles):
        tx, ty = R * np.cos(theta), R * np.sin(theta)
        is_observed = abs(theta - obs_target_angle) < 0.01

        ax.plot(tx, ty, 'o', color='0.15' if is_observed else '0.72', ms=10, zorder=7)

        fx, fy = tx + t_dx, ty + t_dy
        ax.annotate(
            '',
            xy=(fx, fy),
            xytext=(tx, ty),
            arrowprops=dict(
                arrowstyle='->',
                color='#B2182B',
                lw=1.5 if is_observed else 1.0,
                alpha=1.0 if is_observed else 0.4,
                mutation_scale=8,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=10,
        )

        if is_observed:
            ax.plot(fb_x_obs, fb_y_obs, 'o', color='#E66100', ms=6, zorder=15)

    ax.annotate(
        '',
        xy=(t_dx, t_dy),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='0.55', lw=1.2, linestyle='--', mutation_scale=10),
    )
    ax.text(
        t_dx - 0.12,
        t_dy / 2,
        '$\\hat{T}$',
        fontsize=10,
        color='0.5',
        fontstyle='italic',
        ha='right',
    )

    legend_new = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.3', ms=6, label='Target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E66100', ms=5, label='Feedback'),
        Line2D([0], [0], color='#B2182B', lw=1.5, label='$\\hat{T}$ at each target'),
    ]
    ax.legend(
        handles=legend_new,
        fontsize=5.5,
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
    )
    ax.set_title('B   Inferred translation', loc='left', fontweight='bold')
    ax.axis('off')

    ax = ax_b
    ax.set_aspect('equal')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

    ax.annotate(
        '',
        xy=(t_dx, t_dy),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='0.55', lw=1.2, linestyle='--', mutation_scale=10),
    )
    ax.text(
        t_dx - 0.12,
        t_dy / 2,
        '$\\hat{T}$',
        fontsize=10,
        color='0.5',
        fontstyle='italic',
        ha='right',
    )

    for i, theta in enumerate(target_angles):
        is_obs = abs(theta - obs_target_angle) < 0.01
        tx, ty = R * np.cos(theta), R * np.sin(theta)

        fx, fy = tx + t_dx, ty + t_dy
        ax.annotate(
            '',
            xy=(fx, fy),
            xytext=(tx, ty),
            arrowprops=dict(
                arrowstyle='->',
                color='#B2182B',
                lw=1.2 if is_obs else 0.8,
                alpha=0.8 if is_obs else 0.35,
                mutation_scale=7,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=8,
        )

        ax.plot(
            [0, tx], [0, ty], ls='-', color='0.7', lw=0.6, alpha=0.8 if is_obs else 0.3, zorder=4
        )

        proj_angle = np.arctan2(fy, fx)
        cx, cy = R * np.cos(proj_angle), R * np.sin(proj_angle)
        ax.plot(
            [0, cx], [0, cy], ls='--', color='0.45', lw=0.6, alpha=0.8 if is_obs else 0.3, zorder=5
        )

        from matplotlib.patches import Wedge

        ang1 = np.degrees(theta)
        ang2 = np.degrees(proj_angle)
        # Ensure we shade the short way round
        d_ang = (ang2 - ang1 + 180) % 360 - 180
        wedge_color = '#B2182B'
        wedge = Wedge(
            (0, 0),
            R,
            min(ang1, ang1 + d_ang),
            max(ang1, ang1 + d_ang),
            facecolor=wedge_color,
            edgecolor='none',
            alpha=0.18 if is_obs else 0.08,
            zorder=3,
        )
        ax.add_patch(wedge)

        ax.plot(tx, ty, 'o', color='0.15' if is_obs else '0.72', ms=10, zorder=10)

        pert = h1_pert[i]
        if abs(pert) < 0.5:
            ax.plot(
                R * 1.1 * np.cos(theta),
                R * 1.1 * np.sin(theta),
                'x',
                color='#B2182B',
                ms=5,
                mew=1.2,
                alpha=0.5,
            )
            continue
        draw_curved_arrow(ax, theta, pert, '#B2182B', r=R * 1.1, alpha=1.0 if is_obs else 0.55)

    ax.set_title('C   Projected rotation', loc='left', fontweight='bold')

    ax.axis('off')

    ax = ax_c
    theta_fine = np.linspace(0, 2 * np.pi, 500)

    ax.axhline(dc_rotation, color='#2166AC', lw=2, label='DC (global rotation)')

    h_exact = np.array([wrap_deg(translation_perturbation(th, t_dx, t_dy)) for th in theta_fine])
    ax.plot(
        np.degrees(theta_fine),
        h_exact,
        color='#B2182B',
        lw=2,
        label='Translation projection (exact)',
    )

    for i, theta in enumerate(target_angles):
        deg = np.degrees(theta) % 360
        ax.plot(deg, dc_rotation, 'o', color='#2166AC', ms=4, zorder=5)
        ax.plot(deg, h1_pert[i], 'o', color='#B2182B', ms=4, zorder=5)

    ax.annotate(
        f'{h1_pert[0]:.1f}°',
        xy=(0, h1_pert[0]),
        xytext=(18, h1_pert[0] - 5),
        fontsize=8,
        color='#B2182B',
        arrowprops=dict(arrowstyle='->', color='#B2182B', lw=0.8),
    )

    idx_180 = np.argmin(np.abs(target_angles - np.pi))
    ax.annotate(
        f'{h1_pert[idx_180]:.1f}°',
        xy=(180, h1_pert[idx_180]),
        xytext=(198, h1_pert[idx_180] + 5),
        fontsize=8,
        color='#B2182B',
        arrowprops=dict(arrowstyle='->', color='#B2182B', lw=0.8),
    )

    for i, theta in enumerate(target_angles):
        if abs(h1_pert[i]) < 3.0 and abs(theta) > 0.1:
            deg = np.degrees(theta) % 360
            ax.annotate(
                f'{h1_pert[i]:.1f}°',
                xy=(deg, h1_pert[i]),
                xytext=(deg + 15, h1_pert[i] + 4),
                fontsize=8,
                color='#B2182B',
            )

    ax.set_xlabel('Target direction (°)')
    ax.set_ylabel('Angular perturbation (°)')
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xticklabels(['0', '', '90', '', '180', '', '270', '', '360'])
    ax.set_xlim(0, 360)
    ax.set_ylim(-21, 38)
    ax.axhline(0, color='0.85', lw=0.5, zorder=0)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax.set_title('D   Perturbation by target direction', loc='left', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = ax_d
    ax.set_aspect('equal')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
    ax.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

    obs_theta = 0.0
    obs_pert = dc_rotation
    obs_tx, obs_ty = R * np.cos(obs_theta), R * np.sin(obs_theta)
    fb_theta = obs_theta + np.radians(obs_pert)
    fb_x, fb_y = R * np.cos(fb_theta), R * np.sin(fb_theta)

    ax.plot(obs_tx, obs_ty, 'o', color='0.15', ms=10, zorder=10)
    ax.plot(fb_x, fb_y, 'o', color='#E66100', ms=6, zorder=10)

    ax.annotate(
        'observed',
        xy=(obs_tx + 0.05, obs_ty - 0.15),
        fontsize=7.5,
        color='0.3',
        ha='left',
        va='top',
    )

    r_arrow = R * 1.12
    dr = 0.07
    r_dc = r_arrow + dr
    r_h1 = r_arrow - dr

    for theta in target_angles:
        if abs(theta - obs_theta) < 0.01:
            pass
        else:
            ax.plot(R * np.cos(theta), R * np.sin(theta), 'o', color='0.72', ms=10, zorder=4)

        draw_curved_arrow(ax, theta, dc_rotation, '#2166AC', lw=1.3, alpha=0.55, r=r_dc)

        idx = np.argmin(np.abs(target_angles - theta))
        pert_here = h1_pert[idx]
        if abs(pert_here) > 0.5:
            draw_curved_arrow(ax, theta, pert_here, '#B2182B', lw=1.3, alpha=0.55, r=r_h1)
        else:
            ax.plot(
                r_h1 * np.cos(theta),
                r_h1 * np.sin(theta),
                'x',
                color='#B2182B',
                ms=5,
                mew=1.2,
                alpha=0.5,
            )

    ax.text(1.6, 0.35, 'identical', fontsize=7, color='0.4', ha='center', fontstyle='italic')

    theta_180 = np.pi
    bx = (R + 0.75) * np.cos(theta_180)
    by = (R + 0.75) * np.sin(theta_180)
    ax.text(
        bx,
        by + 0.12,
        f'DC: +{dc_rotation}°',
        fontsize=7,
        color='#2166AC',
        ha='right',
        va='bottom',
        fontweight='bold',
    )
    ax.text(
        bx,
        by - 0.10,
        f'Trans: {h1_pert[idx_180]:+.1f}°',
        fontsize=7,
        color='#B2182B',
        ha='right',
        va='bottom',
        fontweight='bold',
    )

    idx_90 = np.argmin(np.abs(target_angles - np.pi / 2))
    theta_90 = np.pi / 2
    bx = (R + 0.6) * np.cos(theta_90)
    by = (R + 0.6) * np.sin(theta_90)
    ax.text(
        bx + 0.12,
        by,
        f'DC: +{dc_rotation}°',
        fontsize=7,
        color='#2166AC',
        ha='left',
        va='bottom',
        fontweight='bold',
    )
    ax.text(
        bx + 0.12,
        by - 0.20,
        f'Trans: {h1_pert[idx_90]:+.1f}°',
        fontsize=7,
        color='#B2182B',
        ha='left',
        va='bottom',
        fontweight='bold',
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.15', ms=6, label='Target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E66100', ms=5, label='Feedback'),
        Line2D([0], [0], color='#2166AC', lw=1.5, alpha=0.6, label='DC hypothesis (outer)'),
        Line2D([0], [0], color='#B2182B', lw=1.5, alpha=0.6, label='Translation hyp. (inner)'),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=7,
        frameon=False,
        loc='lower left',
        bbox_to_anchor=(-0.08, -0.06),
    )
    ax.set_title('E   Single observation: ambiguous', loc='left', fontweight='bold')
    ax.axis('off')

    out_path = os.path.join(out_dir, 'fourier_dc_vs_h1.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved to {out_path}")


def gamma_sweep():

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import math
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.dpi': 300,
        }
    )

    N_HARM = 1024
    DIM = 2 * N_HARM + 1  # 2049: [DC, sin1, cos1, ..., sin1024, cos1024]

    def featureVector(theta_rad):
        """Fourier feature vector.  Vectorised: theta_rad can be an array."""
        theta_rad = np.atleast_1d(theta_rad)
        N = len(theta_rad)
        f = np.empty((N, DIM))
        f[:, 0] = 1.0
        harmonics = np.arange(1, N_HARM + 1)  # (N_HARM,)
        angles = np.outer(theta_rad, harmonics)  # (N, N_HARM)
        f[:, 1::2] = np.sin(angles)  # sin columns: 1, 3, 5, ...
        f[:, 2::2] = np.cos(angles)  # cos columns: 2, 4, 6, ...
        return f.squeeze()  # return 1D if scalar input

    def buildPriorDiag(logVarDC, logVarTrans, gamma=1.0):
        """
        Diagonal prior covariance as a 1D array of length DIM.
        Mirrors buildPriorDiag in the main model code.
        """
        dc_var = np.exp(np.clip(logVarDC, -700, 700))
        s2 = np.exp(np.clip(logVarTrans, -700, 700))
        sigma_T = math.sqrt(s2)
        RAD2DEG_SQ = (180.0 / math.pi) ** 2

        diag = np.zeros(DIM)
        diag[0] = dc_var

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
                    hn_sq = (2.0 - eps ** (-n)) ** 2
                integral += hn_sq * p_eps * d_eps
            v_n = RAD2DEG_SQ * integral / (2.0 * n ** (2.0 * gamma))
            diag[2 * n - 1] = v_n
            diag[2 * n] = v_n

        return diag

    def scalarBLR_one_update(priorDiag, f_obs, y, obsVar):
        """
        Rank-1 scalar BLR: one update from prior (c=0, d=0).

        Returns (c, d, v) where:
            v = priorDiag * f_obs          (DIM-length vector)
            s = f_obs @ v                  (prior predictive variance)
            c = 1 / (s + obsVar)           (covariance scalar)
            d = y / (s + obsVar)           (mean scalar)

        Posterior at any angle θ:
            crossDot = f(θ) @ v
            pred_mean = d * crossDot
            pred_var  = f(θ) @ (priorDiag * f(θ)) - c * crossDot²
        """
        v = priorDiag * f_obs
        s = f_obs @ v
        denom = s + obsVar
        c = 1.0 / denom
        d = y / denom
        return c, d, v

    def predict_at_angles(c, d, v, priorDiag, theta_arr):
        """
        Vectorised prediction at an array of angles.

        Returns (pred_mean, pred_std) arrays.
        """
        F = featureVector(theta_arr)  # (N, DIM)
        crossDots = F @ v  # (N,)  f(θ)' v
        pred_mean = d * crossDots  # (N,)

        fSf = (F**2) @ priorDiag  # (N,)
        pred_var = fSf - c * crossDots**2  # (N,)
        pred_var = np.maximum(pred_var, 0.0)
        return pred_mean, np.sqrt(pred_var)

    R = 1.0
    n_targets = 8
    target_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

    obs_theta = 0.0
    obs_pert_deg = 15.0
    obs_var = 30  # 1e-10            # ← force Kalman gain ≈ 1 (posterior passes through obs)

    logVarDC = -700.0  # DC off — no global rotation
    logVarTrans = np.log(0.15)

    gammas = [0.1, 0.5, 0.75, 1.0, 5.0]

    n_gamma = len(gammas)
    _cmap = plt.cm.winter
    gamma_colors = [_cmap(i / max(n_gamma - 1, 1)) for i in range(n_gamma)]

    theta_fine = np.sort(np.concatenate([np.linspace(-np.pi, np.pi, 500), [obs_theta]]))

    f_obs = featureVector(obs_theta)  # (DIM,) — shared across gammas

    predictions = {}

    for gamma in gammas:
        priorDiag = buildPriorDiag(logVarDC, logVarTrans, gamma)

        c, d, v = scalarBLR_one_update(priorDiag, f_obs, obs_pert_deg, obs_var)

        pert_curve, epi_std = predict_at_angles(c, d, v, priorDiag, theta_fine)
        pert_targets, _ = predict_at_angles(c, d, v, priorDiag, target_angles)

        predictions[gamma] = dict(
            pert_targets=pert_targets,
            pert_curve=pert_curve,
            epi_std=epi_std,
            prior_diag=priorDiag,
        )

    # Target angles wrapped to [-180, 180] for bottom-row plotting
    target_angles_plot = np.degrees(target_angles)
    target_angles_plot = np.where(
        target_angles_plot > 180, target_angles_plot - 360, target_angles_plot
    )
    target_sort = np.argsort(target_angles_plot)

    circ_size = 2.6
    circ_pad_x = 0.15
    bottom_h = 2.8
    row_gap = 0.3

    fig_w = n_gamma * circ_size + (n_gamma - 1) * circ_pad_x + 0.8
    fig_h = circ_size + row_gap + bottom_h + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    def in2frac_x(inches):
        return inches / fig_w

    def in2frac_y(inches):
        return inches / fig_h

    margin_left = 0.4
    margin_top = 0.35
    margin_bottom = 0.55

    col_lefts = [margin_left + col * (circ_size + circ_pad_x) for col in range(n_gamma)]

    row0_top = fig_h - margin_top
    row0_bot = row0_top - circ_size

    obs_fb_angle = obs_theta + np.radians(obs_pert_deg)

    def draw_curved_arrow(ax, theta_start, pert_deg, color, lw=1.8, alpha=1.0, r=R, n_pts=30):
        if abs(pert_deg) < 0.3:
            return
        pert_rad = np.radians(pert_deg)
        theta_end = theta_start + pert_rad
        angles = np.linspace(theta_start, theta_end, n_pts)
        ax.plot(
            r * np.cos(angles),
            r * np.sin(angles),
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle='butt',
            zorder=15,
        )
        head_len, head_width = 0.06, 0.04
        tip_x, tip_y = r * np.cos(theta_end), r * np.sin(theta_end)
        back_angle = theta_end - np.sign(pert_deg) * head_len / r
        base_x, base_y = r * np.cos(back_angle), r * np.sin(back_angle)
        nx, ny = np.cos(theta_end), np.sin(theta_end)
        ax.fill(
            [tip_x, base_x + head_width * nx, base_x - head_width * nx],
            [tip_y, base_y + head_width * ny, base_y - head_width * ny],
            color=color,
            alpha=alpha,
            zorder=15,
        )

    for col in range(n_gamma):
        gamma = gammas[col]
        color = gamma_colors[col]
        rect = [
            in2frac_x(col_lefts[col]),
            in2frac_y(row0_bot),
            in2frac_x(circ_size),
            in2frac_y(circ_size),
        ]
        ax = fig.add_axes(rect)
        ax.set_aspect('equal')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)

        ax.add_patch(plt.Circle((0, 0), R, fill=False, color='0.75', lw=0.8, ls='--'))
        ax.plot(0, 0, '+', color='0.6', ms=8, mew=0.8)

        pert_targets = predictions[gamma]['pert_targets']

        for i, theta in enumerate(target_angles):
            is_obs = abs(theta - obs_theta) < 0.01
            ax.plot(
                R * np.cos(theta),
                R * np.sin(theta),
                'o',
                color='0.15' if is_obs else '0.3',
                ms=11 if is_obs else 9,
                zorder=7,
            )

            pert = pert_targets[i]
            draw_curved_arrow(
                ax,
                theta,
                pert,
                color,
                lw=2.0 if is_obs else 1.4,
                alpha=1.0 if is_obs else 0.7,
                r=R * 1.1,
            )

            label_r = R * 1.38
            label_angle = theta + np.radians(pert) * 0.6
            ax.text(
                label_r * np.cos(label_angle),
                label_r * np.sin(label_angle),
                f'{pert:.1f}°',
                fontsize=5.5,
                color=color,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
            )

        fb_angle = obs_theta + np.radians(obs_pert_deg)
        ax.plot(
            R * np.cos(fb_angle),
            R * np.sin(fb_angle),
            '*',
            color='#E66100',
            ms=12,
            zorder=15,
            markeredgecolor='white',
            markeredgewidth=0.5,
        )
        ax.axis('off')

        diag = predictions[gamma]['prior_diag']
        harm_var = [diag[2 * n - 1] for n in range(1, 5)]
        inset = ax.inset_axes([0.02, 0.02, 0.35, 0.22])
        inset.bar(range(1, 5), harm_var, color=color, alpha=0.6, width=0.7)
        inset.set_xticks(range(1, 5))
        inset.set_xticklabels(['h1', 'h2', 'h3', 'h4'], fontsize=5)
        inset.set_ylabel('σ²', fontsize=5, labelpad=1)
        inset.tick_params(axis='both', labelsize=4, length=2)
        inset.set_title('Prior spectrum', fontsize=5, pad=2)
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)

        if col == 0:
            ax.text(
                -1.95,
                2.15,
                'Predicted angular perturbation',
                fontsize=7.5,
                color='0.4',
                fontstyle='italic',
                va='top',
            )

    from matplotlib.gridspec import GridSpec

    bot_top = row0_bot - row_gap
    gs_bot = GridSpec(
        1,
        n_gamma,
        figure=fig,
        left=in2frac_x(margin_left),
        right=in2frac_x(col_lefts[-1] + circ_size),
        bottom=in2frac_y(margin_bottom),
        top=in2frac_y(bot_top),
        wspace=0.35,
    )

    split = max(n_gamma // 2 + 1, 2)

    ax_curve = fig.add_subplot(gs_bot[0, :split])

    for col in range(n_gamma):
        gamma = gammas[col]
        color = gamma_colors[col]
        ax_curve.plot(
            np.degrees(theta_fine),
            predictions[gamma]['pert_curve'],
            color=color,
            lw=2,
            label=f'γ = {gamma}',
            alpha=0.85,
        )
        for i in target_sort:
            ax_curve.plot(
                target_angles_plot[i],
                predictions[gamma]['pert_targets'][i],
                'o',
                color=color,
                ms=3.5,
                zorder=5,
            )

    cosine_curve = obs_pert_deg * np.cos(theta_fine)
    ax_curve.plot(
        np.degrees(theta_fine),
        cosine_curve,
        color='0.45',
        lw=1.5,
        ls='--',
        label='cos(Δθ)',
        alpha=0.8,
        zorder=1,
    )

    ax_curve.axhline(0, color='0.85', lw=0.5, zorder=0)
    ax_curve.axhline(
        obs_pert_deg, color='0.6', lw=0.8, ls=':', zorder=0, label=f'Observed ({obs_pert_deg}°)'
    )
    ax_curve.axvline(np.degrees(obs_theta), color='0.6', lw=0.5, ls=':', zorder=0)
    ax_curve.set_xlabel('Target direction (°)')
    ax_curve.set_ylabel('Predicted perturbation (°)')
    ax_curve.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax_curve.set_xticklabels(['-180', '', '-90', '', '0', '', '90', '', '180'])
    ax_curve.set_xlim(-180, 180)
    ax_curve.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper right')
    ax_curve.set_title('Predicted perturbation by target direction', loc='left', fontweight='bold')
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['right'].set_visible(False)

    ax_band = fig.add_subplot(gs_bot[0, split:])

    band_indices = [1, 3, 4]  # sorted(set([0, n_gamma // 2, n_gamma - 1]))

    for idx in band_indices:
        gamma = gammas[idx]
        color = gamma_colors[idx]
        pert = predictions[gamma]['pert_curve']
        sd = predictions[gamma]['epi_std']
        deg = np.degrees(theta_fine)
        ax_band.fill_between(deg, pert - sd, pert + sd, color=color, alpha=0.09, zorder=1)
        ax_band.plot(deg, pert, color=color, lw=2, label=f'γ = {gamma}', alpha=0.85, zorder=3)

    ax_band.axhline(0, color='0.85', lw=0.5, zorder=0)
    ax_band.axhline(obs_pert_deg, color='0.6', lw=0.8, ls=':', zorder=0)
    ax_band.axvline(np.degrees(obs_theta), color='0.6', lw=0.5, ls=':', zorder=0)
    ax_band.set_xlabel('Target direction (°)')
    ax_band.set_ylabel('Predicted perturbation (°)')
    ax_band.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax_band.set_xticklabels(['-180', '', '-90', '', '0', '', '90', '', '180'])
    ax_band.set_xlim(-180, 180)
    ax_band.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax_band.set_title('Posterior mean ± 1 SD (epistemic only)', loc='left', fontweight='bold')
    ax_band.spines['top'].set_visible(False)
    ax_band.spines['right'].set_visible(False)

    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gamma_sweep.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved to {out_path}")


def local_global_fields():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.dpi': 300,
        }
    )
    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)
    DIM = 3  # [1, sin(theta), cos(theta)]
    R = 1.0
    n_targets = 8
    target_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    targets_deg = np.degrees(target_angles)
    obs_pert_deg = 15.0
    KAPPA = 5.0
    SIGMA_DEG = 5.0
    OBS_NOISE_DEG = 0.0
    SIGMA_T = 0.25  # Rayleigh scale for translation magnitude (units of R)

    # Symmetry gives E[delta] = 0, so covariance is E[delta_i * delta_j].
    # Integrate over uniform direction and Rayleigh-distributed translation magnitude.
    N_QUAD_PHI = 500
    N_QUAD_EPS = 300
    eps_max = max(6.0 * SIGMA_T, 3.0)

    phi_pts = np.linspace(0, 2 * np.pi, N_QUAD_PHI, endpoint=False)
    d_phi = 2 * np.pi / N_QUAD_PHI
    eps_pts = np.linspace(1e-8, eps_max, N_QUAD_EPS)
    d_eps = eps_pts[1] - eps_pts[0]

    rayleigh_pdf = (eps_pts / SIGMA_T**2) * np.exp(-(eps_pts**2) / (2 * SIGMA_T**2))

    def arctan_pert_deg(theta, eps, phi):
        """Exact arctan projection of translation (eps, phi) at target theta, in degrees."""
        arg_sin = eps * np.sin(theta - phi)
        arg_cos = 1.0 + eps * np.cos(theta - phi)
        return np.degrees(np.arctan2(arg_sin, arg_cos))

    # Build delta[eps_idx, phi_idx, target_idx]
    delta_grid = np.zeros((N_QUAD_EPS, N_QUAD_PHI, n_targets))
    for ti, theta in enumerate(target_angles):
        for ei, eps in enumerate(eps_pts):
            for pi_idx, phi in enumerate(phi_pts):
                delta_grid[ei, pi_idx, ti] = arctan_pert_deg(theta, eps, phi)

    cov_matrix = np.zeros((n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            integrand_eps = np.zeros(N_QUAD_EPS)
            for ei in range(N_QUAD_EPS):
                prod_over_phi = delta_grid[ei, :, i] * delta_grid[ei, :, j]
                integrand_eps[ei] = np.mean(prod_over_phi)  # uniform average over phi
            cov_matrix[i, j] = np.sum(integrand_eps * rayleigh_pdf * d_eps)

    var_diag = np.diag(cov_matrix)
    corr_matrix = np.zeros((n_targets, n_targets))
    r2_matrix = np.zeros((n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(var_diag[i] * var_diag[j])
            r2_matrix[i, j] = corr_matrix[i, j] ** 2

    print("Exact arctan-derived covariance diagonal (variances in deg^2):")
    for k in range(n_targets):
        print(f"  Target {targets_deg[k]:5.1f} deg: var={var_diag[k]:.4f}")

    query_idx = 0
    kappa = KAPPA
    gibbsW_row = np.exp(kappa * r2_matrix[query_idx])
    weights = gibbsW_row / gibbsW_row.sum()

    print(f"Prior: sigma_T={SIGMA_T}")
    print(f"kappa = {kappa}")
    print("R^2 and Gibbs weights for query target 0 deg:")
    for k in range(n_targets):
        print(
            f"  Source {targets_deg[k]:5.1f} deg: corr={corr_matrix[query_idx, k]:+.3f}  "
            f"R^2={r2_matrix[query_idx, k]:.4f}  w={weights[k]:.4f}"
        )

    def get_translation_for_target(theta_k, pert_deg=obs_pert_deg):
        tgt_x = R * np.cos(theta_k)
        tgt_y = R * np.sin(theta_k)
        fb_angle = theta_k + np.radians(pert_deg)
        fb_x = R * np.cos(fb_angle)
        fb_y = R * np.sin(fb_angle)
        return fb_x - tgt_x, fb_y - tgt_y

    def translation_perturbation_vec(theta, t_dx, t_dy):
        u = R * np.cos(theta) + t_dx
        v = R * np.sin(theta) + t_dy
        raw = np.degrees(np.arctan2(v, u) - theta)
        return (raw + 180) % 360 - 180

    context_T = np.array([get_translation_for_target(th) for th in target_angles])

    cmap = plt.cm.tab10
    target_colors = [cmap(i) for i in range(n_targets)]

    def draw_curved_arrow(
        ax,
        theta_start,
        pert_deg,
        color,
        lw=1.8,
        alpha=1.0,
        r=1.0,
        cx=0,
        cy=0,
        n_pts=30,
        head_len=0.06,
        head_width=0.04,
    ):
        if abs(pert_deg) < 0.5:
            return
        pert_rad = np.radians(pert_deg)
        theta_end = theta_start + pert_rad
        angles = np.linspace(theta_start, theta_end, n_pts)
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle='butt', zorder=15)
        tip_x = cx + r * np.cos(theta_end)
        tip_y = cy + r * np.sin(theta_end)
        back_angle = theta_end - np.sign(pert_deg) * head_len / r
        base_x = cx + r * np.cos(back_angle)
        base_y = cy + r * np.sin(back_angle)
        norm_x, norm_y = np.cos(theta_end), np.sin(theta_end)
        tri_xs = [tip_x, base_x + head_width * norm_x, base_x - head_width * norm_x]
        tri_ys = [tip_y, base_y + head_width * norm_y, base_y - head_width * norm_y]
        ax.fill(tri_xs, tri_ys, color=color, alpha=alpha, zorder=15)

    loo_indices = [k for k in range(n_targets) if k != query_idx]
    loo_source_weights = weights.copy()
    loo_source_weights[query_idx] = 0.0
    loo_source_weights /= loo_source_weights.sum()

    fig = plt.figure(figsize=(16, 10))
    outer_gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.1, 0.8, 1.0],
        height_ratios=[2.5, 1.5],
        hspace=0.38,
        wspace=0.35,
    )
    ax_a = fig.add_subplot(outer_gs[0, 0])
    ax_a.set_aspect('equal')
    ax_a.set_xlim(-3.5, 3.5)
    ax_a.set_ylim(-3.5, 3.5)
    main_r = 1.3
    ax_a.add_patch(plt.Circle((0, 0), main_r, fill=False, color='0.8', lw=0.6, ls='--'))
    ax_a.plot(0, 0, '+', color='0.7', ms=6, mew=0.6)
    mini_center_r = 2.5
    mini_r = 0.55
    mini_target_ms = 3
    for k, theta_k in enumerate(target_angles):
        col = target_colors[k]
        tdx_k, tdy_k = context_T[k]
        w_k = weights[k]
        tx = main_r * np.cos(theta_k)
        ty = main_r * np.sin(theta_k)
        ms_main = 6 + 12 * (w_k / weights.max())
        main_alpha = 0.35 + 0.65 * (w_k / weights.max())
        ax_a.plot(tx, ty, 'o', color=col, ms=ms_main, zorder=7, alpha=main_alpha)
        cx = mini_center_r * np.cos(theta_k)
        cy = mini_center_r * np.sin(theta_k)
        conn_start_r = main_r + 0.12
        conn_end_r = mini_center_r - mini_r - 0.05
        ax_a.plot(
            [conn_start_r * np.cos(theta_k), conn_end_r * np.cos(theta_k)],
            [conn_start_r * np.sin(theta_k), conn_end_r * np.sin(theta_k)],
            color=col,
            lw=0.6,
            alpha=0.25 + 0.5 * w_k / weights.max(),
            ls='-',
        )
        mini_alpha = 0.3 + 0.7 * (w_k / weights.max())
        ax_a.add_patch(plt.Circle((cx, cy), mini_r, fill=False, color='0.8', lw=0.5, ls='--'))
        for j, theta_j in enumerate(target_angles):
            mx = cx + mini_r * np.cos(theta_j)
            my = cy + mini_r * np.sin(theta_j)
            ax_a.plot(mx, my, 'o', color='0.6', ms=mini_target_ms, zorder=5)
            pert_j = translation_perturbation_vec(theta_j, tdx_k, tdy_k)
            if abs(pert_j) < 0.5:
                continue
            draw_curved_arrow(
                ax_a,
                theta_j,
                pert_j,
                col,
                lw=1.0,
                alpha=mini_alpha * 0.8,
                r=mini_r * 1.15,
                cx=cx,
                cy=cy,
                head_len=0.04,
                head_width=0.025,
            )
        self_mx = cx + mini_r * np.cos(theta_k)
        self_my = cy + mini_r * np.sin(theta_k)
        ax_a.plot(self_mx, self_my, 'o', color=col, ms=mini_target_ms + 2, zorder=8)
        label_r = mini_center_r + mini_r + 0.22
        lx = label_r * np.cos(theta_k)
        ly = label_r * np.sin(theta_k)
        ha = 'center'
        va = 'center'
        if np.cos(theta_k) > 0.3:
            ha = 'left'
        elif np.cos(theta_k) < -0.3:
            ha = 'right'
        if np.sin(theta_k) > 0.3:
            va = 'bottom'
        elif np.sin(theta_k) < -0.3:
            va = 'top'
        ax_a.text(
            lx,
            ly,
            f'{int(targets_deg[k])}$^\\circ$  w={w_k:.3f}',
            fontsize=5.5,
            color=col,
            ha=ha,
            va=va,
            fontweight='bold',
        )
    ax_a.set_title('A   Per-target learned perturbation fields', loc='left', fontweight='bold')
    ax_a.text(
        0,
        -3.35,
        'Size/opacity proportional to Gibbs weight (prior R$^2$)',
        ha='center',
        fontsize=7.5,
        color='0.45',
    )
    ax_a.axis('off')

    ax_b = fig.add_subplot(outer_gs[0, 1])
    r2_vals = r2_matrix[query_idx]

    loo_labels = [f'{int(targets_deg[k])}$^\\circ$' for k in loo_indices]
    loo_w = np.array([loo_source_weights[k] for k in loo_indices])
    loo_r2 = np.array([r2_vals[k] for k in loo_indices])
    loo_cols = [target_colors[k] for k in loo_indices]

    x_pos_b = np.arange(len(loo_indices))
    bars = ax_b.bar(
        x_pos_b,
        loo_w,
        color=loo_cols,
        edgecolor='white',
        linewidth=0.6,
        width=0.75,
        alpha=0.85,
        zorder=3,
    )
    for bi in range(len(loo_indices)):
        ax_b.text(
            bi,
            loo_w[bi] + 0.008,
            f'{loo_r2[bi]:.2f}',
            ha='center',
            va='bottom',
            fontsize=7,
            color='0.5',
            fontweight='normal',
        )
    ax_b.set_xticks(x_pos_b)
    ax_b.set_xticklabels(loo_labels, fontsize=7.5)
    ax_b.set_xlabel('Source target')
    ax_b.set_ylabel('LOO Gibbs weight  $\\tilde{g}_j$')
    ax_b.set_ylim(0, loo_w.max() * 1.28)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.set_title('B   LOO source weights from prior R$^2$', loc='left', fontweight='bold')
    ax_b.text(
        0.97,
        0.93,
        r'$\tilde{g}_j \propto \exp(\kappa \cdot \rho^2_{0j}),\; j \neq i$',
        transform=ax_b.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        color='0.4',
    )
    ax_b.text(
        0.97,
        0.82,
        f'$\\kappa$ = {kappa:.0f}',
        transform=ax_b.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        color='0.55',
    )
    ax_b.text(
        0.97,
        0.72,
        'R$^2$ labels above bars',
        transform=ax_b.transAxes,
        ha='right',
        va='top',
        fontsize=6.5,
        color='0.7',
        style='italic',
    )

    ax_c = fig.add_subplot(outer_gs[0, 2])
    query_theta = 0.0
    sigma_component = SIGMA_DEG
    preds_at_query = np.array(
        [
            translation_perturbation_vec(query_theta, context_T[k, 0], context_T[k, 1])
            for k in range(n_targets)
        ]
    )
    x_pert = np.linspace(-180, 180, 6000)

    def gaussian_density(x, mu, sigma):
        g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return g / (sigma * np.sqrt(2 * np.pi))

    self_mean = preds_at_query[query_idx]
    self_sigma = sigma_component

    # The outer self/global mixture is distinct from the Gibbs weights inside the LOO field.
    w_self_mix = 0.5
    w_loo_mix = 1.0 - w_self_mix

    self_density = w_self_mix * gaussian_density(x_pert, self_mean, self_sigma)

    # Build LOO global as explicit mixture of individual per-target Gaussians.
    # Each source j != query contributes: loo_source_weights[j] * N(preds_at_query[j], sigma^2)
    loo_density = np.zeros_like(x_pert)
    individual_contributions = {}
    for k in loo_indices:
        comp_density = (
            w_loo_mix
            * loo_source_weights[k]
            * gaussian_density(x_pert, preds_at_query[k], sigma_component)
        )
        individual_contributions[k] = comp_density
        loo_density += comp_density

    mixture_density = self_density + loo_density

    for k in loo_indices:
        comp = individual_contributions[k]
        ax_c.fill_between(x_pert, comp, color=target_colors[k], alpha=0.08, zorder=1)
        ax_c.plot(x_pert, comp, color=target_colors[k], lw=1.0, alpha=0.6, zorder=2)

    ax_c.plot(x_pert, self_density, color=target_colors[query_idx], lw=2.2, zorder=4)
    ax_c.plot(x_pert, loo_density, color='0.45', lw=2.2, ls='--', zorder=4)
    ax_c.fill_between(x_pert, mixture_density, color='0.15', alpha=0.1, zorder=3)
    ax_c.plot(x_pert, mixture_density, color='0.15', lw=2.6, zorder=5)

    y_top = mixture_density.max()
    ax_c.axvline(obs_pert_deg, color='0.4', lw=1.2, ls='--', alpha=0.7)
    ax_c.axvline(-obs_pert_deg, color='0.4', lw=1.2, ls='--', alpha=0.7)
    ax_c.text(
        obs_pert_deg + 2,
        y_top * 0.95,
        f'+{int(obs_pert_deg)}$^\\circ$\n(ideal)',
        fontsize=7.5,
        color='0.4',
        va='top',
        ha='left',
    )
    ax_c.text(
        -obs_pert_deg - 2,
        y_top * 0.95,
        f'-{int(obs_pert_deg)}$^\\circ$\n(sign-flip)',
        fontsize=7.5,
        color='0.4',
        va='top',
        ha='right',
    )
    ax_c.text(
        0.97,
        0.94,
        'Illustrative 50/50\nself-global mix',
        transform=ax_c.transAxes,
        ha='right',
        va='top',
        fontsize=6.5,
        color='0.55',
    )

    other_preds = [(preds_at_query[k], k) for k in loo_indices]
    other_preds.sort(key=lambda x: x[0])
    loo_max = max(loo_source_weights.max(), 1e-12)
    for k in loo_indices:
        tick_alpha = max(0.3, 0.3 + 0.7 * loo_source_weights[k] / loo_max)
        ax_c.plot(
            preds_at_query[k],
            -y_top * 0.03,
            '|',
            color=target_colors[k],
            ms=6 + 12 * loo_source_weights[k] / loo_max,
            mew=1.5,
            alpha=tick_alpha,
        )
    ax_c.plot(
        self_mean, -y_top * 0.03, '|', color=target_colors[query_idx], ms=12, mew=1.8, alpha=0.95
    )
    placed = []
    base_y = -y_top * 0.075
    alt_y = -y_top * 0.125
    for pred_val, k in other_preds:
        x_pos_label = pred_val
        y_pos = base_y
        for px, py in placed:
            if abs(x_pos_label - px) < 6 and abs(y_pos - py) < y_top * 0.06:
                y_pos = alt_y
                break
        placed.append((x_pos_label, y_pos))
        ax_c.text(
            x_pos_label,
            y_pos,
            f'{int(targets_deg[k])}$^\\circ$',
            fontsize=5.5,
            color=target_colors[k],
            ha='center',
            va='top',
            fontweight='bold',
        )
    ax_c.text(
        self_mean,
        alt_y,
        'self',
        fontsize=6.5,
        color=target_colors[query_idx],
        ha='center',
        va='top',
        fontweight='bold',
    )

    ax_c.set_xlabel('Predicted perturbation at 0$^\\circ$ target ($^\\circ$)')
    ax_c.set_ylabel('Density')
    ax_c.set_xlim(-45, 45)
    ax_c.set_ylim(bottom=-y_top * 0.16, top=y_top * 1.1)
    ax_c.axvline(0, color='0.85', lw=0.5, zorder=0)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=target_colors[query_idx], lw=2.2, label='Self component'),
    ]
    for k in loo_indices:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=target_colors[k],
                lw=1.0,
                alpha=0.6,
                label=f'{int(targets_deg[k])}$^\\circ$ ($\\tilde{{g}}$={loo_source_weights[k]:.2f})',
            )
        )
    legend_handles.extend(
        [
            Line2D([0], [0], color='0.45', lw=2.2, ls='--', label='LOO global (sum)'),
            Line2D([0], [0], color='0.15', lw=2.6, label='Final mixture'),
            Line2D(
                [0],
                [0],
                color='0.4',
                lw=1.2,
                ls='--',
                alpha=0.7,
                label='$\\pm$ rotation (ideal / sign-flip)',
            ),
        ]
    )
    ax_c.legend(handles=legend_handles, fontsize=5.5, frameon=False, loc='upper left', ncol=2)
    ax_c.set_title(
        'C   Individual field contributions to LOO global at 0$^\\circ$',
        loc='left',
        fontweight='bold',
    )

    # R² measures prior variance reduction from observing another target.
    from matplotlib.colors import TwoSlopeNorm

    heatmap_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs[1, :], wspace=0.35, width_ratios=[1, 1, 1]
    )

    target_labels = [f'{int(d)}$^\\circ$' for d in targets_deg]

    ax_cov = fig.add_subplot(heatmap_gs[0])
    cov_max = np.max(np.abs(cov_matrix))
    if cov_max < 1e-12:
        cov_max = 1.0
    norm_cov = TwoSlopeNorm(vmin=-cov_max, vcenter=0, vmax=cov_max)
    im_cov = ax_cov.imshow(cov_matrix, cmap='RdBu_r', norm=norm_cov, aspect='equal', origin='upper')
    ax_cov.set_xticks(range(n_targets))
    ax_cov.set_xticklabels(target_labels, fontsize=7)
    ax_cov.set_yticks(range(n_targets))
    ax_cov.set_yticklabels(target_labels, fontsize=7)
    ax_cov.set_xlabel('Target $k$', fontsize=9)
    ax_cov.set_ylabel('Target $i$', fontsize=9)
    ax_cov.set_title(r'$\mathrm{Cov}(\delta_i, \delta_k)$  (prior)', fontsize=10, fontweight='bold')
    for i in range(n_targets):
        for j in range(n_targets):
            val = cov_matrix[i, j]
            text_col = 'white' if cov_max > 0 and abs(val) / cov_max > 0.55 else 'black'
            ax_cov.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5, color=text_col)
    ax_cov.axhline(query_idx - 0.5, color=target_colors[query_idx], lw=2)
    ax_cov.axhline(query_idx + 0.5, color=target_colors[query_idx], lw=2)
    plt.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)

    fig.text(
        0.385,
        0.19,
        r'$\longrightarrow$  normalise  $\longrightarrow$',
        fontsize=9,
        ha='center',
        va='center',
        color='0.4',
    )

    ax_corr = fig.add_subplot(heatmap_gs[1])
    norm_corr = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im_corr = ax_corr.imshow(
        corr_matrix, cmap='RdBu_r', norm=norm_corr, aspect='equal', origin='upper'
    )
    ax_corr.set_xticks(range(n_targets))
    ax_corr.set_xticklabels(target_labels, fontsize=7)
    ax_corr.set_yticks(range(n_targets))
    ax_corr.set_yticklabels(target_labels, fontsize=7)
    ax_corr.set_xlabel('Target $k$', fontsize=9)
    ax_corr.set_title(r'$r_{ik}$  (prior correlation)', fontsize=10, fontweight='bold')
    for i in range(n_targets):
        for j in range(n_targets):
            val = corr_matrix[i, j]
            text_col = 'white' if abs(val) > 0.55 else 'black'
            ax_corr.text(
                j, i, f'{val:+.2f}', ha='center', va='center', fontsize=6.5, color=text_col
            )
    ax_corr.axhline(query_idx - 0.5, color=target_colors[query_idx], lw=2)
    ax_corr.axhline(query_idx + 0.5, color=target_colors[query_idx], lw=2)
    plt.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])

    fig.text(
        0.67,
        0.19,
        r'$\longrightarrow$  square  $\longrightarrow$',
        fontsize=9,
        ha='center',
        va='center',
        color='0.4',
    )

    ax_r2 = fig.add_subplot(heatmap_gs[2])
    im_r2 = ax_r2.imshow(r2_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal', origin='upper')
    ax_r2.set_xticks(range(n_targets))
    ax_r2.set_xticklabels(target_labels, fontsize=7)
    ax_r2.set_yticks(range(n_targets))
    ax_r2.set_yticklabels(target_labels, fontsize=7)
    ax_r2.set_xlabel('Target $k$', fontsize=9)
    ax_r2.set_title(
        r'$\rho^2_{ik} = r_{ik}^2$  $\rightarrow$ Gibbs kernel', fontsize=10, fontweight='bold'
    )
    for i in range(n_targets):
        for j in range(n_targets):
            val = r2_matrix[i, j]
            text_col = 'white' if val > 0.45 else 'black'
            ax_r2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=text_col)
    ax_r2.axhline(query_idx - 0.5, color=target_colors[query_idx], lw=2)
    ax_r2.axhline(query_idx + 0.5, color=target_colors[query_idx], lw=2)
    plt.colorbar(im_r2, ax=ax_r2, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0])

    fig.text(
        0.06,
        0.37,
        r'D   Prior covariance $\rightarrow$ correlation $\rightarrow$ R$^2$'
        r'  $= \Delta\mathrm{Var}/\mathrm{Var}_0$  (Gibbs kernel)',
        fontsize=11,
        fontweight='bold',
        color='k',
        va='bottom',
    )
    fig.text(
        0.06,
        0.345,
        r'$\rho^2_{ij}$ = fractional variance reduction at target $i$ from observing at $j$'
        f'  (exact arctan, $\\sigma_T$={SIGMA_T})',
        fontsize=8,
        color='0.5',
        va='top',
    )

    out_path = os.path.join(out_dir, 'per_target_fields_mixture.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    out_path_svg = os.path.join(out_dir, 'per_target_fields_mixture.svg')
    plt.savefig(out_path_svg, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved to {out_path}")
    print(f"Saved to {out_path_svg}")


def model_schematic():
    import numpy as np
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.linewidth': 0.0,
            'figure.dpi': 300,
        }
    )
    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)
    FS1 = 16
    FS2 = 12
    FS3 = 8.5
    W = 16.0
    H = 33.0
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(-2.0, H)
    ax.axis('off')
    CX = W / 2
    R_MARGIN = W - 0.4
    COL_INPUT = '#4A90D9'
    COL_BOCPD = '#E8913A'
    COL_M0 = '#888888'
    COL_M1 = '#D4760A'
    COL_BLR = '#5BA05B'
    COL_DIRICHLET = '#9B59B6'
    COL_FOURIER = '#2C8C99'
    COL_OUTPUT = '#C0392B'
    COL_GIBBS = '#D4A017'
    COL_OBS = '#2E86C1'
    COL_UPDATE = '#E74C3C'
    COL_IG = '#8E44AD'
    COL_SI = '#34495E'
    COL_ACT = '#6C5B7B'
    COL_STRUCT = '#16A085'
    PAD = 0.10
    BSTYLE = f'round,pad={PAD}'

    def box(ax, x, y, w, h, color, alpha=0.15, lw=1.0, style=None, zorder=3):
        if style is None:
            style = BSTYLE
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=style,
                facecolor=color,
                edgecolor='none',
                alpha=alpha,
                linewidth=0,
                zorder=zorder,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=style,
                facecolor='none',
                edgecolor=color,
                alpha=0.7,
                linewidth=lw,
                zorder=zorder + 1,
            )
        )

    def arrow(ax, x1, y1, x2, y2, color='0.4', lw=1.5, ms=14):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=ms),
            zorder=5,
        )

    ax.text(
        CX,
        29.3,
        'Generative Model Architecture',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color='0.2',
    )

    fp_w = 5
    fp_h = 1.0
    fp_x = R_MARGIN - fp_w
    fp_y = 28.4
    box(ax, fp_x, fp_y, fp_w, fp_h, '0.5', alpha=0.06, lw=0.5)
    ax.text(
        fp_x + fp_w / 2,
        fp_y + fp_h - 0.28,
        'Free parameters (7)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color='0.4',
    )
    ax.text(
        fp_x + fp_w / 2,
        fp_y + 0.25,
        r'$h,\; v_0,\; \sigma_T,\; \alpha,\; \kappa,\; \eta,\; \sigma_{\mathrm{proc}}^2$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        fp_x + fp_w / 2,
        fp_y - 0.22,
        r'$\sigma_0,\,\nu_0$ from baseline Student\'s $t$ fit',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.5',
        fontstyle='italic',
    )

    ti_y = 28.05
    ti_w = 4.0
    box(ax, CX - ti_w / 2, ti_y, ti_w, 0.7, COL_INPUT, alpha=0.18)
    ax.text(
        CX,
        ti_y + 0.47,
        'Target',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_INPUT,
    )
    ax.text(
        CX, ti_y + 0.15, r'$\theta_t$', ha='center', va='center', fontsize=FS2 - 0.5, color='0.45'
    )

    SI_X = 0.15
    SI_R = R_MARGIN
    SI_W = SI_R - SI_X
    SI_TOP = 27.2
    SI_BOT = 12.5
    SI_H = SI_TOP - SI_BOT
    box(ax, SI_X, SI_BOT, SI_W, SI_H, COL_SI, alpha=0.04, lw=2.0, zorder=1)
    ax.text(
        CX,
        SI_TOP - 0.28,
        r'State Inference (outer hazard $h$)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_SI,
        zorder=2,
    )
    arrow(ax, CX, ti_y, CX, SI_TOP + 2 * PAD, COL_INPUT)

    MT = 26.3
    m0_h = 1.5
    m0_w = 3.5
    x_m0 = SI_X + 0.2
    y_m0 = MT - m0_h
    box(ax, x_m0, y_m0, m0_w, m0_h, COL_M0, alpha=0.15)
    ax.text(
        x_m0 + m0_w / 2,
        y_m0 + m0_h - 0.35,
        r'$M_0$: Errors irrelevant',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_M0,
    )
    ax.text(
        x_m0 + m0_w / 2,
        y_m0 + m0_h / 2 - 0.05,
        r'$q_0$',
        ha='center',
        va='center',
        fontsize=FS2,
        color=COL_M0,
    )
    ax.text(
        x_m0 + m0_w / 2,
        y_m0 + 0.3,
        r'$p(\mathrm{aim}) = \mathrm{WT}(0,\; \sigma_0,\; \nu_0)$',
        ha='center',
        va='center',
        fontsize=FS2,
        color='0.45',
    )

    x_m1 = x_m0 + m0_w + 1.0
    w_m1 = SI_R - 0.2 - x_m1
    y_m1_top = MT
    y_m1_bot = SI_BOT + 0.2
    box(ax, x_m1, y_m1_bot, w_m1, y_m1_top - y_m1_bot, COL_M1, alpha=0.06, lw=1.5)
    ax.text(
        x_m1 + w_m1 / 2,
        y_m1_top - 0.35,
        r'$M_1$: Errors relevant ($q_1$)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_M1,
    )
    INX = x_m1 + 0.35
    INW = w_m1 - 0.7
    INR = INX + INW

    trans_y = y_m0 + m0_h / 2
    arrow(ax, x_m0 + m0_w + PAD, trans_y + 0.15, x_m1 - PAD, trans_y + 0.15, COL_M1, lw=1.3, ms=12)
    ax.text(
        (x_m0 + m0_w + x_m1) / 2,
        trans_y + 0.4,
        r'$h$',
        ha='center',
        va='center',
        fontsize=FS2,
        color=COL_M1,
    )
    arrow(ax, x_m1 - PAD, trans_y - 0.15, x_m0 + m0_w + PAD, trans_y - 0.15, COL_M0, lw=1.3, ms=12)
    ax.text(
        (x_m0 + m0_w + x_m1) / 2,
        trans_y - 0.4,
        r'$h$',
        ha='center',
        va='center',
        fontsize=FS2,
        color=COL_M0,
    )

    sh_x = x_m0
    sh_w = m0_w
    sh_y = 19.8
    sh_h = 4.6
    box(ax, sh_x, sh_y, sh_w, sh_h, COL_STRUCT, alpha=0.12, lw=0.8)
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 0.35,
        'Structure hypotheses',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_STRUCT,
    )
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 0.75,
        r'$H_0$: $\mathrm{diag}(v_0,\, v_1,\, v_1)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 1.1,
        r'$H_1$: $\mathrm{diag}(v_0{+}2v_1,\, 0,\, 0)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 1.5,
        r'Prior log-odds: $\eta$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 1.9,
        r'$P(H_m \mid \delta_{1:t}) \propto$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        sh_x + sh_w / 2,
        sh_y + sh_h - 2.25,
        r'$P(H_m \mid \delta_{1:t{-}1}) \cdot p(\delta_t \mid H_m)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )

    pv_x = x_m0
    pv_w = m0_w
    pv_y = 17.2
    pv_h = 2.3
    box(ax, pv_x, pv_y, pv_w, pv_h, COL_ACT, alpha=0.10, lw=0.8)
    ax.text(
        pv_x + pv_w / 2,
        pv_y + pv_h - 0.35,
        'Process variance',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_ACT,
    )
    ax.text(
        pv_x + pv_w / 2,
        pv_y + pv_h - 0.8,
        r'$\sigma_{\mathrm{proc}}^2$ (free parameter)',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        pv_x + pv_w / 2,
        pv_y + pv_h - 1.2,
        r'$\mathrm{var}^{\mathrm{tot}}_{i|j} = \mathrm{var}_{i|j} + \sigma_{\mathrm{proc}}^2$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        pv_x + pv_w / 2,
        pv_y + pv_h - 1.6,
        r'$\Delta c_j = 2\sigma_{\mathrm{proc}}^2 / s_j^2$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.45',
    )

    bk_x = x_m0
    bk_w = m0_w
    bk_y = 13.0
    bk_h = 0.9
    box(ax, bk_x, bk_y, bk_w, bk_h, COL_DIRICHLET, alpha=0.10, lw=0.8)
    ax.text(
        bk_x + bk_w / 2,
        bk_y + bk_h / 2,
        r'New runs: $n_{k,i,j} = 0$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )

    y_bocpd = 24.1
    bocpd_h = 1.5
    box(ax, INX, y_bocpd, INW, bocpd_h, COL_BOCPD, alpha=0.12)
    ax.text(
        INX + INW / 2,
        y_bocpd + bocpd_h - 0.35,
        'BOCPD',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_BOCPD,
    )
    ax.text(
        INX + INW / 2,
        y_bocpd + 0.6,
        r'$h^{\mathrm{cp}}_t = a_t^{(h)}/(a_t^{(h)}{+}b_t^{(h)})$'
        r'$\quad$ (Beta-Bernoulli)',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        INX + INW / 2,
        y_bocpd + 0.3,
        r'$r = 0, \ldots, K_{\max}$   masses $R_{t,k}$',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.4',
    )
    ax.text(
        INX + INW / 2,
        y_bocpd + 0.08,
        r'Moment-matched merging of oldest runs',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.4',
    )

    y_blr = 17.5
    blr_h = 5.0
    box(ax, INX, y_blr, INW, blr_h, COL_BLR, alpha=0.1)
    ax.text(
        INX + INW / 2,
        y_blr + blr_h - 0.35,
        'Per-run rank-1 BLR bank',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_BLR,
    )
    arrow(ax, INX + INW / 2, y_bocpd - 2 * PAD, INX + INW / 2, y_blr + blr_h + 2 * PAD, COL_BOCPD)

    BIN = 0.3
    prior_x = INX + BIN
    prior_y = y_blr + BIN
    prior_w = 3.5
    prior_h = 3.8
    box(ax, prior_x, prior_y, prior_w, prior_h, COL_FOURIER, alpha=0.12, lw=0.8)
    ax.text(
        prior_x + prior_w / 2,
        prior_y + prior_h - 0.35,
        'Prior (diagonal)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_FOURIER,
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + prior_h - 0.7,
        r'$\boldsymbol{\mu}_0 = \mathbf{0}$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + prior_h - 1.05,
        r'$\boldsymbol{\Sigma}_0 = \mathrm{diag}(v_0,\, v_1,\, v_1)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color=COL_FOURIER,
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + prior_h - 1.45,
        r'$v_1 = \frac{(180/\pi)^2}{2}\, \mathbb{E}[\psi_1(\varepsilon)^2]$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.45',
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + prior_h - 1.8,
        r'$\varepsilon \sim \mathrm{Rayleigh}(\sigma_T)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.5',
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + 0.75,
        r'$s_j = \mathbf{f}_j^\top \mathbf{u}_j$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color=COL_FOURIER,
    )
    ax.text(
        prior_x + prior_w / 2,
        prior_y + 0.25,
        r'$\mathcal{K}(\theta_i, \theta_j) = \mathbf{f}_i^\top \boldsymbol{\Sigma}_0 \mathbf{f}_j$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.5',
    )

    blr_bw = 1.35
    blr_x0 = prior_x + prior_w + 0.25
    blr_gap = 0.20
    blr_boxes_x = [
        blr_x0,
        blr_x0 + blr_bw + blr_gap,
        blr_x0 + 2 * (blr_bw + blr_gap),
        blr_x0 + 2 * (blr_bw + blr_gap) + 0.65,
    ]
    blr_labels = [r'$\theta_1$', r'$\theta_2$', None, r'$\theta_K$']
    for bx, lbl in zip(blr_boxes_x, blr_labels):
        if lbl is None:
            ax.text(
                bx + 0.1,
                prior_y + prior_h / 2,
                r'$\cdots$',
                ha='center',
                va='center',
                fontsize=FS1,
                color='0.5',
            )
        else:
            box(ax, bx, prior_y, blr_bw, prior_h, COL_BLR, alpha=0.2, lw=0.8)
            ax.text(
                bx + blr_bw / 2,
                prior_y + prior_h - 0.35,
                lbl,
                ha='center',
                va='center',
                fontsize=FS1,
                color=COL_BLR,
                fontweight='bold',
            )
            ax.text(
                bx + blr_bw / 2,
                prior_y + prior_h - 0.7,
                r'$c_{r,j},\; d_{r,j}$',
                ha='center',
                va='center',
                fontsize=FS2 - 0.5,
                color='0.4',
            )
            ax.text(
                bx + blr_bw / 2,
                prior_y + prior_h - 1.1,
                r'$\boldsymbol{\mu} = d \, \mathbf{u}_j$',
                ha='center',
                va='center',
                fontsize=FS2 - 0.5,
                color='0.4',
            )
            ax.text(
                bx + blr_bw / 2,
                prior_y + prior_h - 1.5,
                r'$\boldsymbol{\Sigma} {=} \boldsymbol{\Sigma}_0 {-} c \, \mathbf{u}_j \mathbf{u}_j^\top$',
                ha='center',
                va='center',
                fontsize=FS2 - 0.5,
                color='0.4',
            )
            ax.text(
                bx + blr_bw / 2,
                prior_y + 0.25,
                r'$\mathbf{u}_j = \boldsymbol{\Sigma}_0 \mathbf{f}_j$',
                ha='center',
                va='center',
                fontsize=FS3 - 0.5,
                color='0.55',
            )

    dir_h = 3.8
    y_dir = y_blr - 0.5 - dir_h
    box(ax, INX, y_dir, INW, dir_h, COL_DIRICHLET, alpha=0.1)
    ax.text(
        INX + INW / 2,
        y_dir + dir_h - 0.35,
        'Dirichlet-process source selection',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_DIRICHLET,
    )
    arrow(ax, INX + INW / 2, y_blr - 2 * PAD, INX + INW / 2, y_dir + dir_h + 2 * PAD, COL_BLR)
    ax.text(
        INX + INW / 2,
        y_dir + dir_h - 0.7,
        r'$w_{ij,k} = \frac{\alpha \, \pi_{ij,k} \;+\; n_{k,i,j}}{\alpha \;+\; n_{k,i}}$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )

    dir_bin = 0.3
    ipx = INX + dir_bin
    ipy = y_dir + dir_bin
    ipw = 4.5
    iph = 2.3
    box(ax, ipx, ipy, ipw, iph, COL_GIBBS, alpha=0.12, lw=0.8)
    ax.text(
        ipx + ipw / 2,
        ipy + iph - 0.35,
        'Gibbs-weighted base',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_GIBBS,
    )
    ax.text(
        ipx + ipw / 2,
        ipy + iph - 0.7,
        r'$\rho_{ij}^2 = \mathcal{K}(\theta_i,\theta_j)^2 / (s_i \cdot s_j)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        ipx + ipw / 2,
        ipy + iph - 1.0,
        r'$\mathcal{G}(i,j) = e^{\kappa \, \rho_{ij}^2}$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        ipx + ipw / 2,
        ipy + iph - 1.3,
        r'$\pi_{ij,k} \propto (\epsilon_w {+} \mathrm{totalObs}_{k,j})\cdot \mathcal{G}(i,j)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        ipx + ipw / 2,
        ipy + 0.15,
        r'$\rho_{ij}^2 = R^2$: anticorrelated $\Rightarrow$ bimodal $\pm\delta$',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.5',
    )

    plus_x = ipx + ipw + 0.3
    ax.text(
        plus_x,
        ipy + iph / 2,
        '+',
        ha='center',
        va='center',
        fontsize=FS1,
        color='0.5',
        fontweight='bold',
    )
    epx = plus_x + 0.3
    epy = ipy
    epw = INR - dir_bin - epx
    eph = iph
    box(ax, epx, epy, epw, eph, COL_DIRICHLET, alpha=0.15, lw=0.8)
    ax.text(
        epx + epw / 2,
        epy + eph - 0.35,
        'Experience counts',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_DIRICHLET,
    )
    ax.text(
        epx + epw / 2,
        epy + eph - 0.7,
        r'$n_{k,i,j}$: accumulated',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        epx + epw / 2,
        epy + eph - 1.0,
        r'responsibilities $\omega_j$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        epx + epw / 2,
        epy + eph - 1.3,
        r'Per run $k$, target pair $(i,j)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )

    four_x = 0.3
    four_y = 8.5
    four_w = 3.5
    four_h = 1.8
    box(ax, four_x, four_y, four_w, four_h, COL_FOURIER, alpha=0.12)
    ax.text(
        four_x + four_w / 2,
        four_y + four_h - 0.35,
        'Fourier feature vector',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_FOURIER,
    )
    ax.text(
        four_x + four_w / 2,
        four_y + four_h - 0.85,
        r'$\mathbf{f}(\theta) = [1,\, \sin\theta,\, \cos\theta]^\top$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        four_x + four_w / 2,
        four_y + 0.3,
        r'$D = 3$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.5',
    )

    BOT_LEFT = 4.2
    BOT_W = R_MARGIN - BOT_LEFT
    BOT_CX = BOT_LEFT + BOT_W / 2
    act_x = BOT_LEFT - 0.15
    act_w = BOT_W + 0.3
    act_y = 5.6
    act_h = 6.5
    box(ax, act_x, act_y, act_w, act_h, COL_ACT, alpha=0.06, lw=1.5, zorder=1)
    ax.text(
        act_x + act_w / 2,
        act_y + act_h - 0.35,
        'Act & observe',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_ACT,
        zorder=2,
    )

    y_pred = 8.7
    pred_h = 2.7
    box(ax, BOT_LEFT, y_pred, BOT_W, pred_h, COL_OUTPUT, alpha=0.1)
    ax.text(
        BOT_CX,
        y_pred + pred_h - 0.35,
        r'Sample aim $z_t$ from predictive',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_OUTPUT,
    )
    ax.text(
        BOT_CX,
        y_pred + pred_h - 0.75,
        r'$z_t \sim \sum_m P(H_m)\!\left[q_m^{(0)} \mathrm{WT}(0,\, \sigma_0,\, \nu_0)\right.$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
    )
    ax.text(
        BOT_CX,
        y_pred + pred_h - 1.35,
        r'$\left.+\; \sum_k R_{m,k} \sum_j w_{ij}\; \mathrm{WT}\!\left(-\hat{\delta}_{i|j},\;'
        r'\sqrt{\mathrm{var}_{i|j,k} + \sigma_{\mathrm{proc}}^2 + \sigma_0^2},\, \nu_m\right)\right]$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
    )

    y_obs = 5.8
    obs_h = 2.6
    box(ax, BOT_LEFT, y_obs, BOT_W, obs_h, COL_OBS, alpha=0.1)
    ax.text(
        BOT_CX,
        y_obs + obs_h - 0.35,
        r'Evaluate observed perturbation ($\delta_t$)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_OBS,
    )
    ax.text(
        BOT_CX,
        y_obs + obs_h - 0.8,
        r'$p(\delta_t \mid j, k) = \mathrm{WT}\!\left(\delta_t;\; \hat{\delta}_{i|j,k},\;'
        r'\sqrt{\mathrm{var}_{i|j,k} + \sigma_{\mathrm{proc}}^2 + \sigma_{\mathrm{obs},t}^2}'
        r',\; \nu_{M_1}\right)$',
        ha='center',
        va='center',
        fontsize=FS2 - 1.0,
        color=COL_OBS,
    )
    ax.text(
        BOT_CX,
        y_obs + obs_h - 1.3,
        r'$\sigma_{\mathrm{obs},t}^2 = c_{\mathrm{vis}}^2 \cdot |\delta_t|^2$'
        r'$\qquad c_{\mathrm{vis}} = 0.3$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.4',
    )
    ax.text(
        BOT_CX,
        y_obs + 0.35,
        r'WT $=$ wrapped Student\'s $t$ (period $360°$)',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.45',
        fontstyle='italic',
    )

    arrow(
        ax,
        four_x + four_w + PAD,
        four_y + four_h - 0.3,
        BOT_LEFT - PAD,
        y_pred + pred_h / 2,
        COL_FOURIER,
        lw=1.2,
        ms=12,
    )
    arrow(
        ax,
        x_m0 + m0_w / 2,
        SI_BOT - 2 * PAD,
        BOT_LEFT + 1.0,
        y_pred + pred_h + 2 * PAD,
        COL_M0,
        lw=1.2,
        ms=12,
    )
    arrow(
        ax,
        x_m1 + w_m1 / 2,
        SI_BOT - 2 * PAD,
        BOT_CX + 2.0,
        y_pred + pred_h + 2 * PAD,
        COL_M1,
        lw=1.2,
        ms=12,
    )
    arrow(ax, BOT_CX, y_pred - 2 * PAD, BOT_CX, y_obs + obs_h + 2 * PAD, COL_OUTPUT, lw=1.2)

    y_upd = -1.2
    upd_h = 6.0
    UPD_LEFT = 0.5
    UPD_W = R_MARGIN - UPD_LEFT
    box(ax, UPD_LEFT, y_upd, UPD_W, upd_h, COL_UPDATE, alpha=0.08)
    ax.text(
        CX,
        y_upd + upd_h - 0.35,
        r'State updates (after feedback $\delta_t$)',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_UPDATE,
    )

    sub_y = y_upd + 2.7
    sub_h = 2.2
    sub_gap = 0.3
    UPD_INNER_L = UPD_LEFT + 0.2
    UPD_INNER_R = UPD_LEFT + UPD_W - 0.2
    avail = UPD_INNER_R - UPD_INNER_L - 3 * sub_gap
    b1_w = avail * 0.24
    b2_w = avail * 0.22
    b3_w = avail * 0.30
    b4_w = avail * 0.24
    b1_x = UPD_INNER_L
    b2_x = b1_x + b1_w + sub_gap
    b3_x = b2_x + b2_w + sub_gap
    b4_x = b3_x + b3_w + sub_gap

    box(ax, b1_x, sub_y, b1_w, sub_h, COL_UPDATE, alpha=0.12, lw=0.8)
    ax.text(
        b1_x + b1_w / 2,
        sub_y + sub_h - 0.3,
        'BOCPD masses',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_UPDATE,
    )
    ax.text(
        b1_x + b1_w / 2,
        sub_y + sub_h / 2 + 0.1,
        r'$R_{t,k} \propto R_{t{-}1,k{-}1}(1{-}h^{\mathrm{cp}})\, p_{\mathrm{self},k}$'
        '\n'
        r'$R_{t,0} \propto \mathrm{Mass}_{\mathrm{cp}}\, p_{\mathrm{self},0}$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
        linespacing=1.5,
    )
    ax.text(
        b1_x + b1_w / 2,
        sub_y + 0.25,
        r'$q_0 + \sum_k R_k = 1$',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.5',
    )

    box(ax, b2_x, sub_y, b2_w, sub_h, COL_UPDATE, alpha=0.12, lw=0.8)
    ax.text(
        b2_x + b2_w / 2,
        sub_y + sub_h - 0.3,
        'Inner hazard',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_UPDATE,
    )
    ax.text(
        b2_x + b2_w / 2,
        sub_y + sub_h / 2 + 0.1,
        r'$a^{(h)} \!\leftarrow\! a^{(h)} {+} q_1 p_{\mathrm{cp},t}$'
        '\n'
        r'$b^{(h)} \!\leftarrow\! b^{(h)} {+} q_1(1{-}p_{\mathrm{cp},t})$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
        linespacing=1.5,
    )
    ax.text(
        b2_x + b2_w / 2,
        sub_y + 0.25,
        r'$p_{\mathrm{cp},t} = R_{t,0} / \sum_k R_{t,k}$',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.5',
    )

    box(ax, b3_x, sub_y, b3_w, sub_h, COL_UPDATE, alpha=0.12, lw=0.8)
    ax.text(
        b3_x + b3_w / 2,
        sub_y + sub_h - 0.3,
        'BLR',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_UPDATE,
    )
    ax.text(
        b3_x + b3_w / 2,
        sub_y + sub_h / 2 + 0.1,
        r'$q = 1 - c \cdot \lambda_k s_j$'
        '\n'
        r'$c \leftarrow c + q^2\!/(q \lambda_k s_j + \sigma_{\mathrm{obs}}^2)$'
        '\n'
        r'$d \leftarrow d + q(\delta_t {-} d \lambda_k s_j)/(q \lambda_k s_j {+} \sigma_{\mathrm{obs}}^2)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
        linespacing=1.5,
    )

    box(ax, b4_x, sub_y, b4_w, sub_h, COL_UPDATE, alpha=0.12, lw=0.8)
    ax.text(
        b4_x + b4_w / 2,
        sub_y + sub_h - 0.3,
        'Dirichlet counts',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_UPDATE,
    )
    ax.text(
        b4_x + b4_w / 2,
        sub_y + sub_h / 2 + 0.1,
        r'$\omega_j \propto w_{ij,k} \, p(\delta_t \mid j, k)$'
        '\n'
        r'$n_{k,i,j} \leftarrow n_{k,i,j} + \omega_j$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
        linespacing=1.5,
    )

    row2_y = y_upd + 0.15
    row2_h = 2.3
    row2_gap = 0.2
    str_w = (UPD_INNER_R - UPD_INNER_L - row2_gap) * 0.55
    ig_w = (UPD_INNER_R - UPD_INNER_L - row2_gap) * 0.45
    str_x = UPD_INNER_L
    ig_x = str_x + str_w + row2_gap

    box(ax, str_x, row2_y, str_w, row2_h, COL_STRUCT, alpha=0.12, lw=0.8)
    ax.text(
        str_x + str_w / 2,
        row2_y + row2_h - 0.3,
        'Structure posterior',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_STRUCT,
    )
    ax.text(
        str_x + str_w / 2,
        row2_y + row2_h - 0.75,
        r'$P_t(H_m) \propto P_{t{-}1}(H_m) \cdot p(\delta_t \mid H_m)$',
        ha='center',
        va='center',
        fontsize=FS2 - 0.5,
        color='0.35',
    )
    ax.text(
        str_x + str_w / 2,
        row2_y + 0.35,
        r'Uses full Dirichlet-mixture marginals',
        ha='center',
        va='center',
        fontsize=FS3,
        color='0.5',
    )

    box(ax, ig_x, row2_y, ig_w, row2_h, COL_IG, alpha=0.12, lw=0.8)
    ax.text(
        ig_x + ig_w / 2,
        row2_y + row2_h - 0.3,
        r'Perturbation-change scale $\lambda$',
        ha='center',
        va='center',
        fontsize=FS1,
        fontweight='bold',
        color=COL_IG,
    )
    ax.text(
        ig_x + ig_w / 2,
        row2_y + row2_h - 0.65,
        r'$\lambda \sim \mathrm{IG}(\alpha_\lambda{=}2,\, \beta_\lambda{=}1)$',
        ha='center',
        va='center',
        fontsize=FS2 - 1,
        color='0.35',
    )
    ax.text(
        ig_x + ig_w / 2,
        row2_y + row2_h - 1.0,
        r'$\alpha_\lambda {+}= p_{\mathrm{cp},t} \cdot 0.5$',
        ha='center',
        va='center',
        fontsize=FS2 - 1,
        color='0.35',
    )
    ax.text(
        ig_x + ig_w / 2,
        row2_y + row2_h - 1.35,
        r'$\beta_\lambda {+}= p_{\mathrm{cp},t} \cdot \delta_t^2/(2s_j)$',
        ha='center',
        va='center',
        fontsize=FS2 - 1,
        color='0.35',
    )
    ax.text(
        ig_x + ig_w / 2,
        row2_y + 0.35,
        r'$\lambda = \beta_\lambda/(\alpha_\lambda{-}1)$',
        ha='center',
        va='center',
        fontsize=FS3 - 0.5,
        color='0.5',
    )

    arrow(ax, BOT_CX, y_obs - 2 * PAD, BOT_CX, y_upd + upd_h + 2 * PAD, COL_OBS, lw=1.2)

    from matplotlib.patches import FancyArrowPatch

    loop_x = R_MARGIN + 0.3
    ax.annotate(
        '',
        xy=(loop_x, SI_TOP + 0.1),
        xytext=(loop_x, y_upd + upd_h - 0.1),
        arrowprops=dict(
            arrowstyle='->', color='0.55', lw=1.2, mutation_scale=12, connectionstyle='arc3,rad=0.0'
        ),
        zorder=4,
    )
    ax.text(
        loop_x + 0.15,
        (y_upd + upd_h + SI_TOP) / 2,
        r'$t{+}1$',
        ha='left',
        va='center',
        fontsize=FS3,
        color='0.5',
        rotation=90,
    )

    out_path = os.path.join(out_dir, 'model_schematic_v6.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved to {out_path}")


def prior_kernel():

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams
    import math

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['font.size'] = 9
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.4
    rcParams['ytick.minor.width'] = 0.4

    # Use smaller N_HARM for speed (N_HARM=64 sufficient for visualization)
    N_HARM = 64
    DIM = 2 * N_HARM + 1

    def buildPriorDiag(logVarDC, logVarTrans, gamma=1.0):
        """
        Build prior diagonal variance spectrum using translation-derived prior.
        Exact implementation from model.
        """
        dc_var = np.exp(np.clip(logVarDC, -700, 700))
        s2 = np.exp(np.clip(logVarTrans, -700, 700))
        sigma_T = math.sqrt(s2)
        RAD2DEG_SQ = (180.0 / math.pi) ** 2
        diag = np.zeros(DIM)
        diag[0] = dc_var
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
                    hn_sq = (2.0 - eps ** (-n)) ** 2
                integral += hn_sq * p_eps * d_eps
            v_n = RAD2DEG_SQ * integral / (2.0 * n ** (2.0 * gamma))
            diag[2 * n - 1] = v_n
            diag[2 * n] = v_n
        return diag

    def featureVector(theta_rad):
        """Compute Fourier feature vector at angle theta (in radians)."""
        f = np.zeros(DIM)
        f[0] = 1.0
        for n in range(1, N_HARM + 1):
            f[2 * n - 1] = np.sin(n * theta_rad)
            f[2 * n] = np.cos(n * theta_rad)
        return f

    def spatialKernel(delta_theta_rad, diag_prior):
        """
        Compute spatial kernel K(Δθ) = v₀ + Σ_n v_n * cos(n*Δθ).
        """
        K = diag_prior[0]  # DC component
        for n in range(1, N_HARM + 1):
            v_n = diag_prior[2 * n - 1]  # sin and cos have same variance
            K += v_n * np.cos(n * delta_theta_rad)
        return K

    fig = plt.figure(figsize=(10, 10), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_a1 = fig.add_subplot(gs[0, 0])
    sigma_T_values = [0.1, 0.3, 0.5, 1.0]
    colors_st = plt.cm.Blues(np.linspace(0.4, 0.9, len(sigma_T_values)))

    for sigma_T, color in zip(sigma_T_values, colors_st):
        logVarTrans = 2 * np.log(sigma_T)
        diag = buildPriorDiag(logVarDC=-700, logVarTrans=logVarTrans, gamma=1.0)
        n_range = np.arange(1, 21)
        v_n = [diag[2 * n - 1] for n in n_range]
        ax_a1.loglog(
            n_range, v_n, 'o-', markersize=4, linewidth=1.5, color=color, label=f'σ_T = {sigma_T}'
        )

    ax_a1.set_xlabel('Harmonic number n', fontsize=9)
    ax_a1.set_ylabel('Prior variance v_n (deg²)', fontsize=9)
    ax_a1.set_title('(A1) Effect of σ_T (γ=1.0)', fontsize=10, fontweight='bold')
    ax_a1.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax_a1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    ax_a2 = fig.add_subplot(gs[0, 1])
    gamma_values = [0.5, 1.0, 2.0]
    colors_g = plt.cm.Oranges(np.linspace(0.4, 0.9, len(gamma_values)))

    for gamma, color in zip(gamma_values, colors_g):
        logVarTrans = 2 * np.log(0.3)
        diag = buildPriorDiag(logVarDC=-700, logVarTrans=logVarTrans, gamma=gamma)
        n_range = np.arange(1, 21)
        v_n = [diag[2 * n - 1] for n in n_range]
        ax_a2.loglog(
            n_range, v_n, 'o-', markersize=4, linewidth=1.5, color=color, label=f'γ = {gamma}'
        )

    ax_a2.set_xlabel('Harmonic number n', fontsize=9)
    ax_a2.set_ylabel('Prior variance v_n (deg²)', fontsize=9)
    ax_a2.set_title('(A2) Effect of γ (σ_T=0.3)', fontsize=10, fontweight='bold')
    ax_a2.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax_a2.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    ax_b = fig.add_subplot(gs[1, 0])
    gamma_values_b = [0.5, 1.0, 2.0, 5.0]
    colors_b = plt.cm.Reds(np.linspace(0.4, 0.9, len(gamma_values_b)))

    for gamma, color in zip(gamma_values_b, colors_b):
        logVarTrans = 2 * np.log(0.3)
        diag = buildPriorDiag(logVarDC=-700, logVarTrans=logVarTrans, gamma=gamma)
        n_range = np.arange(1, 21)
        v_n = [diag[2 * n - 1] for n in n_range]
        ax_b.loglog(
            n_range,
            v_n,
            'o-',
            markersize=4,
            linewidth=1.5,
            color=color,
            label=f'γ = {gamma}',
            zorder=3,
        )

        ref_line = v_n[0] * (n_range / n_range[0]) ** (-2 * gamma)
        ax_b.loglog(n_range, ref_line, '--', color=color, alpha=0.5, linewidth=1.0, zorder=2)

    ax_b.set_xlabel('Harmonic number n', fontsize=9)
    ax_b.set_ylabel('Prior variance v_n (deg²)', fontsize=9)
    ax_b.set_title('(B) Power-law decay n^{-2γ} (σ_T=0.3)', fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax_b.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    delta_theta_deg = np.linspace(-180, 180, 200)
    delta_theta_rad = np.radians(delta_theta_deg)

    ax_c = fig.add_subplot(gs[1, 1])
    gamma_values_c = [0.5, 1.0, 2.0, 5.0]
    colors_c = plt.cm.Purples(np.linspace(0.4, 0.9, len(gamma_values_c)))

    for gamma, color in zip(gamma_values_c, colors_c):
        logVarTrans = 2 * np.log(0.3)
        diag = buildPriorDiag(logVarDC=-700, logVarTrans=logVarTrans, gamma=gamma)
        K_vals = np.array([spatialKernel(dt_rad, diag) for dt_rad in delta_theta_rad])
        ax_c.plot(delta_theta_deg, K_vals, linewidth=1.5, color=color, label=f'γ = {gamma}')

    target_angles = np.arange(0, 360, 45)
    for angle in target_angles:
        ax_c.axvline(
            angle - 180 if angle > 180 else angle,
            color='gray',
            linestyle=':',
            alpha=0.3,
            linewidth=0.8,
        )

    ax_c.set_xlabel('Angular separation Δθ (degrees)', fontsize=9)
    ax_c.set_ylabel('Kernel K(Δθ) (deg²)', fontsize=9)
    ax_c.set_title('(C) Spatial kernel K(Δθ)', fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=8, loc='upper right', framealpha=0.95)
    ax_c.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_c.set_xlim(-180, 180)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d_twin = ax_c.twinx()  # Reuse axis C for D, showing R² on secondary y-axis

    for gamma, color in zip(gamma_values_c, colors_c):
        logVarTrans = 2 * np.log(0.3)
        diag = buildPriorDiag(logVarDC=-700, logVarTrans=logVarTrans, gamma=gamma)
        K_vals = np.array([spatialKernel(dt_rad, diag) for dt_rad in delta_theta_rad])
        K_0 = spatialKernel(0.0, diag)
        R2_vals = (K_vals / K_0) ** 2
        ax_d_twin.plot(
            delta_theta_deg,
            R2_vals,
            linewidth=1.5,
            color=color,
            alpha=0.4,
            linestyle='--',
            label=f'R² γ={gamma}',
        )

    ax_d_twin.axvline(
        180, color='red', linestyle='-.', alpha=0.5, linewidth=1.2, label='Anticorr. (180°)'
    )
    ax_d_twin.axvline(-180, color='red', linestyle='-.', alpha=0.5, linewidth=1.2)

    ax_d_twin.set_ylabel('Squared correlation R²', fontsize=9, color='red', alpha=0.7)
    ax_d_twin.tick_params(axis='y', labelcolor='red', labelsize=8)
    ax_d_twin.set_ylim(-0.1, 1.1)

    ax_c.text(
        0.02,
        0.02,
        'Dashed lines: R²\nSolid lines: K(Δθ)',
        transform=ax_c.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )

    fig.suptitle(
        'Translation-Derived Prior Spectrum & Spatial Kernel',
        fontsize=11,
        fontweight='bold',
        y=0.98,
    )

    import os

    os.makedirs('modelIntuitionPumpsImages', exist_ok=True)

    output_path = 'modelIntuitionPumpsImages/prior_spectrum_kernel.svg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {output_path}")

    plt.close()


def sequential_update():

    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import os

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 9,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.dpi': 300,
        }
    )

    out_dir = os.path.join(os.getcwd(), 'modelIntuitionPumpsImages')
    os.makedirs(out_dir, exist_ok=True)

    N_HARM = 64
    DIM = 2 * N_HARM + 1

    GAMMA = 1.0
    SIGMA_T = 0.3
    LOG_VAR_DC = np.log(1e-6)  # ~0, translation-only for clarity
    LOG_VAR_TRANS = np.log(SIGMA_T**2)
    OBS_VAR = 25.0

    def featureVector(theta_rad):
        f = np.zeros(DIM)
        f[0] = 1.0
        for n in range(1, N_HARM + 1):
            f[2 * n - 1] = np.sin(n * theta_rad)
            f[2 * n] = np.cos(n * theta_rad)
        return f

    def buildPriorDiag(logVarDC, logVarTrans, gamma=GAMMA):
        s2 = np.exp(np.clip(logVarTrans, -700, 700))
        sigma_T = math.sqrt(s2)
        RAD2DEG_SQ = (180.0 / math.pi) ** 2
        diag = np.zeros(DIM)
        diag[0] = np.exp(np.clip(logVarDC, -700, 700))
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
                    hn_sq = (2.0 - eps ** (-n)) ** 2
                integral += hn_sq * p_eps * d_eps
            v_n = RAD2DEG_SQ * integral / (2.0 * n ** (2.0 * gamma))
            diag[2 * n - 1] = v_n
            diag[2 * n] = v_n
        return diag

    prior_diag = buildPriorDiag(LOG_VAR_DC, LOG_VAR_TRANS, GAMMA)

    n_targets = 8
    target_angles_deg = np.arange(0, 360, 45)
    target_angles_rad = np.deg2rad(target_angles_deg)

    TRUE_DELTA = 15.0

    theta_fine_deg = np.linspace(0, 360, 361)
    theta_fine_rad = np.deg2rad(theta_fine_deg)

    F_fine = np.array([featureVector(th) for th in theta_fine_rad])  # (361, DIM)

    s_fine = np.sum(F_fine**2 * prior_diag[None, :], axis=1)

    def blr_single_obs(obs_target_rad, delta, prior_diag, obs_var):
        """One observation at a single target. Returns (c, d, v_j, f_j, s_j)."""
        f_j = featureVector(obs_target_rad)
        v_j = prior_diag * f_j  # Sigma_0 @ f_j (diagonal prior)
        s_j = np.dot(f_j, v_j)  # prior predictive variance
        c = 1.0 / (s_j + obs_var)
        d = delta / (s_j + obs_var)
        return c, d, v_j, f_j, s_j

    def predict_field(c, d, v_j, F_grid, s_grid, prior_diag):
        """Posterior mean and std at all grid points from target j's BLR."""
        crossDots = F_grid @ v_j  # (N_grid,)
        means = d * crossDots
        vars_ = s_grid - c * crossDots**2
        vars_ = np.maximum(vars_, 1e-10)
        stds = np.sqrt(vars_)
        return means, stds, crossDots

    obs_targets = [0, 2, 4]  # indices: 0°, 90°, 180°
    obs_labels = ['0°', '90°', '180°']
    obs_colors = ['#D64541', '#2E86C1', '#27AE60']

    fields = {}
    for idx in obs_targets:
        c, d, v_j, f_j, s_j = blr_single_obs(
            target_angles_rad[idx], TRUE_DELTA, prior_diag, OBS_VAR
        )
        means, stds, crossDots = predict_field(c, d, v_j, F_fine, s_fine, prior_diag)
        fields[idx] = {
            'c': c,
            'd': d,
            'v_j': v_j,
            'means': means,
            'stds': stds,
            'crossDots': crossDots,
            's_j': s_j,
        }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    f = fields[0]
    ax.fill_between(
        theta_fine_deg,
        f['means'] - f['stds'],
        f['means'] + f['stds'],
        alpha=0.2,
        color=obs_colors[0],
    )
    ax.plot(
        theta_fine_deg,
        f['means'],
        color=obs_colors[0],
        lw=2.0,
        label=r'Posterior mean ($d_0 \cdot \mathrm{crossDot}$)',
    )

    # Mean prediction is d * crossDot; match peaks to compare its shape with the kernel.
    kernel = f['crossDots']
    kernel_scaled = kernel / f['s_j'] * f['means'].max()
    ax.plot(
        theta_fine_deg,
        kernel_scaled,
        '--',
        color='0.4',
        lw=1.2,
        label=r'$\mathcal{K}(\theta, 0°) / s_0$ (scaled)',
    )

    ax.axhline(TRUE_DELTA, color='0.6', ls=':', lw=0.8, label='True δ = +15°')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.scatter(target_angles_deg, np.zeros(n_targets), c='k', s=25, marker='x', zorder=5)
    ax.scatter(
        [0], [TRUE_DELTA], c=obs_colors[0], s=80, marker='*', zorder=5, label='Observation at 0°'
    )

    ax.set_xlabel('Query angle θ (°)')
    ax.set_ylabel('Predicted perturbation (°)')
    ax.set_xlim(-5, 365)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(-22, 22)
    ax.set_title('A.  Single target BLR: prediction field', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, ls=':', lw=0.4, alpha=0.4)

    ax = axes[1]
    for idx, lbl, col in zip(obs_targets, obs_labels, obs_colors):
        f = fields[idx]
        ax.fill_between(
            theta_fine_deg, f['means'] - f['stds'], f['means'] + f['stds'], alpha=0.12, color=col
        )
        ax.plot(theta_fine_deg, f['means'], color=col, lw=1.8, label=f'BLR from obs at {lbl}')
        ax.scatter([target_angles_deg[idx]], [TRUE_DELTA], c=col, s=80, marker='*', zorder=5)

    ax.axhline(TRUE_DELTA, color='0.6', ls=':', lw=0.8, label='True δ')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.scatter(target_angles_deg, np.zeros(n_targets), c='k', s=25, marker='x', zorder=5)

    ax.set_xlabel('Query angle θ (°)')
    ax.set_xlim(-5, 365)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(-22, 22)
    ax.set_title('B.  Independent per-target fields', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, ls=':', lw=0.4, alpha=0.4)

    ax.text(
        0.5,
        0.02,
        'Each target maintains independent $(c_j, d_j)$;\n'
        'Dirichlet source selection mixes these fields',
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=7,
        fontstyle='italic',
        color='0.4',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.8'),
    )

    ax = axes[2]
    f0 = fields[0]

    prior_std = np.sqrt(s_fine)
    post_var_0 = s_fine - f0['c'] * f0['crossDots'] ** 2
    post_var_0 = np.maximum(post_var_0, 1e-10)
    post_std_0 = np.sqrt(post_var_0)

    frac_reduction = 1.0 - post_var_0 / s_fine

    ax.plot(theta_fine_deg, prior_std, 'k-', lw=1.5, label='Prior SD')
    ax.plot(
        theta_fine_deg, post_std_0, color=obs_colors[0], lw=1.5, label='Posterior SD (obs at 0°)'
    )

    ax2 = ax.twinx()
    ax2.fill_between(theta_fine_deg, frac_reduction * 100, alpha=0.15, color='#F39C12')
    ax2.plot(
        theta_fine_deg,
        frac_reduction * 100,
        color='#F39C12',
        lw=1.2,
        ls='--',
        label='Variance reduction %',
    )
    ax2.set_ylabel('Variance reduction (%)', fontsize=9, color='#D4760A')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', colors='#D4760A')

    ax.scatter([0], [post_std_0[0]], c=obs_colors[0], s=80, marker='*', zorder=5)
    ax.scatter(target_angles_deg, np.zeros(n_targets), c='k', s=25, marker='x', zorder=5)

    ax.set_xlabel('Query angle θ (°)')
    ax.set_ylabel('Predictive SD (°)')
    ax.set_xlim(-5, 365)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title('C.  Rank-1 variance reduction', fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
    ax.grid(True, ls=':', lw=0.4, alpha=0.4)

    fig.suptitle(
        'Per-target rank-1 BLR: each observation updates only its own $(c_j, d_j)$',
        fontsize=12,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'sequential_blr_update.svg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved to {out_path}")


def run():
    print("Figure 3.1: translation_prediction")
    translation_prediction()
    plt.close("all")
    print("Figure 3.1: fourier_fields")
    fourier_fields()
    plt.close("all")
    print("Figure 3.1: gamma_sweep")
    gamma_sweep()
    plt.close("all")
    print("Figure 3.1: local_global_fields")
    local_global_fields()
    plt.close("all")
    print("Figure 3.1: model_schematic")
    model_schematic()
    plt.close("all")
    print("Figure 3.1: prior_kernel")
    prior_kernel()
    plt.close("all")
    print("Figure 3.1: sequential_update")
    sequential_update()
    plt.close("all")
