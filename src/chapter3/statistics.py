"""Chapter 3 inferential analyses and wildcard quartile summaries."""

import matplotlib.pyplot as plt


def core():
    import json
    import math
    import os
    import itertools
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.special import logsumexp

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.formula.api import ols
    from statsmodels.stats.multitest import multipletests

    import pingouin as pg

    import BayesHypothesisTesting as BHT
    import dingBLR
    import HMM

    warnings.filterwarnings("ignore")

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (8, 4.5)

    plt.close("all")

    def _load_fit_npy(path, required=True):
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        if required:
            raise FileNotFoundError(path)
        return None

    def _wide_dataset_summary(name, df, participant_col_candidates):
        participant_col = next((c for c in participant_col_candidates if c in df.columns), None)
        return {
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "participant_col": participant_col,
            "n_participants": int(df[participant_col].nunique()) if participant_col else np.nan,
        }

    def _fit_container_summary(name, obj):
        if isinstance(obj, dict):
            sample = next(iter(obj.values()))
            n_units = sum(len(getattr(fit, "participantNums", [])) for fit in obj.values())
            return {
                "fit_name": name,
                "container_type": type(obj).__name__,
                "n_entries": len(obj),
                "n_participant_units": int(n_units),
                "sample_entry_type": type(sample).__name__,
            }

        if isinstance(obj, np.ndarray):
            sample = obj[0] if obj.size else None
            n_units = 0
            for entry in obj:
                n_units += len(getattr(entry, "participantNums", []))
            return {
                "fit_name": name,
                "container_type": type(obj).__name__,
                "n_entries": int(len(obj)),
                "n_participant_units": int(n_units),
                "sample_entry_type": type(sample).__name__ if sample is not None else None,
            }

        return {
            "fit_name": name,
            "container_type": type(obj).__name__,
            "n_entries": np.nan,
            "n_participant_units": np.nan,
            "sample_entry_type": np.nan,
        }

    dingFitsIndiv = _load_fit_npy("dingBLRDing.npy").item()

    savingsFits = _load_fit_npy("BHTAvrahamSavings.npy")
    WildCardFits = _load_fit_npy("BHTWildcard.npy")

    ding_df = pd.read_csv("ding_modified.csv")
    savings_df = pd.read_csv("Exp1.csv")
    wildcard_df = pd.read_csv("wildCardTask.csv")

    dataset_summary = pd.DataFrame(
        [
            _wide_dataset_summary("Ding", ding_df, ["participantNum"]),
            _wide_dataset_summary(
                "Savings", savings_df, ["participantNum", "Participant", "participant"]
            ),
            _wide_dataset_summary("Wildcard", wildcard_df, ["participantNum"]),
        ]
    )

    fit_summary = pd.DataFrame(
        [
            _fit_container_summary("dingFitsIndiv", dingFitsIndiv),
            _fit_container_summary("savingsFits", savingsFits),
            _fit_container_summary("WildCardFits", WildCardFits),
        ]
    )

    print(dataset_summary)
    print(fit_summary)

    plt.close("all")

    FREE_PARAM_NAMES = [
        "logNegLogH",
        "logVarTrans",
        "logAlpha",
        "kappa",
        "logPriorOddsStruct",
        "logVisCoeff",
        "logBeta",
    ]

    PARAM_METADATA = {
        "logNegLogH": {
            "manuscript_name": "Outer hazard",
            "raw_label": "logNegLogH",
            "transformed_label": "h",
            "description": "Higher values imply smaller outer hazard after h = exp(-exp(logNegLogH)).",
        },
        "logVarTrans": {
            "manuscript_name": "Translation scale",
            "raw_label": "logVarTrans",
            "transformed_label": "sigma_T",
            "description": "Converted to sigma_T = exp(0.5 * logVarTrans).",
        },
        "logAlpha": {
            "manuscript_name": "Coherence bias",
            "raw_label": "logAlpha",
            "transformed_label": "signed other-vs-self bias",
            "description": "Positive values favour borrowing from the other/LOO field; negative values favour self isolation.",
        },
        "kappa": {
            "manuscript_name": "Gibbs sharpness",
            "raw_label": "kappa",
            "transformed_label": "kappa",
            "description": "Higher values sharpen geometry-based weighting of cross-target information.",
        },
        "logPriorOddsStruct": {
            "manuscript_name": "Structure prior odds",
            "raw_label": "logPriorOddsStruct",
            "transformed_label": "log prior odds (translation vs rotation)",
            "description": "Positive values favour translation/harmonic structure; negative values favour rotation/DC structure.",
        },
        "logVisCoeff": {
            "manuscript_name": "Visual noise coefficient",
            "raw_label": "logVisCoeff",
            "transformed_label": "visCoeff",
            "description": "Converted to visCoeff = exp(logVisCoeff).",
        },
        "logBeta": {
            "manuscript_name": "Policy inverse temperature",
            "raw_label": "logBeta",
            "transformed_label": "beta",
            "description": "Converted to beta = exp(logBeta).",
        },
    }

    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def add_transformed_param_columns(df):
        out = df.copy()
        if "logNegLogH" in out:
            out["hazard_h"] = np.exp(-np.exp(np.clip(out["logNegLogH"], -700, 700)))
        if "logVarTrans" in out:
            out["sigma_T"] = np.exp(0.5 * np.clip(out["logVarTrans"], -700, 700))
        if "logAlpha" in out:
            out["alpha_self"] = softplus(-out["logAlpha"])
            out["alpha_other"] = softplus(out["logAlpha"])
        if "logVisCoeff" in out:
            out["visCoeff"] = np.exp(np.clip(out["logVisCoeff"], -700, 700))
        if "logBeta" in out:
            out["beta"] = np.exp(np.clip(out["logBeta"], -700, 700))
        return out

    def two_level_effect_size(x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        n1 = len(x1)
        n2 = len(x2)
        if n1 < 2 or n2 < 2:
            return np.nan
        pooled_var = ((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)) / max(n1 + n2 - 2, 1)
        if pooled_var <= 0:
            return np.nan
        return (x1.mean() - x2.mean()) / np.sqrt(pooled_var)

    def run_two_way_anova(df, dv, factor_a="geometry", factor_b="targets"):
        model = ols(f'Q("{dv}") ~ C({factor_a}) * C({factor_b})', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "effect"})
        residual_ss = anova.loc[anova["effect"] == "Residual", "sum_sq"].iloc[0]
        anova["partial_eta_sq"] = np.where(
            anova["effect"] != "Residual",
            anova["sum_sq"] / (anova["sum_sq"] + residual_ss),
            np.nan,
        )
        return model, anova

    def run_two_way_anova_with_nuisance(
        df, dv, nuisance="rotation_group", factor_a="geometry", factor_b="targets"
    ):
        model = ols(
            f'Q("{dv}") ~ C({factor_a}) * C({factor_b}) + C({nuisance})',
            data=df,
        ).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "effect"})
        residual_ss = anova.loc[anova["effect"] == "Residual", "sum_sq"].iloc[0]
        anova["partial_eta_sq"] = np.where(
            anova["effect"] != "Residual",
            anova["sum_sq"] / (anova["sum_sq"] + residual_ss),
            np.nan,
        )
        return model, anova

    def run_ding_posthocs(df, dv, anova, alpha=0.05):
        p_lookup = anova.set_index("effect")["PR(>F)"]
        tests = []

        if p_lookup.get("C(geometry)", 1.0) < alpha:
            outer = df.loc[df["geometry"] == "Outer", dv].dropna()
            inner = df.loc[df["geometry"] == "Inner", dv].dropna()
            t_stat, p_val = stats.ttest_ind(outer, inner, equal_var=False)
            tests.append(
                {
                    "contrast": "geometry: Outer vs Inner",
                    "group_a": "Outer",
                    "group_b": "Inner",
                    "n_a": len(outer),
                    "n_b": len(inner),
                    "mean_a": outer.mean(),
                    "mean_b": inner.mean(),
                    "difference_a_minus_b": outer.mean() - inner.mean(),
                    "test": "Welch t-test",
                    "statistic": t_stat,
                    "p_unc": p_val,
                    "cohen_d": two_level_effect_size(outer, inner),
                }
            )

        if p_lookup.get("C(targets)", 1.0) < alpha:
            two_t = df.loc[df["targets"] == "2T", dv].dropna()
            eight_t = df.loc[df["targets"] == "8T", dv].dropna()
            t_stat, p_val = stats.ttest_ind(two_t, eight_t, equal_var=False)
            tests.append(
                {
                    "contrast": "targets: 2T vs 8T",
                    "group_a": "2T",
                    "group_b": "8T",
                    "n_a": len(two_t),
                    "n_b": len(eight_t),
                    "mean_a": two_t.mean(),
                    "mean_b": eight_t.mean(),
                    "difference_a_minus_b": two_t.mean() - eight_t.mean(),
                    "test": "Welch t-test",
                    "statistic": t_stat,
                    "p_unc": p_val,
                    "cohen_d": two_level_effect_size(two_t, eight_t),
                }
            )

        if p_lookup.get("C(geometry):C(targets)", 1.0) < alpha:
            for target_level in ["2T", "8T"]:
                outer = df.loc[
                    (df["geometry"] == "Outer") & (df["targets"] == target_level), dv
                ].dropna()
                inner = df.loc[
                    (df["geometry"] == "Inner") & (df["targets"] == target_level), dv
                ].dropna()
                t_stat, p_val = stats.ttest_ind(outer, inner, equal_var=False)
                tests.append(
                    {
                        "contrast": f"geometry within {target_level}: Outer vs Inner",
                        "group_a": f"Outer | {target_level}",
                        "group_b": f"Inner | {target_level}",
                        "n_a": len(outer),
                        "n_b": len(inner),
                        "mean_a": outer.mean(),
                        "mean_b": inner.mean(),
                        "difference_a_minus_b": outer.mean() - inner.mean(),
                        "test": "Welch t-test",
                        "statistic": t_stat,
                        "p_unc": p_val,
                        "cohen_d": two_level_effect_size(outer, inner),
                    }
                )

            for geometry_level in ["Outer", "Inner"]:
                two_t = df.loc[
                    (df["geometry"] == geometry_level) & (df["targets"] == "2T"), dv
                ].dropna()
                eight_t = df.loc[
                    (df["geometry"] == geometry_level) & (df["targets"] == "8T"), dv
                ].dropna()
                t_stat, p_val = stats.ttest_ind(two_t, eight_t, equal_var=False)
                tests.append(
                    {
                        "contrast": f"targets within {geometry_level}: 2T vs 8T",
                        "group_a": f"{geometry_level} | 2T",
                        "group_b": f"{geometry_level} | 8T",
                        "n_a": len(two_t),
                        "n_b": len(eight_t),
                        "mean_a": two_t.mean(),
                        "mean_b": eight_t.mean(),
                        "difference_a_minus_b": two_t.mean() - eight_t.mean(),
                        "test": "Welch t-test",
                        "statistic": t_stat,
                        "p_unc": p_val,
                        "cohen_d": two_level_effect_size(two_t, eight_t),
                    }
                )

        if not tests:
            return pd.DataFrame(
                columns=[
                    "contrast",
                    "group_a",
                    "group_b",
                    "n_a",
                    "n_b",
                    "mean_a",
                    "mean_b",
                    "difference_a_minus_b",
                    "test",
                    "statistic",
                    "p_unc",
                    "p_holm",
                    "reject_holm",
                    "cohen_d",
                ]
            )

        posthoc = pd.DataFrame(tests)
        reject, p_holm, _, _ = multipletests(posthoc["p_unc"], method="holm")
        posthoc["p_holm"] = p_holm
        posthoc["reject_holm"] = reject
        return posthoc.sort_values(["p_holm", "p_unc", "contrast"]).reset_index(drop=True)

    plt.close("all")

    def extract_ding_param_table(ding_fit_dict):
        rows = []
        for condition, fit in ding_fit_dict.items():
            geometry, targets, rotation_group = condition.split("_")
            for participant_num, xs in zip(fit.participantNums, fit.xs):
                xs = list(xs)
                if len(xs) != 9:
                    continue
                record = {
                    "participantNum": participant_num,
                    "condition": condition,
                    "geometry": geometry,
                    "targets": targets,
                    "rotation_group": rotation_group,
                    "scale_S0": float(xs[0]),
                    "nu_S0": float(xs[1]),
                }
                for name, value in zip(FREE_PARAM_NAMES, xs[2:]):
                    record[name] = float(value)
                rows.append(record)
        return add_transformed_param_columns(pd.DataFrame(rows))

    ding_param_df = extract_ding_param_table(dingFitsIndiv)

    param_overview = pd.DataFrame(
        [
            {
                "raw_parameter": name,
                "manuscript_name": PARAM_METADATA[name]["manuscript_name"],
                "transformed_summary": PARAM_METADATA[name]["transformed_label"],
                "interpretation": PARAM_METADATA[name]["description"],
            }
            for name in FREE_PARAM_NAMES
        ]
    )

    print(param_overview)
    print(
        ding_param_df.groupby(["geometry", "targets"])
        .agg(n_participants=("participantNum", "nunique"))
        .reset_index()
    )

    plt.close("all")

    ding_param_descriptives = (
        ding_param_df.groupby(["geometry", "targets"])[FREE_PARAM_NAMES]
        .agg(["mean", "std"])
        .round(4)
    )
    print(ding_param_descriptives)

    plt.close("all")

    ding_anova_tables = {}
    ding_posthoc_tables = {}
    ding_anova_rows = []

    for dv in FREE_PARAM_NAMES:
        _, anova = run_two_way_anova(ding_param_df, dv)
        ding_anova_tables[dv] = anova
        effect_rows = anova.loc[
            anova["effect"] != "Residual", ["effect", "F", "PR(>F)", "partial_eta_sq"]
        ].copy()
        effect_rows.insert(0, "parameter", dv)
        ding_anova_rows.append(effect_rows)
        ding_posthoc_tables[dv] = run_ding_posthocs(ding_param_df, dv, anova)

    ding_anova_results = pd.concat(ding_anova_rows, ignore_index=True)

    for effect in ding_anova_results["effect"].unique():
        mask = ding_anova_results["effect"] == effect
        ding_anova_results.loc[mask, "p_fdr_across_params"] = multipletests(
            ding_anova_results.loc[mask, "PR(>F)"],
            method="fdr_bh",
        )[1]

    ding_anova_results = ding_anova_results.sort_values(["effect", "parameter"]).reset_index(
        drop=True
    )
    print(ding_anova_results)

    plt.close("all")

    significant_ding_posthocs = {
        param: table for param, table in ding_posthoc_tables.items() if len(table) > 0
    }

    if significant_ding_posthocs:
        for param, table in significant_ding_posthocs.items():
            print(f"\n### {param}")
            print(table)
    else:
        print("No ANOVA effects crossed p < .05, so no post-hoc tests were required.")

    plt.close("all")

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 14))
    axes = axes.flatten()

    for ax, dv in zip(axes, FREE_PARAM_NAMES):
        sns.pointplot(
            data=ding_param_df,
            x="targets",
            y=dv,
            hue="geometry",
            dodge=True,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(dv)
        ax.set_xlabel("Targets")
        ax.set_ylabel("Fitted value")
        ax.legend_.remove()

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Ding free-parameter means by geometry and target count", y=1.01)
    fig.tight_layout()

    plt.close("all")

    def extract_ding_structure_summary(ding_fit_dict):
        rows = []
        for condition, fit in ding_fit_dict.items():
            geometry, targets, rotation_group = condition.split("_")
            for participant_num in fit.participantNums:
                pred_state = fit.predState[participant_num]
                log_weights = np.asarray(pred_state["predStructLogW"], dtype=float)
                if log_weights.ndim != 2 or log_weights.shape[1] != 2:
                    continue

                probs = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
                p_rotation = probs[:, 1]

                participant_trials = fit.df.loc[
                    fit.df["participantNum"] == participant_num
                ].reset_index(drop=True)
                rotation_idx = np.flatnonzero(
                    participant_trials["rotation"].to_numpy(dtype=float) != 0
                )
                if len(rotation_idx) == 0:
                    continue

                early_idx, mid_idx, late_idx = np.array_split(rotation_idx, 3)

                rows.append(
                    {
                        "participantNum": participant_num,
                        "condition": condition,
                        "geometry": geometry,
                        "targets": targets,
                        "rotation_group": rotation_group,
                        "p_rotation_early": float(np.nanmean(p_rotation[early_idx])),
                        "p_rotation_mid": float(np.nanmean(p_rotation[mid_idx])),
                        "p_rotation_late": float(np.nanmean(p_rotation[late_idx])),
                        "p_rotation_delta_late_minus_early": float(
                            np.nanmean(p_rotation[late_idx]) - np.nanmean(p_rotation[early_idx])
                        ),
                    }
                )
        return pd.DataFrame(rows)

    ding_structure_df = extract_ding_structure_summary(dingFitsIndiv)

    plt.close("all")

    ding_structure_descriptives = (
        ding_structure_df.groupby(["geometry", "targets"])[
            [
                "p_rotation_early",
                "p_rotation_mid",
                "p_rotation_late",
                "p_rotation_delta_late_minus_early",
            ]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    print(ding_structure_descriptives)

    plt.close("all")

    structure_dvs = [
        "p_rotation_early",
        "p_rotation_mid",
        "p_rotation_late",
        "p_rotation_delta_late_minus_early",
    ]

    ding_structure_anova_tables = {}
    ding_structure_posthocs = {}
    ding_structure_anova_rows = []

    for dv in structure_dvs:
        _, anova = run_two_way_anova(ding_structure_df, dv)
        ding_structure_anova_tables[dv] = anova
        effect_rows = anova.loc[
            anova["effect"] != "Residual", ["effect", "F", "PR(>F)", "partial_eta_sq"]
        ].copy()
        effect_rows.insert(0, "metric", dv)
        ding_structure_anova_rows.append(effect_rows)
        ding_structure_posthocs[dv] = run_ding_posthocs(ding_structure_df, dv, anova)

    ding_structure_anova_results = pd.concat(ding_structure_anova_rows, ignore_index=True)
    print(ding_structure_anova_results)

    plt.close("all")

    for metric, table in ding_structure_posthocs.items():
        if len(table) == 0:
            continue
        print(f"\n### {metric}")
        print(table)

    plt.close("all")

    structure_plot_df = ding_structure_df.melt(
        id_vars=["participantNum", "geometry", "targets"],
        value_vars=["p_rotation_early", "p_rotation_mid", "p_rotation_late"],
        var_name="phase",
        value_name="p_rotation",
    ).assign(
        phase=lambda d: d["phase"].map(
            {
                "p_rotation_early": "Early",
                "p_rotation_mid": "Mid",
                "p_rotation_late": "Late",
            }
        )
    )

    g = sns.catplot(
        data=structure_plot_df,
        kind="point",
        x="phase",
        y="p_rotation",
        hue="geometry",
        col="targets",
        errorbar=("ci", 95),
        dodge=True,
        height=4.2,
        aspect=0.95,
    )
    g.fig.suptitle("Ding structural belief: mean P(rotation) over the rotation phase", y=1.02)
    plt.show()

    plt.close("all")

    WILDCARD_FINAL_BLOCK = 4

    def parse_array_col(text):
        return json.loads(text.replace("nan", "null"))

    def phase_boundaries(rotations):
        first = next(i for i, r in enumerate(rotations) if r is not None and r != 0)
        last = max(i for i, r in enumerate(rotations) if r is not None and r != 0)
        return first, last

    def build_wildcard_tables(csv_path="wildCardTask.csv", final_n=WILDCARD_FINAL_BLOCK):
        wide = pd.read_csv(csv_path)
        rotation_phase_rows = []
        final_mag_rows = []
        final_target_rows = []
        permutation_rows = []

        for _, row in wide.iterrows():
            participant_num = int(row["participantNum"])
            rotations = parse_array_col(row["rotation"])
            aims = parse_array_col(row["aim"])
            targets = parse_array_col(row["targetPosition"])

            first_rot, last_rot = phase_boundaries(rotations)

            mag_series = {}
            target_series = {}
            target_to_signed_rotation = {}

            for rotation, target in zip(rotations, targets):
                if target is None:
                    continue
                target = float(target)
                target_to_signed_rotation.setdefault(target, 0.0)
                if rotation is not None and rotation != 0:
                    target_to_signed_rotation[target] = float(rotation)

            for trial_idx in range(first_rot, last_rot + 1):
                rotation = rotations[trial_idx]
                aim = aims[trial_idx]
                target = targets[trial_idx]

                if aim is None or target is None:
                    continue

                target = float(target)
                rotation = 0.0 if rotation is None else float(rotation)
                magnitude = int(abs(rotation)) if rotation != 0 else 0
                signed_comp = float(aim) if rotation == 0 else -float(aim) * np.sign(rotation)

                rotation_phase_rows.append(
                    {
                        "participantNum": participant_num,
                        "trial_idx": trial_idx,
                        "target": target,
                        "rotation": rotation,
                        "magnitude": magnitude,
                        "signed_comp": signed_comp,
                    }
                )

                mag_series.setdefault(magnitude, []).append(signed_comp)
                target_series.setdefault(target, []).append(signed_comp)

            for magnitude, values in mag_series.items():
                if len(values) >= final_n:
                    final_mag_rows.append(
                        {
                            "participantNum": participant_num,
                            "magnitude": magnitude,
                            "final_comp": float(np.nanmean(values[-final_n:])),
                            "n_trials_at_magnitude": len(values),
                        }
                    )

            target_final_means = []
            for target in sorted(target_to_signed_rotation):
                values = target_series.get(target, [])
                if len(values) >= final_n:
                    final_mean = float(np.nanmean(values[-final_n:]))
                    signed_rotation = float(target_to_signed_rotation[target])
                    final_target_rows.append(
                        {
                            "participantNum": participant_num,
                            "target": target,
                            "true_rotation": signed_rotation,
                            "abs_rotation": int(abs(signed_rotation)),
                            "final_comp": final_mean,
                        }
                    )
                    target_final_means.append((target, signed_rotation, final_mean))

            if len(target_final_means) == 4:
                target_final_means = sorted(target_final_means, key=lambda x: x[0])
                signed_rotations = np.array([item[1] for item in target_final_means], dtype=float)
                final_means = np.array([item[2] for item in target_final_means], dtype=float)

                all_perm_maes = np.array(
                    [
                        np.mean(np.abs(final_means - np.array(perm, dtype=float)))
                        for perm in itertools.permutations(signed_rotations)
                    ]
                )
                actual_mae = float(np.mean(np.abs(final_means - signed_rotations)))

                permutation_rows.append(
                    {
                        "participantNum": participant_num,
                        "actual_mae": actual_mae,
                        "perm_mae_mean": float(all_perm_maes.mean()),
                        "perm_mae_median": float(np.median(all_perm_maes)),
                        "perm_mae_min": float(all_perm_maes.min()),
                        "delta_mae_vs_perm_mean": float(all_perm_maes.mean() - actual_mae),
                        "actual_better_than_perm_mean": bool(actual_mae < all_perm_maes.mean()),
                        "actual_is_best_permutation": bool(
                            np.isclose(actual_mae, all_perm_maes.min())
                        ),
                    }
                )

        rotation_phase_df = pd.DataFrame(rotation_phase_rows)
        final_mag_df = pd.DataFrame(final_mag_rows)
        final_target_df = pd.DataFrame(final_target_rows)
        permutation_df = pd.DataFrame(permutation_rows)
        return rotation_phase_df, final_mag_df, final_target_df, permutation_df

    (
        wildcard_rotation_phase_df,
        wildcard_final_mag_df,
        wildcard_final_target_df,
        wildcard_permutation_df,
    ) = build_wildcard_tables()

    print(
        pd.DataFrame(
            {
                "table": [
                    "wildcard_rotation_phase_df",
                    "wildcard_final_mag_df",
                    "wildcard_final_target_df",
                    "wildcard_permutation_df",
                ],
                "rows": [
                    len(wildcard_rotation_phase_df),
                    len(wildcard_final_mag_df),
                    len(wildcard_final_target_df),
                    len(wildcard_permutation_df),
                ],
            }
        )
    )

    print(
        wildcard_final_mag_df.groupby("magnitude")
        .agg(
            n_rows=("participantNum", "size"),
            n_participants=("participantNum", "nunique"),
            mean_final_comp=("final_comp", "mean"),
            sd_final_comp=("final_comp", "std"),
        )
        .reset_index()
    )

    plt.close("all")

    wildcard_final_mag_df["magnitude_str"] = wildcard_final_mag_df["magnitude"].astype(str)

    wildcard_magnitude_mixedlm = smf.mixedlm(
        "final_comp ~ C(magnitude_str)",
        wildcard_final_mag_df,
        groups=wildcard_final_mag_df["participantNum"],
    ).fit(reml=False)

    print(wildcard_magnitude_mixedlm.summary())

    plt.close("all")

    wildcard_pivot = wildcard_final_mag_df.pivot(
        index="participantNum",
        columns="magnitude",
        values="final_comp",
    )

    pairwise_rows = []
    for magnitude_a, magnitude_b in [(0, 15), (0, 30), (0, 45), (15, 30), (15, 45), (30, 45)]:
        paired = wildcard_pivot[[magnitude_a, magnitude_b]].dropna()
        t_stat, p_val = stats.ttest_rel(paired[magnitude_a], paired[magnitude_b])
        pairwise_rows.append(
            {
                "contrast": f"{magnitude_a} vs {magnitude_b}",
                "n_pairs": len(paired),
                "mean_a": paired[magnitude_a].mean(),
                "mean_b": paired[magnitude_b].mean(),
                "mean_diff_b_minus_a": (paired[magnitude_b] - paired[magnitude_a]).mean(),
                "t_stat": t_stat,
                "p_unc": p_val,
            }
        )

    wildcard_pairwise = pd.DataFrame(pairwise_rows)
    wildcard_pairwise["p_holm"] = multipletests(wildcard_pairwise["p_unc"], method="holm")[1]
    wildcard_pairwise = wildcard_pairwise.sort_values("p_holm").reset_index(drop=True)
    print(wildcard_pairwise)

    plt.close("all")

    wildcard_one_sample_rows = []
    for magnitude in sorted(wildcard_final_mag_df["magnitude"].unique()):
        values = wildcard_final_mag_df.loc[
            wildcard_final_mag_df["magnitude"] == magnitude, "final_comp"
        ].dropna()
        comparison_value = 0 if magnitude == 0 else magnitude
        t_stat, p_val = stats.ttest_1samp(values, popmean=comparison_value)
        wildcard_one_sample_rows.append(
            {
                "magnitude": magnitude,
                "n": len(values),
                "mean_final_comp": values.mean(),
                "sd_final_comp": values.std(ddof=1),
                "tested_against": comparison_value,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    wildcard_one_sample = pd.DataFrame(wildcard_one_sample_rows)
    print(wildcard_one_sample)

    plt.close("all")

    wildcard_catch = wildcard_final_mag_df.loc[wildcard_final_mag_df["magnitude"] == 0].set_index(
        "participantNum"
    )["final_comp"]
    wildcard_trained_mean = (
        wildcard_final_mag_df.loc[wildcard_final_mag_df["magnitude"] > 0]
        .groupby("participantNum")["final_comp"]
        .mean()
    )
    paired_pp = wildcard_catch.index.intersection(wildcard_trained_mean.index)
    catch_vs_trained_t, catch_vs_trained_p = stats.ttest_rel(
        wildcard_trained_mean.loc[paired_pp],
        wildcard_catch.loc[paired_pp],
    )

    catch_summary = pd.DataFrame(
        [
            {
                "comparison": "trained-target mean vs catch target",
                "n_pairs": len(paired_pp),
                "mean_trained": wildcard_trained_mean.loc[paired_pp].mean(),
                "mean_catch": wildcard_catch.loc[paired_pp].mean(),
                "t_stat": catch_vs_trained_t,
                "p_value": catch_vs_trained_p,
            }
        ]
    )
    print(catch_summary)

    plt.close("all")

    wildcard_permutation_summary = pd.DataFrame(
        [
            {
                "n_participants": len(wildcard_permutation_df),
                "mean_actual_mae": wildcard_permutation_df["actual_mae"].mean(),
                "mean_perm_mae": wildcard_permutation_df["perm_mae_mean"].mean(),
                "mean_delta_mae_vs_perm_mean": wildcard_permutation_df[
                    "delta_mae_vs_perm_mean"
                ].mean(),
                "prop_actual_better_than_perm_mean": wildcard_permutation_df[
                    "actual_better_than_perm_mean"
                ].mean(),
                "prop_actual_is_best_permutation": wildcard_permutation_df[
                    "actual_is_best_permutation"
                ].mean(),
            }
        ]
    )

    delta_t, delta_p = stats.ttest_1samp(
        wildcard_permutation_df["delta_mae_vs_perm_mean"],
        popmean=0.0,
    )

    wildcard_permutation_inference = pd.DataFrame(
        [
            {
                "metric": "delta_mae_vs_perm_mean",
                "mean": wildcard_permutation_df["delta_mae_vs_perm_mean"].mean(),
                "sd": wildcard_permutation_df["delta_mae_vs_perm_mean"].std(ddof=1),
                "t_stat": delta_t,
                "p_value": delta_p,
            }
        ]
    )

    print(wildcard_permutation_summary)
    print(wildcard_permutation_inference)

    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    sns.violinplot(
        data=wildcard_final_mag_df,
        x="magnitude",
        y="final_comp",
        inner=None,
        cut=0,
        ax=axes[0],
        color="#c6dbef",
    )
    sns.stripplot(
        data=wildcard_final_mag_df,
        x="magnitude",
        y="final_comp",
        ax=axes[0],
        color="#2171b5",
        alpha=0.55,
        size=4,
    )
    for magnitude in [15, 30, 45]:
        axes[0].axhline(magnitude, ls=":", lw=1, color="grey", alpha=0.5)
    axes[0].axhline(0, ls="--", lw=1, color="black", alpha=0.5)
    axes[0].set_title("Wildcard final-block compensation by magnitude")
    axes[0].set_xlabel("Perturbation magnitude (°)")
    axes[0].set_ylabel("Final-block compensatory aim (°)")

    permutation_plot_df = wildcard_permutation_df.melt(
        value_vars=["actual_mae", "perm_mae_mean"],
        var_name="metric",
        value_name="mae",
    )
    sns.violinplot(
        data=permutation_plot_df,
        x="metric",
        y="mae",
        inner=None,
        cut=0,
        ax=axes[1],
        color="#fdd0a2",
    )
    sns.stripplot(
        data=permutation_plot_df,
        x="metric",
        y="mae",
        ax=axes[1],
        color="#d94801",
        alpha=0.4,
        size=3.5,
    )
    axes[1].set_title("Wildcard coupling fidelity vs shuffled mappings")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean absolute error to target-specific rotation (°)")
    axes[1].set_xticklabels(["Actual mapping", "Mean shuffled mapping"])

    fig.tight_layout()

    plt.close("all")

    ding_sensitivity_rows = []
    for dv in FREE_PARAM_NAMES:
        _, anova = run_two_way_anova_with_nuisance(ding_param_df, dv)
        subset = anova.loc[
            anova["effect"].isin(
                ["C(geometry)", "C(targets)", "C(rotation_group)", "C(geometry):C(targets)"]
            ),
            ["effect", "F", "PR(>F)", "partial_eta_sq"],
        ].copy()
        subset.insert(0, "parameter", dv)
        ding_sensitivity_rows.append(subset)

    ding_sensitivity_results = pd.concat(ding_sensitivity_rows, ignore_index=True)
    print(ding_sensitivity_results)

    plt.close("all")

    table_dir = Path("statistics")
    table_dir.mkdir(exist_ok=True)
    tables = {
        "dataset_summary": dataset_summary,
        "fit_summary": fit_summary,
        "ding_param_df": ding_param_df,
        "ding_param_descriptives": ding_param_descriptives,
        "ding_anova_results": ding_anova_results,
        "ding_structure_df": ding_structure_df,
        "ding_structure_descriptives": ding_structure_descriptives,
        "ding_structure_anova_results": ding_structure_anova_results,
        "wildcard_pairwise": wildcard_pairwise,
        "wildcard_one_sample": wildcard_one_sample,
        "catch_summary": catch_summary,
        "wildcard_permutation_summary": wildcard_permutation_summary,
        "wildcard_permutation_inference": wildcard_permutation_inference,
        "ding_sensitivity_results": ding_sensitivity_results,
    }
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=True)
    (table_dir / "wildcard_magnitude_mixedlm.txt").write_text(
        str(wildcard_magnitude_mixedlm.summary()), encoding="utf-8"
    )
    print(f"Saved statistics to {table_dir}")


def manuscript():
    from pathlib import Path
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.special import logsumexp
    from statsmodels.formula.api import ols
    from statsmodels.stats.multitest import multipletests

    import blr_ding_manuscript_helpers as dingh
    import blr_wildcard_manuscript_helpers as wildh

    warnings.filterwarnings("ignore")

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
        }
    )

    OUTPUT_DIR = Path("BLRManuscriptOutputs")
    FIG_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    plt.close("all")

    ding_bundle = dingh.run_ding_manuscript_pipeline(save_dir=None)

    ding_param_df = ding_bundle["param_df"]
    ding_logprior_simple_df = ding_bundle["simple_effect_df"]
    ding_logprior_nuisance_df = ding_bundle["nuisance_df"]
    ding_structure_df = ding_bundle["structure_df"]
    ding_alignment_df = ding_bundle["alignment_df"]
    ding_alignment_pp_df = ding_bundle["participant_alignment_df"]
    ding_alignment_exposure_df = ding_bundle["exposure_alignment_df"]

    ding_logprior_simple_df.to_csv(TABLE_DIR / "ding_logprior_simple_effects.csv", index=False)
    ding_logprior_nuisance_df.to_csv(TABLE_DIR / "ding_logprior_nuisance.csv", index=False)

    plt.close("all")

    print(ding_logprior_simple_df)

    plt.close("all")

    print(ding_logprior_nuisance_df)

    plt.close("all")

    fig, ax = dingh.figure_ding_logprior_simple_effects(ding_logprior_simple_df)
    dingh.save_figure_bundle(fig, FIG_DIR / "ding_logprior_simple_effects")
    plt.show()
    plt.close(fig)

    plt.close("all")

    first4_model = ols("p_rotation_first4 ~ C(geometry) * C(targets)", data=ding_structure_df).fit()
    first4_model_anova = sm.stats.anova_lm(first4_model, typ=2)
    first4_model_anova["partial_eta_sq"] = first4_model_anova["sum_sq"] / (
        first4_model_anova["sum_sq"] + first4_model_anova.loc["Residual", "sum_sq"]
    )
    first4_model_anova.to_csv(TABLE_DIR / "ding_p_rotation_first4_anova.csv")
    print(first4_model_anova)

    plt.close("all")

    def build_alignment_phase_table(alignment_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (participant_num, geometry, targets), sub in alignment_df.groupby(
            ["participantNum", "geometry", "targets"]
        ):
            sub = sub.sort_values("gen_trial_index")
            p_trans = sub["p_trans_model"].to_numpy(dtype=float)
            human = sub["human_sign_flip"].to_numpy(dtype=float)
            if len(p_trans) < 6:
                continue
            thirds = np.array_split(np.arange(len(p_trans)), 3)
            early = np.arange(min(4, len(p_trans)))
            rows.append(
                {
                    "participantNum": participant_num,
                    "geometry": geometry,
                    "targets": targets,
                    "model_ptrans_first4": float(np.nanmean(p_trans[early])),
                    "human_flip_first4": float(np.nanmean(human[early])),
                    "model_ptrans_mid": float(np.nanmean(p_trans[thirds[1]])),
                    "human_flip_mid": float(np.nanmean(human[thirds[1]])),
                    "model_ptrans_late": float(np.nanmean(p_trans[thirds[2]])),
                    "human_flip_late": float(np.nanmean(human[thirds[2]])),
                    "model_rot_delta": float(
                        np.nanmean(1 - p_trans[thirds[2]]) - np.nanmean(1 - p_trans[early])
                    ),
                    "human_rot_delta": float(
                        np.nanmean(1 - human[thirds[2]]) - np.nanmean(1 - human[early])
                    ),
                }
            )
        return pd.DataFrame(rows)

    ding_alignment_phase_df = build_alignment_phase_table(ding_alignment_df)
    ding_alignment_phase_df.to_csv(TABLE_DIR / "ding_alignment_phase_summary.csv", index=False)

    for dv in [
        "model_ptrans_first4",
        "human_flip_first4",
        "model_ptrans_mid",
        "human_flip_mid",
        "model_ptrans_late",
        "human_flip_late",
        "model_rot_delta",
        "human_rot_delta",
    ]:
        model = ols(f"{dv} ~ C(geometry) * C(targets)", data=ding_alignment_phase_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        anova["partial_eta_sq"] = anova["sum_sq"] / (
            anova["sum_sq"] + anova.loc["Residual", "sum_sq"]
        )
        anova.to_csv(TABLE_DIR / f"ding_{dv}_anova.csv")

    ding_alignment_phase_df.groupby(["geometry", "targets"])[
        [
            "model_ptrans_first4",
            "human_flip_first4",
            "model_ptrans_mid",
            "human_flip_mid",
            "model_ptrans_late",
            "human_flip_late",
            "model_rot_delta",
            "human_rot_delta",
        ]
    ].mean()

    plt.close("all")

    def simple_geometry_tests(df: pd.DataFrame, dvs: list[str]) -> pd.DataFrame:
        rows = []
        for dv in dvs:
            pvals = []
            stash = []
            for targets in ["2T", "8T"]:
                sub = df.loc[df["targets"] == targets]
                inner = sub.loc[sub["geometry"] == "Inner", dv].astype(float)
                outer = sub.loc[sub["geometry"] == "Outer", dv].astype(float)
                t_res = stats.ttest_ind(inner, outer, equal_var=True)
                pvals.append(t_res.pvalue)
                stash.append(
                    {
                        "dv": dv,
                        "targets": targets,
                        "mean_inner": float(inner.mean()),
                        "mean_outer": float(outer.mean()),
                        "t_stat": float(t_res.statistic),
                        "p_unc": float(t_res.pvalue),
                    }
                )
            p_adj = multipletests(pvals, method="holm")[1]
            for rec, adj in zip(stash, p_adj):
                rec["p_holm"] = float(adj)
                rows.append(rec)
        return pd.DataFrame(rows)

    ding_alignment_simple_df = simple_geometry_tests(
        ding_alignment_phase_df,
        [
            "model_ptrans_first4",
            "human_flip_first4",
            "model_ptrans_mid",
            "human_flip_mid",
            "model_ptrans_late",
            "human_flip_late",
            "model_rot_delta",
            "human_rot_delta",
        ],
    )
    ding_alignment_simple_df.to_csv(TABLE_DIR / "ding_alignment_simple_effects.csv", index=False)
    print(ding_alignment_simple_df)

    plt.close("all")

    overall_alignment_r, overall_alignment_p = stats.pearsonr(
        ding_alignment_exposure_df["mean_p_trans_model"],
        ding_alignment_exposure_df["mean_human_sign_flip"],
    )

    per_group_alignment = []
    for (geometry, targets), sub in ding_alignment_exposure_df.groupby(["geometry", "targets"]):
        r, p = stats.pearsonr(sub["mean_p_trans_model"], sub["mean_human_sign_flip"])
        per_group_alignment.append(
            {
                "geometry": geometry,
                "targets": targets,
                "n_points": int(len(sub)),
                "r": float(r),
                "p_value": float(p),
            }
        )

    alignment_corr_df = pd.DataFrame(per_group_alignment)
    alignment_corr_df.loc[len(alignment_corr_df)] = {
        "geometry": "All",
        "targets": "All",
        "n_points": int(len(ding_alignment_exposure_df)),
        "r": float(overall_alignment_r),
        "p_value": float(overall_alignment_p),
    }
    alignment_corr_df.to_csv(TABLE_DIR / "ding_alignment_correlations.csv", index=False)
    print(alignment_corr_df)

    plt.close("all")

    alignment_plot_df = ding_alignment_exposure_df.melt(
        id_vars=["geometry", "targets", "gen_trial_index"],
        value_vars=["mean_p_trans_model", "mean_human_sign_flip"],
        var_name="metric",
        value_name="value",
    )
    alignment_plot_df["metric"] = alignment_plot_df["metric"].map(
        {
            "mean_p_trans_model": "Model P(trans)",
            "mean_human_sign_flip": "Human sign-flip",
        }
    )

    g = sns.relplot(
        data=alignment_plot_df,
        kind="line",
        x="gen_trial_index",
        y="value",
        hue="geometry",
        style="metric",
        col="targets",
        markers=True,
        dashes=True,
        linewidth=2,
        height=4.2,
        aspect=1.08,
    )
    g.set_axis_labels("Generalisation trial index", "Probability")
    g.set_titles("{col_name}")
    g.fig.suptitle("Ding generalisation: model P(trans) versus human sign-flip", y=1.03)
    dingh.save_figure_bundle(g.fig, FIG_DIR / "ding_model_human_alignment")
    plt.show()
    plt.close(g.fig)

    plt.close("all")

    def build_ding_targetclass_phase_table(ding_fits) -> pd.DataFrame:
        rows = []
        for condition, fit in ding_fits.items():
            geometry, targets, gen_target = condition.split("_")
            for participant_num in fit.participantNums:
                statuses = np.asarray(fit.trialStatuses[participant_num])
                df_pp = (
                    fit.df.loc[fit.df["participantNum"] == participant_num]
                    .sort_values("trial_number")
                    .reset_index(drop=True)
                )
                if len(df_pp) > len(statuses):
                    df_pp = df_pp.iloc[: len(statuses)].copy()
                if len(df_pp) != len(statuses):
                    continue
                df_pp["trialStatus"] = statuses
                rot_df = df_pp.loc[df_pp["rotation"] != 0].copy().reset_index(drop=True)
                if rot_df.empty:
                    continue
                aim_col = "aim_signed" if "aim_signed" in rot_df.columns else "aim"
                rot_df["target_class"] = np.where(
                    rot_df["trialStatus"] == "generalisation",
                    "Generalisation",
                    "Training",
                )
                rot_df["norm_comp"] = -rot_df[aim_col].astype(float) * np.sign(
                    rot_df["rotation"].astype(float)
                )

                for target_class, sub in rot_df.groupby("target_class"):
                    thirds = np.array_split(np.arange(len(sub)), 3)
                    for phase, idxs in zip(["Early", "Mid", "Late"], thirds):
                        rows.append(
                            {
                                "participantNum": participant_num,
                                "condition": condition,
                                "geometry": geometry,
                                "targets": targets,
                                "gen_target": gen_target,
                                "target_class": target_class,
                                "phase": phase,
                                "mean_norm_comp": float(sub.iloc[idxs]["norm_comp"].mean()),
                            }
                        )
        return pd.DataFrame(rows)

    ding_targetclass_phase_df = build_ding_targetclass_phase_table(ding_bundle["fits"])
    ding_targetclass_phase_df.to_csv(TABLE_DIR / "ding_targetclass_phase_summary.csv", index=False)

    targetclass_mixedlm = smf.mixedlm(
        "mean_norm_comp ~ C(target_class) * C(phase) * C(geometry) * C(targets)",
        ding_targetclass_phase_df,
        groups=ding_targetclass_phase_df["participantNum"],
    ).fit(reml=False)

    with open(TABLE_DIR / "ding_targetclass_mixedlm.txt", "w", encoding="utf-8") as f:
        f.write(str(targetclass_mixedlm.summary()))

    plt.close("all")

    targetclass_simple_rows = []
    for target_class in ["Training", "Generalisation"]:
        for phase in ["Early", "Mid", "Late"]:
            sub = ding_targetclass_phase_df.loc[
                (ding_targetclass_phase_df["target_class"] == target_class)
                & (ding_targetclass_phase_df["phase"] == phase)
            ]
            pvals = []
            stash = []
            for targets in ["2T", "8T"]:
                ss = sub.loc[sub["targets"] == targets]
                inner = ss.loc[ss["geometry"] == "Inner", "mean_norm_comp"].astype(float)
                outer = ss.loc[ss["geometry"] == "Outer", "mean_norm_comp"].astype(float)
                t_res = stats.ttest_ind(inner, outer, equal_var=True)
                pvals.append(t_res.pvalue)
                stash.append(
                    {
                        "target_class": target_class,
                        "phase": phase,
                        "targets": targets,
                        "mean_inner": float(inner.mean()),
                        "mean_outer": float(outer.mean()),
                        "t_stat": float(t_res.statistic),
                        "p_unc": float(t_res.pvalue),
                    }
                )
            p_adj = multipletests(pvals, method="holm")[1]
            for rec, adj in zip(stash, p_adj):
                rec["p_holm"] = float(adj)
                targetclass_simple_rows.append(rec)

    ding_targetclass_simple_df = pd.DataFrame(targetclass_simple_rows)
    ding_targetclass_simple_df.to_csv(
        TABLE_DIR / "ding_targetclass_simple_effects.csv", index=False
    )
    print(ding_targetclass_simple_df)

    plt.close("all")

    g = sns.catplot(
        data=ding_targetclass_phase_df,
        kind="point",
        x="phase",
        y="mean_norm_comp",
        hue="geometry",
        row="target_class",
        col="targets",
        errorbar=("ci", 95),
        dodge=True,
        markers=["o", "s"],
        linestyles=["-", "-"],
        height=3.4,
        aspect=1.08,
    )
    g.set_axis_labels("Rotation phase", "Human sign-normalised compensation")
    g.set_titles("{row_name} | {col_name}")
    g.fig.suptitle(
        "Ding human data: training and generalisation targets diverge by geometry", y=1.03
    )
    dingh.save_figure_bundle(g.fig, FIG_DIR / "ding_human_training_vs_generalisation")
    plt.show()
    plt.close(g.fig)

    plt.close("all")

    wild_bundle = wildh.build_wildcard_analysis_bundle()

    wild_bundle["participant_metrics_df"].to_csv(
        TABLE_DIR / "wildcard_participant_metrics.csv", index=False
    )
    wild_bundle["param_df"].to_csv(TABLE_DIR / "wildcard_param_table.csv", index=False)
    wild_bundle["merged_df"].to_csv(TABLE_DIR / "wildcard_metrics_params_merged.csv", index=False)
    wild_bundle["correlation_df"].to_csv(TABLE_DIR / "wildcard_correlations.csv", index=False)
    wild_bundle["regression_df"].to_csv(TABLE_DIR / "wildcard_regressions.csv", index=False)

    plt.close("all")

    plt.close("all")

    plt.close("all")

    plt.close("all")

    fig = wildh.plot_wildcard_behavior_summary(
        wild_bundle["final_mag_df"],
        wild_bundle["final_target_df"],
        wild_bundle["participant_metrics_df"],
    )
    wildh.save_figure_triplet(fig, FIG_DIR / "wildcard_behavior_summary")
    plt.show()
    plt.close(fig)

    plt.close("all")

    def annotate_spearman(ax, x, y):
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(sub) < 5:
            return
        rho, p = stats.spearmanr(sub["x"], sub["y"])
        ax.text(
            0.03,
            0.97,
            f"rho = {rho:0.2f}\np = {p:0.3g}\nn = {len(sub)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.8",
            },
            fontsize=9,
        )

    wild_plot_specs = [
        ("trained_mean", "sigma_T", "Trained compensation vs translation scale"),
        ("trained_mean", "alpha_other_minus_self", "Trained compensation vs other-self bias"),
        ("coupling_fidelity", "alpha_other_minus_self", "Coupling fidelity vs other-self bias"),
        ("coupling_fidelity", "sigma_T", "Coupling fidelity vs translation scale"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    axes = axes.flatten()
    for ax, (metric, param, title) in zip(axes, wild_plot_specs):
        sub = wild_bundle["merged_df"][[metric, param]].dropna()
        sns.regplot(
            data=sub,
            x=param,
            y=metric,
            ax=ax,
            scatter_kws={"s": 42, "alpha": 0.75, "color": "#355C7D"},
            line_kws={"color": "#C06C84", "lw": 2},
        )
        annotate_spearman(ax, sub[param], sub[metric])
        ax.set_title(title)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)

    fig.suptitle("Wildcard individual learning ability mapped onto fitted parameters", y=1.02)
    fig.tight_layout()
    wildh.save_figure_triplet(fig, FIG_DIR / "wildcard_parameter_map_custom")
    plt.show()
    plt.close(fig)

    plt.close("all")

    fig = wildh.plot_wildcard_regression_heatmap(wild_bundle["correlation_df"])
    wildh.save_figure_triplet(fig, FIG_DIR / "wildcard_correlation_heatmap")
    plt.show()
    plt.close(fig)

    plt.close("all")

    plt.close("all")


def wildcard_slope_statistics():
    from itertools import combinations
    from pathlib import Path
    import importlib
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm
    import wildcard_pen12_window16_multipanel as wc_multi

    importlib.reload(wc_multi)

    wildcardWindowCycles = 16
    wildcardHeterogeneityFolder = Path('BLRFigures') / 'WildCardFigures' / 'heterogeneity_trueother'
    wildcardHeterogeneityFolder.mkdir(exist_ok=True, parents=True)

    fit, recovered = wc_multi.load_fit_and_recovery()
    panelA = wc_multi.build_panel_a_table(fit, recovered)
    learnerPP = panelA.loc[panelA['learner'], 'pp'].astype(int).tolist()

    analysisDf = wc_multi.build_analysis_table(fit, recovered, window_cycles=wildcardWindowCycles)
    analysisDf = analysisDf[analysisDf['pp'].isin(learnerPP)].copy()

    quartileDf, quartileOrder, _ = wc_multi.assign_discrimination_groups(
        analysisDf[['pp', 'disc']].dropna(),
        scheme='quartiles',
    )
    analysisDf = analysisDf.merge(quartileDf[['pp', 'disc_group']], on='pp', how='left')

    allDf = fit.df.copy().reset_index(drop=True)
    absSlopeRows = []
    for pp in learnerPP:
        part = recovered['participants'][int(pp)]
        pdf = allDf[allDf['participantNum'] == int(pp)].copy().reset_index(drop=True)
        phase = np.asarray(part['phase'])
        if len(pdf) != len(phase):
            pdf = pdf.iloc[: len(phase)].copy().reset_index(drop=True)

        pdf['phase_rec'] = phase
        pdf['rotation_rec'] = np.asarray(part['rotation'], dtype=float)

        rotation = (
            pdf[pdf['phase_rec'].astype(str).str.lower() == 'rotation']
            .copy()
            .reset_index(drop=True)
        )
        if rotation.empty:
            continue

        rotation['rot_trial_idx'] = np.arange(len(rotation))
        rotation['cycle'] = rotation['rot_trial_idx'] // 4
        onsetSignal = (
            rotation.loc[rotation['rotation_rec'] != 0]
            .groupby('cycle')['aim']
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .sort_index()
        )
        onsetCycle = wc_multi.first_interior_breakpoint(
            onsetSignal.to_numpy(dtype=float), wc_multi.PEN
        )
        if onsetCycle is None:
            continue

        selected = rotation[
            (rotation['cycle'] >= onsetCycle)
            & (rotation['cycle'] < onsetCycle + wildcardWindowCycles)
        ].copy()
        selected['abs_rotation'] = np.abs(selected['rotation_rec'])

        byAbsRot = (
            selected.groupby('abs_rotation')['aim']
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .reset_index()
        )
        byAbsRot.columns = ['abs_rotation', 'mean_abs_aim']
        byAbsRot = byAbsRot.sort_values('abs_rotation')
        if byAbsRot['abs_rotation'].nunique() < 2:
            continue

        absSlope = float(
            np.polyfit(
                byAbsRot['abs_rotation'].to_numpy(dtype=float),
                byAbsRot['mean_abs_aim'].to_numpy(dtype=float),
                1,
            )[0]
        )
        absSlopeRows.append({'pp': int(pp), 'abs_slope': absSlope})

    absSlopeDf = pd.DataFrame(absSlopeRows)
    analysisWithSlope = analysisDf.merge(absSlopeDf, on='pp', how='inner')
    rankMap = {group: idx + 1 for idx, group in enumerate(quartileOrder)}
    analysisWithSlope['quartile_rank'] = analysisWithSlope['disc_group'].map(rankMap)

    def mean_ci(series: pd.Series) -> pd.Series:
        vals = series.dropna().to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
        tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan
        ci_half = float(tcrit * sem) if n > 1 else np.nan
        return pd.Series(
            {
                'n': n,
                'mean_abs_slope': mean,
                'sd_abs_slope': sd,
                'sem_abs_slope': sem,
                'ci95_lo': mean - ci_half if n > 1 else np.nan,
                'ci95_hi': mean + ci_half if n > 1 else np.nan,
                'median_abs_slope': float(np.median(vals)) if n else np.nan,
                'min_abs_slope': float(np.min(vals)) if n else np.nan,
                'max_abs_slope': float(np.max(vals)) if n else np.nan,
            }
        )

    def fisher_ci(r: float, n: int) -> tuple[float, float]:
        if (not np.isfinite(r)) or n <= 3 or abs(r) >= 1:
            return (np.nan, np.nan)
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))

    def hedges_g_with_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return (np.nan, np.nan, np.nan)
        s1 = np.std(x, ddof=1)
        s2 = np.std(y, ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if (not np.isfinite(pooled)) or pooled == 0:
            return (np.nan, np.nan, np.nan)
        d = (np.mean(x) - np.mean(y)) / pooled
        j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
        g = j * d
        se_g = np.sqrt((n1 + n2) / (n1 * n2) + (g**2) / (2.0 * (n1 + n2 - 2)))
        return (float(g), float(g - 1.96 * se_g), float(g + 1.96 * se_g))

    def welch_mean_diff_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n1, n2 = len(x), len(y)
        m1, m2 = np.mean(x), np.mean(y)
        s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
        v1 = s1**2 / n1
        v2 = s2**2 / n2
        se = np.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / ((v1**2) / (n1 - 1) + (v2**2) / (n2 - 1))
        tcrit = stats.t.ppf(0.975, df)
        diff = m1 - m2
        return (float(diff), float(diff - tcrit * se), float(diff + tcrit * se), float(df))

    quartileSlopeSummary = (
        analysisWithSlope.groupby('disc_group')['abs_slope'].apply(mean_ci).unstack().reset_index()
    )
    quartileSlopeSummary['quartile_rank'] = quartileSlopeSummary['disc_group'].map(rankMap)
    quartileSlopeSummary = quartileSlopeSummary.sort_values('quartile_rank').drop(
        columns='quartile_rank'
    )

    trendModel = sm.OLS(
        analysisWithSlope['abs_slope'], sm.add_constant(analysisWithSlope['quartile_rank'])
    ).fit()
    rho, rhoP = stats.spearmanr(analysisWithSlope['quartile_rank'], analysisWithSlope['abs_slope'])
    rhoLo, rhoHi = fisher_ci(float(rho), len(analysisWithSlope))

    globalTrendStats = pd.DataFrame(
        [
            {
                'window_cycles': wildcardWindowCycles,
                'learner_n': int(analysisWithSlope['pp'].nunique()),
                'spearman_rho': float(rho),
                'spearman_ci95_lo': rhoLo,
                'spearman_ci95_hi': rhoHi,
                'spearman_p': float(rhoP),
                'ols_slope_per_quartile': float(trendModel.params['quartile_rank']),
                'ols_slope_ci95_lo': float(trendModel.conf_int().loc['quartile_rank', 0]),
                'ols_slope_ci95_hi': float(trendModel.conf_int().loc['quartile_rank', 1]),
                'ols_p': float(trendModel.pvalues['quartile_rank']),
                'ols_r2': float(trendModel.rsquared),
            }
        ]
    )

    pairwiseRows = []
    for left, right in combinations(quartileOrder, 2):
        leftVals = (
            analysisWithSlope.loc[analysisWithSlope['disc_group'] == left, 'abs_slope']
            .dropna()
            .to_numpy(dtype=float)
        )
        rightVals = (
            analysisWithSlope.loc[analysisWithSlope['disc_group'] == right, 'abs_slope']
            .dropna()
            .to_numpy(dtype=float)
        )
        test = stats.ttest_ind(leftVals, rightVals, equal_var=False)
        diff, diffLo, diffHi, welchDf = welch_mean_diff_ci(leftVals, rightVals)
        g, gLo, gHi = hedges_g_with_ci(leftVals, rightVals)
        leftSummary = mean_ci(pd.Series(leftVals))
        rightSummary = mean_ci(pd.Series(rightVals))
        pairwiseRows.append(
            {
                'contrast': f'{left} vs {right}',
                'group_1': left,
                'group_2': right,
                'n_1': int(leftSummary['n']),
                'mean_1': float(leftSummary['mean_abs_slope']),
                'sd_1': float(leftSummary['sd_abs_slope']),
                'ci95_lo_1': float(leftSummary['ci95_lo']),
                'ci95_hi_1': float(leftSummary['ci95_hi']),
                'n_2': int(rightSummary['n']),
                'mean_2': float(rightSummary['mean_abs_slope']),
                'sd_2': float(rightSummary['sd_abs_slope']),
                'ci95_lo_2': float(rightSummary['ci95_lo']),
                'ci95_hi_2': float(rightSummary['ci95_hi']),
                'mean_diff_1_minus_2': diff,
                'mean_diff_ci95_lo': diffLo,
                'mean_diff_ci95_hi': diffHi,
                'welch_t': float(test.statistic),
                'welch_df': welchDf,
                'p_uncorrected': float(test.pvalue),
                'hedges_g': g,
                'hedges_g_ci95_lo': gLo,
                'hedges_g_ci95_hi': gHi,
            }
        )

    pairwiseContrasts = pd.DataFrame(pairwiseRows)
    reject, pHolm, _, _ = multipletests(
        pairwiseContrasts['p_uncorrected'].to_numpy(dtype=float), method='holm'
    )
    pairwiseContrasts['p_holm'] = pHolm
    pairwiseContrasts['reject_holm_0p05'] = reject

    quartileSlopeSummaryPath = (
        wildcardHeterogeneityFolder / 'wildcard_panelB_abs_slope_quartile_summary.csv'
    )
    globalTrendStatsPath = (
        wildcardHeterogeneityFolder / 'wildcard_panelB_abs_slope_global_stats.csv'
    )
    pairwiseContrastsPath = (
        wildcardHeterogeneityFolder / 'wildcard_panelB_abs_slope_pairwise_contrasts.csv'
    )

    quartileSlopeSummary.to_csv(quartileSlopeSummaryPath, index=False)
    globalTrendStats.to_csv(globalTrendStatsPath, index=False)
    pairwiseContrasts.to_csv(pairwiseContrastsPath, index=False)

    print(f'Saved {quartileSlopeSummaryPath}')
    print(f'Saved {globalTrendStatsPath}')
    print(f'Saved {pairwiseContrastsPath}')
    print()
    print(
        f"Global trend: Spearman rho = {rho:.3f} "
        f"[95% CI {rhoLo:.3f}, {rhoHi:.3f}], p = {rhoP:.3e}; "
        f"OLS slope per quartile = {trendModel.params['quartile_rank']:.3f} "
        f"[95% CI {trendModel.conf_int().loc['quartile_rank', 0]:.3f}, {trendModel.conf_int().loc['quartile_rank', 1]:.3f}], "
        f"p = {trendModel.pvalues['quartile_rank']:.3e}, R? = {trendModel.rsquared:.3f}"
    )
    print()
    print(quartileSlopeSummary.round(4))
    print(globalTrendStats.round(4))
    print(pairwiseContrasts.round(4))


def wildcard_aim_statistics():
    from itertools import combinations
    from pathlib import Path
    import importlib
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm
    import wildcard_pen12_window16_multipanel as wc_multi

    importlib.reload(wc_multi)

    wildcardWindowCycles = 16
    wildcardHeterogeneityFolder = Path('BLRFigures') / 'WildCardFigures' / 'heterogeneity_trueother'
    wildcardHeterogeneityFolder.mkdir(exist_ok=True, parents=True)

    fit, recovered = wc_multi.load_fit_and_recovery()
    panelA = wc_multi.build_panel_a_table(fit, recovered)
    learnerPP = panelA.loc[panelA['learner'], 'pp'].astype(int).tolist()
    analysisDf = wc_multi.build_analysis_table(fit, recovered, window_cycles=wildcardWindowCycles)
    analysisDf = analysisDf[analysisDf['pp'].isin(learnerPP)].copy()
    quartileDf, quartileOrder, _ = wc_multi.assign_discrimination_groups(
        analysisDf[['pp', 'disc']].dropna(),
        scheme='quartiles',
    )
    analysisDf = analysisDf.merge(quartileDf[['pp', 'disc_group']], on='pp', how='left')

    allDf = fit.df.copy().reset_index(drop=True)
    aimRows = []
    for pp in learnerPP:
        part = recovered['participants'][int(pp)]
        pdf = allDf[allDf['participantNum'] == int(pp)].copy().reset_index(drop=True)
        phase = np.asarray(part['phase'])
        if len(pdf) != len(phase):
            pdf = pdf.iloc[: len(phase)].copy().reset_index(drop=True)

        pdf['phase_rec'] = phase
        pdf['rotation_rec'] = np.asarray(part['rotation'], dtype=float)
        rotation = (
            pdf[pdf['phase_rec'].astype(str).str.lower() == 'rotation']
            .copy()
            .reset_index(drop=True)
        )
        if rotation.empty:
            continue

        rotation['rot_trial_idx'] = np.arange(len(rotation))
        rotation['cycle'] = rotation['rot_trial_idx'] // 4
        onsetSignal = (
            rotation.loc[rotation['rotation_rec'] != 0]
            .groupby('cycle')['aim']
            .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
            .sort_index()
        )
        onsetCycle = wc_multi.first_interior_breakpoint(
            onsetSignal.to_numpy(dtype=float), wc_multi.PEN
        )
        if onsetCycle is None:
            continue

        selected = rotation[
            (rotation['cycle'] >= onsetCycle)
            & (rotation['cycle'] < onsetCycle + wildcardWindowCycles)
        ].copy()
        if selected.empty:
            continue

        meanAbsAimInc0 = float(np.nanmean(np.abs(selected['aim'].to_numpy(dtype=float))))
        aimRows.append({'pp': int(pp), 'mean_abs_aim_inc0': meanAbsAimInc0})

    aimDf = pd.DataFrame(aimRows)
    analysisDf = analysisDf.merge(aimDf, on='pp', how='inner')
    rankMap = {group: idx + 1 for idx, group in enumerate(quartileOrder)}
    analysisDf['quartile_rank'] = analysisDf['disc_group'].map(rankMap)

    def mean_ci(series: pd.Series) -> pd.Series:
        vals = series.dropna().to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
        tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan
        ci_half = float(tcrit * sem) if n > 1 else np.nan
        return pd.Series(
            {
                'n': n,
                'mean_abs_aim_inc0': mean,
                'sd_abs_aim_inc0': sd,
                'sem_abs_aim_inc0': sem,
                'ci95_lo': mean - ci_half if n > 1 else np.nan,
                'ci95_hi': mean + ci_half if n > 1 else np.nan,
                'median_abs_aim_inc0': float(np.median(vals)) if n else np.nan,
                'min_abs_aim_inc0': float(np.min(vals)) if n else np.nan,
                'max_abs_aim_inc0': float(np.max(vals)) if n else np.nan,
            }
        )

    def fisher_ci(r: float, n: int) -> tuple[float, float]:
        if (not np.isfinite(r)) or n <= 3 or abs(r) >= 1:
            return (np.nan, np.nan)
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))

    def hedges_g_with_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return (np.nan, np.nan, np.nan)
        s1 = np.std(x, ddof=1)
        s2 = np.std(y, ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if (not np.isfinite(pooled)) or pooled == 0:
            return (np.nan, np.nan, np.nan)
        d = (np.mean(x) - np.mean(y)) / pooled
        j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
        g = j * d
        se_g = np.sqrt((n1 + n2) / (n1 * n2) + (g**2) / (2.0 * (n1 + n2 - 2)))
        return (float(g), float(g - 1.96 * se_g), float(g + 1.96 * se_g))

    def welch_mean_diff_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n1, n2 = len(x), len(y)
        m1, m2 = np.mean(x), np.mean(y)
        s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
        v1 = s1**2 / n1
        v2 = s2**2 / n2
        se = np.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / ((v1**2) / (n1 - 1) + (v2**2) / (n2 - 1))
        tcrit = stats.t.ppf(0.975, df)
        diff = m1 - m2
        return (float(diff), float(diff - tcrit * se), float(diff + tcrit * se), float(df))

    quartileAimSummary = (
        analysisDf.groupby('disc_group')['mean_abs_aim_inc0'].apply(mean_ci).unstack().reset_index()
    )
    quartileAimSummary['quartile_rank'] = quartileAimSummary['disc_group'].map(rankMap)
    quartileAimSummary = quartileAimSummary.sort_values('quartile_rank').drop(
        columns='quartile_rank'
    )

    trendModel = sm.OLS(
        analysisDf['mean_abs_aim_inc0'], sm.add_constant(analysisDf['quartile_rank'])
    ).fit()
    rho, rhoP = stats.spearmanr(analysisDf['quartile_rank'], analysisDf['mean_abs_aim_inc0'])
    rhoLo, rhoHi = fisher_ci(float(rho), len(analysisDf))

    globalTrendStats = pd.DataFrame(
        [
            {
                'window_cycles': wildcardWindowCycles,
                'learner_n': int(analysisDf['pp'].nunique()),
                'spearman_rho': float(rho),
                'spearman_ci95_lo': rhoLo,
                'spearman_ci95_hi': rhoHi,
                'spearman_p': float(rhoP),
                'ols_slope_per_quartile': float(trendModel.params['quartile_rank']),
                'ols_slope_ci95_lo': float(trendModel.conf_int().loc['quartile_rank', 0]),
                'ols_slope_ci95_hi': float(trendModel.conf_int().loc['quartile_rank', 1]),
                'ols_p': float(trendModel.pvalues['quartile_rank']),
                'ols_r2': float(trendModel.rsquared),
            }
        ]
    )

    pairwiseRows = []
    for left, right in combinations(quartileOrder, 2):
        leftVals = (
            analysisDf.loc[analysisDf['disc_group'] == left, 'mean_abs_aim_inc0']
            .dropna()
            .to_numpy(dtype=float)
        )
        rightVals = (
            analysisDf.loc[analysisDf['disc_group'] == right, 'mean_abs_aim_inc0']
            .dropna()
            .to_numpy(dtype=float)
        )
        test = stats.ttest_ind(leftVals, rightVals, equal_var=False)
        diff, diffLo, diffHi, welchDf = welch_mean_diff_ci(leftVals, rightVals)
        g, gLo, gHi = hedges_g_with_ci(leftVals, rightVals)
        leftSummary = mean_ci(pd.Series(leftVals))
        rightSummary = mean_ci(pd.Series(rightVals))
        pairwiseRows.append(
            {
                'contrast': f'{left} vs {right}',
                'group_1': left,
                'group_2': right,
                'n_1': int(leftSummary['n']),
                'mean_1': float(leftSummary['mean_abs_aim_inc0']),
                'sd_1': float(leftSummary['sd_abs_aim_inc0']),
                'ci95_lo_1': float(leftSummary['ci95_lo']),
                'ci95_hi_1': float(leftSummary['ci95_hi']),
                'n_2': int(rightSummary['n']),
                'mean_2': float(rightSummary['mean_abs_aim_inc0']),
                'sd_2': float(rightSummary['sd_abs_aim_inc0']),
                'ci95_lo_2': float(rightSummary['ci95_lo']),
                'ci95_hi_2': float(rightSummary['ci95_hi']),
                'mean_diff_1_minus_2': diff,
                'mean_diff_ci95_lo': diffLo,
                'mean_diff_ci95_hi': diffHi,
                'welch_t': float(test.statistic),
                'welch_df': welchDf,
                'p_uncorrected': float(test.pvalue),
                'hedges_g': g,
                'hedges_g_ci95_lo': gLo,
                'hedges_g_ci95_hi': gHi,
            }
        )

    pairwiseContrasts = pd.DataFrame(pairwiseRows)
    reject, pHolm, _, _ = multipletests(
        pairwiseContrasts['p_uncorrected'].to_numpy(dtype=float), method='holm'
    )
    pairwiseContrasts['p_holm'] = pHolm
    pairwiseContrasts['reject_holm_0p05'] = reject

    quartileAimSummaryPath = (
        wildcardHeterogeneityFolder / 'wildcard_meanabsaim_inc0_quartile_summary.csv'
    )
    globalTrendStatsPath = wildcardHeterogeneityFolder / 'wildcard_meanabsaim_inc0_global_stats.csv'
    pairwiseContrastsPath = (
        wildcardHeterogeneityFolder / 'wildcard_meanabsaim_inc0_pairwise_contrasts.csv'
    )

    quartileAimSummary.to_csv(quartileAimSummaryPath, index=False)
    globalTrendStats.to_csv(globalTrendStatsPath, index=False)
    pairwiseContrasts.to_csv(pairwiseContrastsPath, index=False)

    print(f'Saved {quartileAimSummaryPath}')
    print(f'Saved {globalTrendStatsPath}')
    print(f'Saved {pairwiseContrastsPath}')
    print()
    print(
        f"Global trend: Spearman rho = {rho:.3f} "
        f"[95% CI {rhoLo:.3f}, {rhoHi:.3f}], p = {rhoP:.3e}; "
        f"OLS slope per quartile = {trendModel.params['quartile_rank']:.3f} "
        f"[95% CI {trendModel.conf_int().loc['quartile_rank', 0]:.3f}, {trendModel.conf_int().loc['quartile_rank', 1]:.3f}], "
        f"p = {trendModel.pvalues['quartile_rank']:.3e}, R? = {trendModel.rsquared:.3f}"
    )
    print()
    print(quartileAimSummary.round(4))
    print(globalTrendStats.round(4))
    print(pairwiseContrasts.round(4))
