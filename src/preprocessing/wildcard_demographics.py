"""Summarise the linked wildcard participant demographics."""


def print_summary(df):
    """Print participant counts, age, sex and country summaries."""
    n_total = len(df)
    n_with_demog = df["age"].notna().sum()
    n_no_demog = n_total - n_with_demog

    print("\n" + "=" * 60)
    print(f"WILDCARD EXPERIMENT — DEMOGRAPHIC SUMMARY")
    print(f"=" * 60)
    print(f"Total participants in wildCardTask.csv: {n_total}")
    print(f"Participants with linked demographics:  {n_with_demog}")
    if n_no_demog > 0:
        missing = df[df["age"].isna()]["participantNum"].tolist()
        print(f"  (demographics unavailable for participantNums: {missing})")

    print(f"\n--- Sex / Gender ---")
    sex_counts = df["sex"].value_counts()
    for sex_label, count in sex_counts.items():
        print(f"  {sex_label}: {count}")
    n_reported = sex_counts.sum()
    n_na = n_total - n_reported
    if n_na > 0:
        print(f"  N/A (demographics unavailable): {n_na}")

    ages = df["age"].dropna()
    print(f"\n--- Age ---")
    print(f"  N:      {len(ages)}  (excludes {n_no_demog} with unavailable demographics)")
    print(f"  Mean:   {ages.mean():.1f}")
    print(f"  SD:     {ages.std():.1f}")
    print(f"  Median: {ages.median():.1f}")
    print(f"  Range:  {ages.min():.0f} – {ages.max():.0f}")

    countries = df["country"].dropna()
    print(f"\n--- Country ---")
    country_counts = countries.value_counts()
    for country, count in country_counts.head(10).items():
        print(f"  {country}: {count}")
    if len(country_counts) > 10:
        print(f"  ... and {len(country_counts) - 10} more countries")
    if n_no_demog > 0:
        print(f"  N/A: {n_no_demog}")

    print(f"\n--- Handedness ---")
    print(f"  Not recorded in this study's questionnaires.")

    male_n = sex_counts.get("Male", 0)
    female_n = sex_counts.get("Female", 0)
    other_n = n_total - male_n - female_n
    other_parts = []
    pnts = sex_counts.get("Prefer not to say", 0)
    nb = sex_counts.get("Non-binary", 0)
    if pnts + nb > 0:
        other_parts.append(f"{pnts + nb} preferred not to say or identified as non-binary")
    if n_no_demog > 0:
        other_parts.append(f"{n_no_demog} with demographics unavailable")
    other_str = ""
    if other_parts:
        other_str = ", " + ", ".join(other_parts)

    print(f"\n--- Participant summary ---")
    print(
        f"  {n_total} participants ({female_n} female, {male_n} male{other_str}; "
        f"age M = {ages.mean():.1f}, SD = {ages.std():.1f}, "
        f"range {ages.min():.0f}–{ages.max():.0f}) completed the wildcard experiment."
    )
    print("=" * 60)
