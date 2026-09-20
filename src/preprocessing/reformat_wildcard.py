"""Convert array-valued wildcard participant records to trial-level data.

Output columns include participantNum, rotation, aim, targetPosition,
Cursor FB, condition and blockNum.
"""

import pandas as pd
import numpy as np
import json


def parse_array_col(s):
    """Parse a stringified list that may contain 'nan' values."""
    return json.loads(s.replace('nan', 'null'))


def reformat_wildcard(input_path, output_path=None):
    df_wide = pd.read_csv(input_path)

    rows = []
    for _, row in df_wide.iterrows():
        pp = int(row['participantNum'])
        rotations = parse_array_col(row['rotation'])
        aims = parse_array_col(row['aim'])
        targets = parse_array_col(row['targetPosition'])

        n_trials = len(rotations)
        assert len(aims) == n_trials and len(targets) == n_trials, (
            f"Participant {pp}: mismatched array lengths"
        )

        for t in range(n_trials):
            rows.append(
                {
                    'participantNum': pp,
                    'rotation': rotations[t] if rotations[t] is not None else np.nan,
                    'aim': aims[t] if aims[t] is not None else np.nan,
                    'targetPosition': targets[t] if targets[t] is not None else np.nan,
                    'Cursor FB': 'cursor',  # all trials have feedback
                    'condition': 'inner_4T_0',  # 4 targets, metadata only
                    'blockNum': 0,
                }
            )

    df_long = pd.DataFrame(rows)

    # Replace rotation NaNs with 0 (missed trials — no perturbation applied)
    df_long['rotation'] = df_long['rotation'].fillna(0.0)

    print(f"Reformatted: {len(df_wide)} participants × {n_trials} trials → {len(df_long)} rows")
    print(f"Unique targets: {sorted(df_long['targetPosition'].dropna().unique())}")
    print(
        f"Unique non-zero rotations: "
        f"{sorted(df_long.loc[df_long['rotation'] != 0, 'rotation'].unique())}"
    )

    if output_path is not None:
        df_long.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df_long


if __name__ == '__main__':
    df = reformat_wildcard('wildCardTask.csv', 'wildCardTask_long.csv')
    print(df.head(10))
    print(df.dtypes)
