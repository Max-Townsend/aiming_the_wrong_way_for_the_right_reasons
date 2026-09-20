"""Prepare the Chapter 3 datasets and print the participant summary."""


def preprocess():
    import numpy as np
    import pandas as pd
    from convert_exp1 import convert_exp1
    from reformat_wildcard import reformat_wildcard

    df = pd.read_csv('2025-12_Ding_Data.csv')
    # Wrap hand-to-target differences to [-180, 180) degrees.
    df['aim_signed'] = (df['Hand Angle'] - df['Target Angle'] + 180) % 360 - 180
    df['aim_abs'] = np.abs(df['aim_signed'])
    df = df.rename(
        columns={
            'participant_idx': 'participantNum',
            'Rotation': 'rotation',
            'Target Angle': 'targetPosition',
        }
    )
    df['blockNum'] = 0
    df['aim'] = df['aim_signed']
    df.to_csv('ding_modified.csv', index=False)
    convert_exp1('Exp1.mat', 'Exp1.csv')
    reformat_wildcard('wildCardTask.csv').to_csv('wildcard_long.csv', index=False)
    print('Prepared Ding, savings, and wildcard data.')


def demographics():
    import pandas as pd
    from wildcard_demographics import print_summary

    demographics = pd.read_csv('wildcard_demographics.csv')
    assert set(demographics.participantNum) == set(pd.read_csv('wildCardTask.csv').participantNum)
    print_summary(demographics)
