"""Fit SHIFT and Q-HMM to the Ding, savings and wildcard datasets."""

import importlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import BayesHypothesisTesting as BHT
import HMM


def _setup():
    np.random.seed(42)
    Path('tempFigures').mkdir(exist_ok=True)


def fit_ding_shift():
    _setup()
    import dingBLR
    import numpy as np
    import os

    importlib.reload(dingBLR)
    targetFolder = 'ding_plots/'
    os.makedirs(targetFolder, exist_ok=True)

    numCores = int(os.environ.get('SHIFT_CORES', '4'))

    df = pd.read_csv('ding_modified.csv')
    uniqueConditions = np.sort(df['condition'].unique())
    print(f"Found conditions: {uniqueConditions}")
    print(f"\n=== Fitting pooled Ding dataset: {df['participantNum'].nunique()} participants ===")
    pooledFit = dingBLR.FitShell(
        df=df,
        condition='none',
        conVal='none',
        plotIdentifier=f'{targetFolder}dingBLRCGDing_',
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='Ding',
        n_harm=1,
        maxrun=2,
        make_quick_plots=False,
    )
    pooledFit.fitRot(numCores)

    def _subset_ding_fitshell(fit, condStr):
        condDf = fit.df[fit.df['condition'] == condStr].copy()
        pp_subset = [
            pp
            for pp in fit.participantNums
            if fit.participantInfo.get(pp, {}).get('condition') == condStr
        ]
        pp_to_idx = {pp: i for i, pp in enumerate(fit.participantNums)}
        idx = [pp_to_idx[pp] for pp in pp_subset]

        sub = dingBLR.FitShell(
            df=condDf,
            condition='rotation',
            conVal='none',
            plotIdentifier=f'{targetFolder}dingBLRCGDing_{condStr}_',
            numCores=fit.numCores,
            popSizeMultiplier=fit.popSizeMultiplier,
            datasetName=f'Ding_{condStr}',
            fitWashout=fit.fitWashout,
        )
        sub.dat = condDf
        sub.heightCap = fit.heightCap
        sub._hyper_overrides = fit._hyper_overrides.copy()
        sub.participantNums = np.array(pp_subset)
        sub.bics = np.array([fit.bics[i] for i in idx])
        sub.rmses = np.array([fit.rmses[i] for i in idx])
        sub.rSquareds = np.array([fit.rSquareds[i] for i in idx])
        sub.negLl = np.array([fit.negLl[i] for i in idx])
        sub.mStates = [fit.mStates[i] for i in idx]
        sub.allAims = [fit.allAims[i] for i in idx]
        sub.xs = [fit.xs[i] for i in idx]
        sub.q0s = [fit.q0s[i] for i in idx]
        sub.predEntropyAll = [fit.predEntropyAll[i] for i in idx]
        sub.participantInfo = {pp: fit.participantInfo[pp] for pp in pp_subset}
        sub.trialStatuses = {pp: fit.trialStatuses[pp] for pp in pp_subset}
        sub.genTargets = {pp: fit.genTargets[pp] for pp in pp_subset}
        sub.trainTargets = {pp: fit.trainTargets[pp] for pp in pp_subset}
        sub.destTargets = {pp: fit.destTargets[pp] for pp in pp_subset}
        sub.predState = {pp: fit.predState[pp] for pp in pp_subset}
        return sub

    dingBLRs = {}
    for condStr in uniqueConditions:
        dingFit = _subset_ding_fitshell(pooledFit, condStr)
        dingBLRs[condStr] = dingFit
        inner_outer, numTargets, rotMag = dingBLR.parseConditionString(condStr)
        print(
            f"Done {condStr}: {inner_outer}, {numTargets}T, {rotMag}°, "
            f"mean BIC = {np.mean(dingFit.bics):.1f}"
        )
        print(f"  BICs: {dingFit.bics}")
        print(f"  Participant info sample: {list(dingFit.participantInfo.values())[:2]}")

    np.save(file='dingBLRDing.npy', arr=dingBLRs, allow_pickle=True)

    print("\n=== Summary ===")
    for condStr, dingFit in dingBLRs.items():
        nPp = len(dingFit.bics)
        nGen = sum(1 for pp in dingFit.genTargets if len(dingFit.genTargets[pp]) > 0)
        print(
            f"{condStr}: {nPp} pp, mean BIC={np.mean(dingFit.bics):.1f}, "
            f"mean negLL={np.mean(dingFit.negLl):.1f}, "
            f"{nGen} pp with generalisation targets"
        )


def fit_ding_qhmm():
    _setup()
    importlib.reload(HMM)

    tDat = pd.read_csv('ding_modified.csv')

    numCores = int(os.environ.get('SHIFT_CORES', '4'))

    hmm_ding = HMM.FitShell(
        df=tDat,
        condition='none',
        conVal='none',
        fitPhase=None,
        annealing_mode=0,
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='HMMDing',
        fitWashout=False,
    )
    hmm_ding.fitRot(numCores=numCores)

    print('done Ding', np.mean(hmm_ding.bics))
    print(hmm_ding.bics.tolist())

    np.save('HMMDing.npy', hmm_ding, allow_pickle=True)


def fit_savings_shift():
    _setup()
    from convert_exp1 import convert_exp1

    importlib.reload(BHT)
    df = convert_exp1('Exp1.mat', 'Exp1.csv')
    numCores = int(os.environ.get('SHIFT_CORES', '4'))
    bht = BHT.FitShell(
        df=df,
        condition='none',
        conVal='none',
        fitPhase='rotation',
        plotIdentifier='tempFigures/BHTExp1_All',
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='Exp1_All',
        fitWashout=True,
        n_harm=1,
        maxrun=2,
        make_quick_plots=False,
    )
    bht.fitRot(numCores=numCores)
    np.save(arr=[bht], file='BHTAvrahamSavings')
    print(bht.bics)


def fit_savings_qhmm():
    _setup()
    from convert_exp1 import convert_exp1

    importlib.reload(HMM)
    df = convert_exp1('Exp1.mat', 'Exp1.csv')

    df_savings = df.copy()
    df_savings['blockNum'] = 0  # collapse to one block so _fitWorker uses all trials
    numCores = int(os.environ.get('SHIFT_CORES', '4'))
    hmm_savings = HMM.FitShell(
        df=df_savings,
        condition='none',
        conVal='none',
        fitPhase=None,
        annealing_mode=0,
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='HMMSavings',
    )
    hmm_savings.fitRot()

    print('done savings', np.mean(hmm_savings.bics))
    print(hmm_savings.bics.tolist())
    np.save('HMMSavings.npy', hmm_savings)


def fit_wildcard_shift():
    _setup()
    import importlib
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from reformat_wildcard import reformat_wildcard

    df = reformat_wildcard('wildCardTask.csv')

    importlib.reload(BHT)
    folderPath = Path("tempFigures")
    folderPath.mkdir(exist_ok=True)
    targetFolder = 'tempFigures/'

    numCores = int(os.environ.get('SHIFT_CORES', '4'))
    trialsPerParticipant = 360

    bht = BHT.FitShell(
        df=df,
        condition='none',  # no filtering — every participant has mixed rotations
        conVal='none',
        fitPhase='rotation',
        plotIdentifier=targetFolder + 'BHTCGWildcard',
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='Wildcard',
        fitWashout=True,
        n_harm=1,
        maxrun=2,
        make_quick_plots=False,
    )
    bht.fitRot(numCores)

    print('done', np.mean(bht.bics))
    print(bht.bics)
    np.save(arr=[bht], file='BHTWildcard')


def fit_wildcard_qhmm():
    _setup()
    from reformat_wildcard import reformat_wildcard

    importlib.reload(HMM)
    df = reformat_wildcard('wildCardTask.csv')
    numCores = int(os.environ.get('SHIFT_CORES', '4'))
    hmm_wildcard = HMM.FitShell(
        df=df,
        condition='none',
        conVal='none',
        fitPhase=None,
        annealing_mode=0,
        numCores=numCores,
        popSizeMultiplier=1,
        datasetName='HMMWildcard',
    )
    hmm_wildcard.fitRot()

    print('done wildcard', np.mean(hmm_wildcard.bics))
    print(hmm_wildcard.bics.tolist())
    np.save('HMMWildcard.npy', hmm_wildcard)


def recover_weights():
    from pathlib import Path
    from recover_true_self_other_weights import save_recovered_weights

    save_recovered_weights(Path('BHTWildcard.npy'), Path('BHTWildcard_true_self_other.pkl'))
