"""Exercise all six production fitting entry points with a bounded test budget."""

from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import prepare, workspace, TASKS


class SerialPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starmap(self, fn, items):
        return [fn(*args) for args in items]

    def map(self, fn, items):
        return [fn(x) for x in items]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='reproduction')
    parser.add_argument('--run-name', default='fit-smoke')
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    dest = workspace(args.workspace)
    if not (dest / '.prepared.json').exists():
        dest = prepare(args.workspace)
    if not args.run_name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid run name')
    out = dest / args.run_name
    if out.exists():
        raise ValueError('Choose a new --run-name to avoid reusing prior smoke fits')
    out.mkdir()
    sys.path.insert(0, str(dest))
    os.environ['MPLBACKEND'] = 'Agg'
    import multiprocessing
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import convert_exp1
    import reformat_wildcard
    from chapter3 import fitting

    frames = {
        'ding': pd.read_csv(dest / 'ding_modified.csv'),
        'savings': pd.read_csv(dest / 'Exp1.csv'),
        'wildcard': reformat_wildcard.reformat_wildcard(dest / 'wildCardTask.csv'),
    }

    def fit_shells(value):
        if isinstance(value, dict):
            return [shell for v in value.values() for shell in fit_shells(v)]
        if isinstance(value, np.ndarray):
            return [shell for v in value.flat for shell in fit_shells(v)]
        return [value]

    records = []
    real_reload = importlib.reload
    real_read_csv = pd.read_csv
    for dataset, frame in frames.items():
        participant = frame.participantNum.iloc[0]
        data = frame.loc[frame.participantNum == participant].copy().reset_index(drop=True)
        perturbed = np.flatnonzero(data.rotation.fillna(0).to_numpy() != 0)
        data = data.iloc[: int(perturbed[0]) + 12].copy()
        for family in ('shift', 'qhmm'):
            task = f'fit-{dataset}-{family}'
            case = out / task
            case.mkdir()
            (case / 'tempFigures').mkdir()
            os.chdir(case)
            module_name = (
                'HMM'
                if family == 'qhmm'
                else ('dingBLR' if dataset == 'ding' else 'BayesHypothesisTesting')
            )
            module = importlib.import_module(module_name)
            real_optimizer = module._run_cma_restarts
            calls = []

            def bounded_optimizer(objective, **kwargs):
                kwargs.update(nRestarts=1, nPolish=0, maxfevals_per_run=24)
                calls.append(dict(kwargs))
                return real_optimizer(objective, **kwargs)

            def test_reload(candidate):
                # Keep the optimizer budget patched when a fitting task reloads its model.
                if candidate.__name__ in {'BayesHypothesisTesting', 'dingBLR', 'HMM'}:
                    return candidate
                return real_reload(candidate)

            def test_read_csv(path, *a, **kw):
                if Path(path).name == 'ding_modified.csv':
                    return data.copy()
                return real_read_csv(path, *a, **kw)

            output_name = {
                ('ding', 'shift'): 'dingBLRDing.npy',
                ('ding', 'qhmm'): 'HMMDing.npy',
                ('savings', 'shift'): 'BHTAvrahamSavings.npy',
                ('savings', 'qhmm'): 'HMMSavings.npy',
                ('wildcard', 'shift'): 'BHTWildcard.npy',
                ('wildcard', 'qhmm'): 'HMMWildcard.npy',
            }[dataset, family]
            function = getattr(fitting, TASKS[task][1])
            started = time.monotonic()
            with (
                patch.dict(os.environ, {'SHIFT_CORES': '1'}),
                patch.object(importlib, 'reload', test_reload),
                patch.object(module, '_run_cma_restarts', bounded_optimizer),
                patch.object(multiprocessing, 'Pool', SerialPool),
                patch.object(pd, 'read_csv', test_read_csv),
                patch.object(convert_exp1, 'convert_exp1', side_effect=lambda *a, **k: data.copy()),
                patch.object(
                    reformat_wildcard, 'reformat_wildcard', side_effect=lambda *a, **k: data.copy()
                ),
            ):
                function()
                assert len(calls) == 1, 'The entry point must perform a fresh optimization'
                shells = fit_shells(np.load(output_name, allow_pickle=True))
                assert len(shells) == 1 and len(shells[0].participantNums) == 1
                likelihoods = [shell.negLl.copy() for shell in shells]
                for shell in shells:
                    assert np.isfinite(shell.negLl).all() and np.max(shell.negLl) < 1e11
                    assert np.isfinite(np.asarray(shell.xs)).all()
                function()
                assert len(calls) == 1, 'Checkpoint resume unexpectedly refitted'
                resumed = fit_shells(np.load(output_name, allow_pickle=True))
                assert len(resumed) == len(likelihoods)
                for before, shell in zip(likelihoods, resumed):
                    np.testing.assert_allclose(before, shell.negLl, rtol=1e-10, atol=1e-8)
            records.append(
                {
                    'task': task,
                    'entry_point': f'chapter3.fitting.{function.__name__}',
                    'trials': len(data),
                    'seconds': round(time.monotonic() - started, 2),
                    'optimizer_runs': 1,
                    'max_evaluations_requested': 24,
                    'serialization_passed': True,
                    'checkpoint_resume_passed': True,
                }
            )
            (out / 'report.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            plt.close('all')
            print('PASS', task, flush=True)
    print(f'All six production fitting entry points passed. Report: {out / "report.json"}')


if __name__ == '__main__':
    main()
