"""Run Chapter 3 analyses in a workspace under outputs/."""

from pathlib import Path
import argparse
import importlib
import json
import os
import shutil
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parent
TASKS = {
    'figure31': ('schematics', 'run', {}),
    'figure32': ('figures', 'figure32', {}),
    'figure33': ('figures', 'figure33', {}),
    'figure34': ('figures', 'figure34', {}),
    'check-figure32': ('figures', 'figure32', {'quick': True}),
    'check-figure34': ('figures', 'figure34', {'quick': True}),
    'geometry': ('figures', 'geometry', {}),
    'statistics': ('statistics', 'core', {}),
    'manuscript-statistics': ('statistics', 'manuscript', {}),
    'preprocess': ('data', 'preprocess', {}),
    'demographics': ('data', 'demographics', {}),
    'recover-weights': ('fitting', 'recover_weights', {}),
}
for dataset in ('ding', 'savings', 'wildcard'):
    for model in ('shift', 'qhmm'):
        TASKS[f'fit-{dataset}-{model}'] = ('fitting', f'fit_{dataset}_{model}', {})
RAW_DATA = {
    'data/raw/2025-12_Ding_Data.csv': '2025-12_Ding_Data.csv',
    'data/raw/Exp1.mat': 'Exp1.mat',
    'data/raw/wildCardTask.csv': 'wildCardTask.csv',
}
PROCESSED_DATA = {f'data/processed/{name}': name for name in ('Exp1.csv', 'ding_modified.csv')}
SAVED_FITS = {
    f'data/fits/{name}': name
    for name in (
        'dingBLRDing.npy',
        'HMMDing.npy',
        'BHTAvrahamSavings.npy',
        'HMMSavings.npy',
        'BHTWildcard.npy',
        'HMMWildcard.npy',
        'BHTWildcard_true_self_other.pkl',
    )
}
DEMOGRAPHICS = {
    'data/processed/wildcard_demographics.csv': 'wildcard_demographics.csv',
}


def workspace_files(profile='core'):
    """Map repository files to the paths used by the analysis modules."""
    if profile not in {'core', 'raw'}:
        raise ValueError('Profile must be core or raw')
    files = {**RAW_DATA, **DEMOGRAPHICS}
    if profile == 'core':
        files.update(PROCESSED_DATA)
        files.update(SAVED_FITS)
    # Flat module names are also stored in the serialized model fits.
    for folder in ('models', 'preprocessing', 'analysis'):
        for source in sorted((ROOT / 'src' / folder).glob('*.py')):
            files[source.relative_to(ROOT).as_posix()] = source.name
    for source in sorted((ROOT / 'src/chapter3').glob('*.py')):
        files[source.relative_to(ROOT).as_posix()] = f'chapter3/{source.name}'
    return files


def check_files(paths):
    """Check that inputs exist and Git LFS has downloaded their contents."""
    failures = []
    for path in paths:
        if not path.is_file():
            failures.append(f'Missing: {path}')
            continue
        with path.open('rb') as handle:
            if handle.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                failures.append(f'Git LFS content missing: {path}; run git lfs pull')
    if failures:
        raise FileNotFoundError('\n'.join(failures))


def workspace(name):
    if not name or any(
        c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name
    ):
        raise ValueError(
            'Workspace name must contain only letters, numbers, hyphens, or underscores'
        )
    return ROOT / 'outputs' / name


def prepare(name, profile='core'):
    files = workspace_files(profile)
    dest = workspace(name)
    marker = dest / '.prepared.json'
    if marker.exists():
        prior = json.loads(marker.read_text(encoding='utf-8'))['profile']
        if prior != profile:
            raise ValueError(
                f'{name} was prepared as {prior}; choose a new workspace for {profile}'
            )
        print(f'Using existing workspace: {dest}')
        return dest
    if dest.exists():
        raise ValueError(f'Incomplete/existing workspace: {dest}. Use a new name.')
    selected_files = [(ROOT / source, dest / target) for source, target in files.items()]
    check_files(source for source, _ in selected_files)
    dest.mkdir(parents=True)
    for src, target in selected_files:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copies keep fitting and preprocessing writes inside the workspace.
        shutil.copyfile(src, target)
    count = len(selected_files)
    marker.write_text(
        json.dumps({'profile': profile, 'files_copied': count}, indent=2),
        encoding='utf-8',
    )
    print(f'Prepared {dest}: {count} independent file copies ({profile})')
    return dest


def run(task, name):
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    dest = workspace(name)
    if not (dest / '.prepared.json').exists():
        dest = prepare(name)
    # Resolve both working-directory and module-relative paths in the workspace.
    os.chdir(dest)
    sys.path.insert(0, str(dest))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('NUMBA_NUM_THREADS', '4')
    import matplotlib.pyplot as plt

    module_name, function_name, options = TASKS[task]
    log = {
        'task': task,
        'workspace': name,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'entry_point': f'chapter3.{module_name}.{function_name}',
        'options': options,
    }
    log_dir = dest / 'run_logs'
    log_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        module = importlib.import_module(f'chapter3.{module_name}')
        print(f'[{task}] {log["entry_point"]}', flush=True)
        getattr(module, function_name)(**options)
        log['success'] = True
    except BaseException as exc:
        log.update(success=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        plt.close('all')
        log['seconds'] = round(time.monotonic() - started, 3)
        (log_dir / f'{task}.json').write_text(json.dumps(log, indent=2), encoding='utf-8')


def verify():
    paths = [ROOT / source for source in workspace_files()]
    paths.extend([ROOT / 'reproduce.py', ROOT / 'scripts/smoke_test.py'])
    check_files(paths)
    for path in paths:
        if path.suffix == '.py':
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print(f'Checked {len(paths)} files: required inputs are present and Python sources compile.')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List available tasks')
    p = sub.add_parser('prepare', help='Copy code and inputs into a workspace')
    p.add_argument('--profile', choices=['core', 'raw'], default='core')
    p.add_argument('--workspace', default='reproduction')
    p = sub.add_parser('run', help='Run one task')
    p.add_argument('task', choices=sorted(TASKS))
    p.add_argument('--workspace', default='reproduction')
    sub.add_parser('verify', help='Check required inputs and Python syntax')
    p = sub.add_parser('all', help='Run all figures and statistics')
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--workspace', default='reproduction')
    args = parser.parse_args()
    if args.command == 'list':
        for task in sorted(TASKS):
            print(task)
    elif args.command == 'prepare':
        prepare(args.workspace, args.profile)
    elif args.command == 'run':
        run(args.task, args.workspace)
    elif args.command == 'verify':
        verify()
    else:
        if args.from_scratch and workspace(args.workspace).exists():
            raise ValueError('A from-scratch run requires a new workspace name.')
        prepare(args.workspace, 'raw' if args.from_scratch else 'core')
        tasks = []
        if args.from_scratch:
            tasks = [
                'preprocess',
                'fit-ding-shift',
                'fit-ding-qhmm',
                'fit-savings-shift',
                'fit-savings-qhmm',
                'fit-wildcard-shift',
                'fit-wildcard-qhmm',
                'recover-weights',
            ]
        tasks += [
            'figure31',
            'figure32',
            'figure33',
            'figure34',
            'statistics',
            'manuscript-statistics',
            'demographics',
        ]
        for task in tasks:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / 'reproduce.py'),
                    'run',
                    task,
                    '--workspace',
                    args.workspace,
                ],
                check=True,
            )


if __name__ == '__main__':
    main()
