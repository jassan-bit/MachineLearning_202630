"""Finish local evidence after every candidate is frozen; never publish.

The primary matrix is run separately with run_final_regression.py. This command
then evaluates unchanged targets, writes explanations, performs serial timing
experiments, validates artifacts, and creates the separate final delivery.
Completed statistical/explanation stages can be resumed without retraining.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/final_regression_20261018'


def read(path):
    """Read a saved local completion marker."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_stage(out):
    """Join the two frozen export periods and evaluate the predeclared family."""
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from final_regression_evaluation import evaluate, compare_outer_folds
    plan = read(out / 'inference_plan.json')
    destination = out / 'evaluation'
    if not (destination / 'summary.json').exists():
        frame = pd.concat([pd.read_csv(out / f'predictions_{year}.csv.gz') for year in (2025, 2026)],
                          ignore_index=True)
        with threadpool_limits(limits=1):
            evaluate(frame, destination, reference_label=plan['reference'], primary_year=2026,
                candidate_labels=plan['primary_candidates'], bootstrap_reps=2000,
                mcs_reps=2000, block_length=34, seed=42, full_expensive_diagnostics=True)
        del frame
    if not (out / 'outer_comparison/summary.json').exists():
        with threadpool_limits(limits=1):
            compare_outer_folds(pd.read_csv(out / 'outer_metrics.csv'), out / 'outer_comparison')


def completed_stage(out, name):
    """Skip only a complete stage with a recorded report or verification."""
    if name == 'evaluation':
        return (out / 'evaluation/summary.json').exists() and (out / 'outer_comparison/summary.json').exists()
    if name == 'global_shap':
        path = out / 'explain_global/summary.json'
        return path.exists() and read(path).get('status') == 'complete'
    if name == 'best_shap':
        path = out / 'explain_best/summary.json'
        return path.exists() and read(path).get('status') == 'complete'
    if name in ('explain', 'plots', 'efficiency'):
        return (out / f'postprocess/verification_{name}.json').exists()
    if name == 'sensitivity':
        path = out / 'sensitivity/summary.json'
        return path.exists() and read(path).get('status') == 'complete'
    return False


def finish(out):
    """Run independent analyses concurrently, then isolate timing experiments."""
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results':
        raise ValueError('Outputs must remain in a study directory under workspace/results.')
    if read(out / 'status.json').get('status') != 'complete':
        raise ValueError('Finish the primary training and forecast export before this command.')
    if len(read(out / 'selection_freeze.json')['model_sha256']) != 512:
        raise ValueError('All 512 candidates must be frozen before statistical evaluation.')
    logdir = out / 'stage_logs'
    logdir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env['PYTHONUTF8'] = '1'
    started = time.perf_counter()
    timings = []

    def run(name, script, extra=(), skip=True):
        if skip and completed_stage(out, name):
            print('Verified completion marker; retaining stage: ' + name, flush=True)
            return dict(stage=name, retained=True)
        args = [sys.executable, '-u', str(ROOT / 'src' / script), '--out', str(out), *extra]
        print('Starting local stage: ' + name, flush=True)
        begin = time.perf_counter()
        with (logdir / (name + '.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'Local stage continues: {name}, {time.perf_counter()-begin:.0f} seconds', flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} failed; inspect {logdir / (name + ".log")}')
        result = dict(stage=name, retained=False, seconds=time.perf_counter()-begin)
        print('Completed local stage: ' + name, flush=True)
        return result

    independent = [('evaluation', 'finish_final_regression.py', ('--evaluate-only',)),
                   ('global_shap', 'final_regression_global_shap.py', ()),
                   ('explain', 'final_regression_postprocess.py', ('--part', 'explain'))]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, name, script, extra) for name, script, extra in independent]
        for future in as_completed(futures):
            timings.append(future.result())
    timings.append(run('plots', 'final_regression_postprocess.py', ('--part', 'plots')))
    timings.append(run('best_shap', 'final_regression_best_explain.py'))
    # The following comparisons are deliberately serial, after competing heavy
    # analyses finish. Primary matrix wall times retain their concurrency caveat.
    timings.append(run('sensitivity', 'final_regression_sensitivity.py'))
    timings.append(run('efficiency', 'final_regression_postprocess.py', ('--part', 'efficiency')))
    timings.append(run('validation', 'validate_final_regression.py', skip=False))
    timings.append(run('report', 'report_final_regression.py', skip=False))
    command = [sys.executable, '-u', str(ROOT / 'src/build_final_regression_delivery.py'),
               '--results-dir', str(out), '--execute-notebook']
    with (logdir / 'delivery.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Delivery construction failed; inspect stage_logs/delivery.log')
    print('Compiling the separate local book and ten-minute presentation', flush=True)
    with (logdir / 'delivery_compile.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'src/compile_final_regression_delivery.py'),
            '--pdf'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Offline delivery compilation failed; inspect stage_logs/delivery_compile.log')
    (out / 'completion_pipeline.json').write_text(json.dumps(dict(status='complete',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        stages=timings, publication='LOCAL ONLY', github_or_render_updated=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--wait-for-primary', action='store_true')
    args = parser.parse_args()
    if args.wait_for_primary:
        deadline = time.monotonic() + 4 * 3600
        while True:
            try:
                ready = read(args.out / 'status.json').get('status') == 'complete'
            except (FileNotFoundError, json.JSONDecodeError):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Primary study has not completed within four hours; no dependent work started.')
            time.sleep(10)
    if args.evaluate_only:
        evaluate_stage(args.out.resolve())
    else:
        finish(args.out)
