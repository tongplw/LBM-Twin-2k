"""Revise selection using saved checkpoints; never train or overwrite the source run.

The test has already been examined. Results from this workflow are retrospective.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .data import ROOT
from .benchmark_data import load_cohort
from .benchmark_metrics import average_assignments, fit_temperature, paired_difference, summarize, temperature_scale
from .experiment_engine import Engine, atomic_json
from .run_experiment import now, re_safe_name, saved_rows
from .selection import AGREEMENT_RULE, select_best, selection_description


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_source(source, hashes):
    for name, expected in hashes.items():
        if file_hash(source/name) != expected:
            raise ValueError(f'Source artifact changed: {name}')


def prepare(source, root):
    """Copy completed evidence with original identities and explicit provenance."""
    if source.resolve() == root.resolve():
        raise ValueError('Revision must use a separate run directory')
    (root/'report').mkdir(parents=True, exist_ok=True)
    path = root/'report/experiment.json'
    if path.exists():
        run = json.loads(path.read_text())
        if run['revision']['source_run'] != source.name:
            raise ValueError('Revision already refers to a different source run')
        verify_source(source, run['revision']['source_sha256'])
        return run
    original = json.loads((source/'report/experiment.json').read_text())
    if original['status'] != 'complete':
        raise ValueError('Source study must be complete')
    if json.loads((source/'selection.json').read_text()) != original['selection']:
        raise ValueError('Source selection differs from its sealed decision')
    run = deepcopy(original)
    hashes = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*')
              if p.is_file() and p.suffix in {'.json', '.jsonl', '.safetensors'}}
    run['revision'] = {'source_run': source.name, 'created': now(), 'retrospective': True,
        'source_sha256': hashes, 'original_selection': original['selection'],
        'original_test': {k: v for k, v in original['stages'].items() if k.startswith('test_selected')},
        'policy': 'partial credit selects saved checkpoints and eligible configurations; no retraining'}
    run.update(status='running', started=now(), updated=now(), final_test_opened=False, comparisons={})
    for key in ('selection', 'error', 'publication_length_matching'):
        run.pop(key, None)
    run['stages'] = {k: v for k, v in run['stages'].items()
                     if not k.startswith(('sft_', 'test_selected', 'selected_calibrated'))}
    manifest = deepcopy(original['manifest'])
    manifest.update(created=now(), source_sha256={str(p.relative_to(ROOT)): file_hash(p)
                    for p in (ROOT/'src/twinlab').glob('*.py')})
    manifest['plan']['selection_metric'] = AGREEMENT_RULE
    manifest['plan']['execution_budget']['checkpoint_selection'] = 'highest_development_macro_partial_credit'
    manifest['revision'] = {k: v for k, v in run['revision'].items() if k not in ('source_sha256', 'original_test')}
    run['manifest'] = manifest
    # Preserve cached identities as evidence of the original computation. They
    # are never passed to Engine under a changed plan or relabeled as new runs.
    for directory in [*source.glob('frozen_*'), source/'retrieval-dev']:
        shutil.copytree(directory, root/directory.name, dirs_exist_ok=True)
    for mode, report in run['training'].items():
        directory = root/('sft_'+mode)
        directory.mkdir(parents=True, exist_ok=True)
        scores = {c['update']: c['development'] for c in report['checkpoints']}
        chosen = select_best(scores, 'agreement')
        report['original_selected_update'] = report['selected_update']
        report.update(selected_update=chosen, selected_by=selection_description('agreement'))
        source_dir = source/directory.name
        adapter = source_dir/f'update-{chosen}-adapter'
        if not adapter.exists():
            if chosen == report['original_selected_update']:
                adapter = source_dir/'best_adapter'
            elif mode == 'question_only':
                # This checkpoint's complete development predictions survive,
                # but its weights were overwritten by historical NLL selection.
                # It can be reused only if another available candidate strictly
                # dominates it, so it cannot be required for the final test.
                full = run['training']['full_history']['checkpoints']
                if max(c['development']['agreement']['mean'] for c in full) <= scores[chosen]['agreement']['mean']:
                    raise ValueError('The potentially winning question-only checkpoint is unavailable')
                adapter = None
            else:
                raise ValueError(f'Selected saved adapter is unavailable: {adapter}')
        report['selected_adapter_available'] = adapter is not None
        if adapter is not None:
            shutil.copytree(adapter, directory/'best_adapter', dirs_exist_ok=True)
        for checkpoint in source_dir.glob('update-*-dev'):
            shutil.copytree(checkpoint, directory/checkpoint.name, dirs_exist_ok=True)
        atomic_json(directory/'training.json', report)
        run['stages']['sft_'+mode] = json.loads((directory/f'update-{chosen}-dev/summary.json').read_text())
    shutil.copyfile(source/'runtime_profile.json', root/'runtime_profile.json')
    atomic_json(root/'manifest.json', manifest)
    atomic_json(path, run)
    return run


def run_revision(source, root):
    run = prepare(source, root)
    if run['status'] == 'complete':
        print(f'Revision already complete: {root}', flush=True)
        return
    run.update(status='running')
    run.pop('error', None)
    plan = run['manifest']['plan']
    runtime = json.loads((root/'runtime_profile.json').read_text())

    def publish(stage=None, summary=None):
        if stage:
            run['stages'][stage] = summary
        run['updated'] = now()
        atomic_json(root/'report/experiment.json', run)

    engine = None
    try:
        people, examples = load_cohort('dev', 'full')
        jobs = [('sft_hybrid', 'hybrid', 42)] + [(f'sft_shuffled_{s}', 'shuffled', s) for s in (42, 43, 44)]
        if any(stage not in run['stages'] for stage, _, _ in jobs):
            engine = Engine(plan, root/'sft_full_history/best_adapter', runtime_profile=runtime)
            for stage, mode, seed in jobs:
                if stage in run['stages']:
                    continue
                run['active_stage'] = stage
                publish()
                _, summary = engine.evaluate(people, examples, mode, root/stage, root/'retrieval-dev', seed)
                publish(stage, summary)
            engine.close()
            engine = None
        outcomes = {}
        for name in run['stages']:
            if name.startswith('test_') or name == 'selected_calibrated_development':
                continue
            directory = root/name
            if name in ('sft_question_only', 'sft_full_history'):
                mode = name.removeprefix('sft_')
                directory /= f"update-{run['training'][mode]['selected_update']}-dev"
            if (directory/'predictions.jsonl').exists():
                outcomes[name] = saved_rows(directory)
        for mode in ('frozen_shuffled', 'frozen_random', 'sft_shuffled'):
            outcomes[mode] = average_assignments([outcomes[f'{mode}_{s}'] for s in (42, 43, 44)])
            summary = summarize(outcomes[mode])
            summary['assignment_macro_nll'] = [run['stages'][f'{mode}_{s}']['nll']['mean'] for s in (42, 43, 44)]
            publish(mode, summary)
        # Statistical references require no new model inference.
        from .benchmark_metrics import Baselines, score_prediction
        _, training = load_cohort('train', 'full')
        baseline = Baselines(training)
        for mode in ('common_answer', 'price_aware', 'human_retest'):
            outcomes[mode] = [r for e in examples for r in score_prediction(e, baseline.predict(e, mode))]
        from .finalize_submission import EFFECTS
        for _, name in EFFECTS:
            a, b = name.split(' minus ')
            run['comparisons'][name] = {f: paired_difference(outcomes[a], outcomes[b], f)
                                        for f in ('exact', 'agreement', 'nll')}
        candidates = ['frozen_question_only', 'frozen_full_history', 'frozen_hybrid',
                      'sft_question_only', 'sft_full_history', 'sft_hybrid']
        selected = select_best({n: run['stages'][n] for n in candidates}, 'agreement')
        calibration = fit_temperature(outcomes[selected])
        decision = {'method': selected, 'rule': selection_description('agreement'),
                    'calibration': calibration, 'retrospective': True}
        seal = root/'selection.json'
        if seal.exists():
            existing = json.loads(seal.read_text())
            if any(existing[k] != v for k, v in decision.items()):
                raise ValueError('Revised selection changed after sealing')
            decision = existing
        else:
            decision['frozen_at'] = now()
            atomic_json(seal, decision)
        run['selection'] = decision
        publish('selected_calibrated_development', summarize(temperature_scale(outcomes[selected], calibration['temperature'])))
        run.update(final_test_opened=True, active_stage='final-test')
        publish()
        print(f'Retrospective test selection: {selected}', flush=True)
        test_people, testing = load_cohort('test', 'full')
        mode = selected.removeprefix('sft_').removeprefix('frozen_')
        if mode == 'hybrid':
            from transformers import AutoTokenizer
            from .retrieval import prepare_rankings
            tokenizer = AutoTokenizer.from_pretrained(plan['model'], revision=plan['model_revision'])
            prepare_rankings(test_people, testing, tokenizer, root/'retrieval-test', plan['embedding_revision'])
        adapter = root/('sft_question_only' if mode == 'question_only' else 'sft_full_history')/'best_adapter' if selected.startswith('sft_') else None
        engine = Engine(plan, adapter, runtime_profile=runtime)
        rows, summary = engine.evaluate(test_people, testing, mode, root/'final-test', root/'retrieval-test')
        engine.close()
        engine = None
        publish('test_selected_uncalibrated', summary)
        publish('test_selected_calibrated', summarize(temperature_scale(rows, calibration['temperature'])))
        verify_source(source, run['revision']['source_sha256'])
        run.update(status='complete', active_stage=None)
        publish()
        print(f'Revised evaluation complete: {root}. Finalize separately after report review.', flush=True)
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error=f'{type(error).__name__}: {error}')
        publish()
        raise
    finally:
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='main-study')
    parser.add_argument('--name', default='partial-credit-study')
    args = parser.parse_args()
    if not all(re_safe_name(n) for n in (args.source, args.name)):
        parser.error('Run names may contain letters, digits, underscores and hyphens only')
    source, root = (ROOT/'checkpoints'/n for n in (args.source, args.name))
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_revision(source, root)


if __name__ == '__main__':
    main()
