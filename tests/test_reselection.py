"""Changing the decision rule must preserve historical evidence and model identity."""
import json

import pytest

from twinlab import reselect_experiment as revision
from twinlab.selection import NLL_RULE


def source_run(tmp_path, question_score=.70):
    source = tmp_path/'original'
    (source/'report').mkdir(parents=True)
    training, stages = {}, {}
    for mode, peak in [('question_only', question_score), ('full_history', .71)]:
        directory = source/('sft_'+mode)
        checkpoints = []
        for update, agreement in [(64, peak), (128, .69)]:
            score = {'agreement': {'mean': agreement}, 'nll': {'mean': 1.2 if update == 64 else 1.1}}
            path = directory/f'update-{update}-dev'
            path.mkdir(parents=True)
            (path/'summary.json').write_text(json.dumps(score))
            (path/'predictions.jsonl').write_text('{"saved":true}\n')
            checkpoints.append({'update': update, 'development': score})
        # Match the real retention gap: only full-history update 64 survived.
        if mode == 'full_history':
            path = directory/'update-64-adapter'
            path.mkdir()
            (path/'adapter_model.safetensors').write_bytes(b'full-history-64')
        training[mode] = {'selected_update': 128, 'checkpoints': checkpoints}
        stages['sft_'+mode] = score
    (source/'retrieval-dev').mkdir()
    (source/'runtime_profile.json').write_text('{}')
    decision = {'method': 'sft_question_only', 'calibration': {'temperature': 1.}}
    run = {'status': 'complete', 'selection': decision, 'training': training, 'stages': stages,
           'manifest': {'plan': {'selection_metric': NLL_RULE, 'execution_budget': {}}}}
    (source/'selection.json').write_text(json.dumps(decision))
    (source/'report/experiment.json').write_text(json.dumps(run))
    return source


def test_revision_preserves_original_and_does_not_substitute_wrong_weights(tmp_path):
    source = source_run(tmp_path)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    root = tmp_path/'revised'
    run = revision.prepare(source, root)
    assert run['training']['question_only']['selected_update'] == 64
    assert not run['training']['question_only']['selected_adapter_available']
    assert not (root/'sft_question_only/best_adapter').exists()
    assert (root/'sft_full_history/best_adapter/adapter_model.safetensors').read_bytes() == b'full-history-64'
    assert run['revision']['retrospective']
    assert not run['final_test_opened']
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    assert revision.prepare(source, root) == run
    (source/'selection.json').write_text('{}')
    with pytest.raises(ValueError, match='Source artifact changed'):
        revision.prepare(source, root)


@pytest.mark.parametrize('question_score', [.71, .72])
def test_missing_weights_block_a_potential_winner_including_ties(tmp_path, question_score):
    source = source_run(tmp_path, question_score)
    with pytest.raises(ValueError, match='potentially winning'):
        revision.prepare(source, tmp_path/'revision')


def test_only_missing_controls_and_development_selected_test_run(tmp_path, monkeypatch):
    source = source_run(tmp_path)
    path = source/'report/experiment.json'
    run = json.loads(path.read_text())
    frozen = ['frozen_'+n for n in ('question_only', 'full_history', 'demographics', 'hybrid', 'bm25', 'semantic')]
    frozen += [f'frozen_{n}_{s}' for n in ('random', 'shuffled') for s in (42, 43, 44)]
    summary = {'agreement': {'mean': .6}, 'nll': {'mean': 1.5}}
    for name in frozen:
        directory = source/name
        directory.mkdir()
        (directory/'predictions.jsonl').write_text('{}\n')
        run['stages'][name] = summary
    path.write_text(json.dumps(run))
    calls, cohorts = [], []

    class FakeEngine:
        def __init__(self, plan, adapter, **kwargs):
            assert adapter.name == 'best_adapter'
            assert adapter.parent.name == 'sft_full_history'
            assert (adapter/'adapter_model.safetensors').read_bytes() == b'full-history-64'

        def evaluate(self, people, examples, mode, destination, *args):
            calls.append((destination.name, mode))
            if destination.name == 'final-test':
                assert json.loads((destination.parent/'selection.json').read_text())['method'] == 'sft_full_history'
            destination.mkdir()
            (destination/'predictions.jsonl').write_text('{}\n')
            return [], {'agreement': {'mean': .705}, 'nll': {'mean': 1.25}}

        def close(self):
            pass

    def load(split, scope):
        cohorts.append(split)
        return {}, []

    monkeypatch.setattr(revision, 'Engine', FakeEngine)
    monkeypatch.setattr(revision, 'load_cohort', load)
    monkeypatch.setattr(revision, 'saved_rows', lambda path: [])
    monkeypatch.setattr(revision, 'average_assignments', lambda rows: [])
    monkeypatch.setattr(revision, 'summarize', lambda rows: summary)
    monkeypatch.setattr(revision, 'paired_difference', lambda *args: {'mean': 0., 'lo': -.01, 'hi': .01})
    monkeypatch.setattr(revision, 'fit_temperature', lambda rows: {'temperature': 1.2})
    monkeypatch.setattr(revision, 'temperature_scale', lambda *args: [])
    from twinlab import benchmark_metrics
    monkeypatch.setattr(benchmark_metrics, 'Baselines', lambda rows: None)
    root = tmp_path/'revision'
    root.mkdir()
    revision.run_revision(source, root)
    assert calls == [('sft_hybrid', 'hybrid'), ('sft_shuffled_42', 'shuffled'),
                     ('sft_shuffled_43', 'shuffled'), ('sft_shuffled_44', 'shuffled'), ('final-test', 'full_history')]
    assert cohorts == ['dev', 'train', 'test']
    assert json.loads((root/'report/experiment.json').read_text())['status'] == 'complete'
    revision.run_revision(source, root)
    assert len(calls) == 5
