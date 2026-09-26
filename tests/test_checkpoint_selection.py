import json
import pytest

from twinlab import run_experiment as runner
from twinlab.selection import AGREEMENT_RULE, select_best, selection_metric


def test_deferred_selection_preserves_all_checkpoints_and_reuses_completed_checks(tmp_path, monkeypatch):
    scores = {64: .4, 96: .45, 128: .6}
    calls = []
    for update in scores:
        directory = tmp_path / f"update-{update}-adapter"
        directory.mkdir()
        (directory / "adapter_config.json").write_text(str(update))
    (tmp_path / "best_adapter").mkdir()
    (tmp_path / "best_adapter/adapter_config.json").write_text("32")

    class FakeEngine:
        def __init__(self, plan, adapter, runtime_profile):
            self.update = int(adapter.name.split("-")[1])
            calls.append(self.update)

        def evaluate(self, *args):
            return [], {"nll": {"mean": scores[self.update]}}

        def close(self):
            pass

    monkeypatch.setattr(runner, "Engine", FakeEngine)
    report = {"condition": "full_history", "status": "trained", "selected_update": 32,
              "checkpoints": [{"update": 32, "development": {"nll": {"mean": .5}}}]}
    result = runner.select_saved_checkpoints({}, None, tmp_path, {}, [], report, 128, 32, lambda *args: None)
    assert calls == [64, 96, 128]
    assert result["status"] == "complete" and result["selected_update"] == 64
    assert [c["update"] for c in result["checkpoints"]] == [32, 64, 96, 128]
    assert (tmp_path / "best_adapter/adapter_config.json").read_text() == "64"
    saved = json.loads((tmp_path / "training.json").read_text())
    runner.select_saved_checkpoints({}, None, tmp_path, {}, [], saved, 128, 32, lambda *args: None)
    assert calls == [64, 96, 128]


def test_partial_credit_selects_a_different_saved_checkpoint_without_inference(tmp_path, monkeypatch):
    def no_engine(*args, **kwargs):
        raise AssertionError("Completed development predictions must be reused")
    monkeypatch.setattr(runner, "Engine", no_engine)
    checkpoints = []
    for update, agreement, nll in ((32, .68, 1.3), (64, .72, 1.2), (96, .71, 1.1), (128, .70, 1.0)):
        p = tmp_path / f"update-{update}-adapter"
        p.mkdir()
        (p/'adapter_config.json').write_text(str(update))
        checkpoints.append({'update': update, 'development': {'agreement': {'mean': agreement}, 'nll': {'mean': nll}}})
    report = {'condition': 'full_history', 'selected_update': 128, 'checkpoints': checkpoints}
    plan = {'selection_metric': AGREEMENT_RULE}
    result = runner.select_saved_checkpoints(plan, None, tmp_path, {}, [], report, 128, 32, lambda *args: None)
    assert result['selected_update'] == 64
    assert 'partial-credit' in result['selected_by']
    assert (tmp_path/'best_adapter/adapter_config.json').read_text() == '64'
    assert select_best({c['update']: c['development'] for c in checkpoints}, 'nll') == 128


def test_historical_checkpoint_rule_can_differ_from_configuration_selection():
    plan = {'selection_metric': AGREEMENT_RULE,
            'execution_budget': {'checkpoint_selection': 'lowest_development_macro_categorical_nll'}}
    assert selection_metric(plan) == 'agreement'
    assert selection_metric(plan, checkpoint=True) == 'nll'
    with pytest.raises(ValueError, match='Non-finite'):
        select_best({'model': {'agreement': {'mean': float('nan')}}}, 'agreement')
