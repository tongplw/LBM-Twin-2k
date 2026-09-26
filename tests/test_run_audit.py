import pytest

from twinlab.audit_run import check_score, snapshot_records


def test_audit_rejects_copied_exact_score_in_partial_credit_column():
    atom = {"value": 60., "kind": "bounded_numeric", "levels": None,
            "lower": 0., "upper": 100., "key": "example"}
    row = {"truth": 60., "kind": "bounded_numeric", "prediction": 70.,
           "invalid": 0., "exact": 0., "agreement": .9, "nll": None,
           "brier": None, "absolute_error": 10.}
    check_score(atom, row)
    with pytest.raises(AssertionError):
        check_score(atom, {**row, "agreement": 0.})


def test_live_audit_does_not_truncate_inflight_prediction_write(tmp_path):
    path = tmp_path / "predictions.jsonl"
    original = b'{"complete": true}\n{"unfinished":'
    path.write_bytes(original)
    assert snapshot_records(path) == [{"complete": True}]
    assert path.read_bytes() == original


def test_test_audit_cannot_open_outcomes_before_completion(tmp_path):
    import json
    from twinlab.audit_run import audit
    (tmp_path / "report").mkdir()
    (tmp_path / "report/experiment.json").write_text(json.dumps({"status": "running", "final_test_opened": False}))
    with pytest.raises(ValueError, match="completed reserved-test"):
        audit(tmp_path, include_test=True)


def test_history_audit_detects_target_copied_under_another_id():
    import json
    from twinlab.audit_run import check_history_targets
    target = {"QuestionID": "target", "QuestionType": "MC", "QuestionText": "Choose a color", "Options": ["Red", "Blue"]}
    history = {**target, "QuestionID": "different", "Answers": {"SelectedText": "Red"}}
    person = {"wave1_3_persona_json": json.dumps([{"Questions": [history]}])}
    with pytest.raises(ValueError, match="duplicate target"):
        check_history_targets(person, [{"question": target}])
