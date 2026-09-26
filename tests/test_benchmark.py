import copy
import json

import numpy as np
import pytest

from twinlab.benchmark_data import evidence_items, pack_evidence, messages_for, target_text, balanced_schedule, sampled_coverage
from twinlab.benchmark_metrics import (score_prediction, summarize, paired_difference,
    average_assignments, Baselines, temperature_scale)
from twinlab.experiment_engine import read_predictions, Engine
from twinlab.retrieval import donor_assignment, rrf_scores, matched_random


def example(pid="1", group="Task", value=1, levels=2):
    return {"pid": pid, "id": "block|Q1", "group": group, "context": [],
        "question": {"QuestionID": "Q1", "QuestionType": "MC", "QuestionText": "Choose", "Options": ["yes", "no"]},
        "atoms": [{"key": "block|Q1|0", "family": group, "kind": "binary" if levels else "bounded_numeric",
                   "value": value, "levels": levels, "lower": 1 if levels else 0,
                   "upper": levels or 100, "earlier": 1, "price": None}]}


def test_matrix_retrieval_items_preserve_rows_scales_zero_and_sources():
    history = [{"BlockName": "Personality", "Questions": [{
        "QuestionID": "QID3", "QuestionType": "Matrix", "QuestionText": "Shared wording",
        "Rows": ["first", "second"], "RowsID": ["2", "7"], "Columns": ["a", "b"],
        "Answers": {"SelectedByPosition": [1, 2], "SelectedText": ["a", "b"]}}]}]
    items = evidence_items(history)
    assert len(items) == 2
    assert items[1]["question"]["RowsID"] == ["7"]
    assert items[1]["answer"]["SelectedByPosition"] == [2]
    packed = pack_evidence(items)
    assert len(packed) == 1
    assert packed[0]["question"]["Rows"] == ["first", "second"]
    assert packed[0]["question"]["Columns"] == ["a", "b"]
    assert packed[0]["answer"] == history[0]["Questions"][0]["Answers"]
    assert pack_evidence(items[1:])[0]["question"]["Rows"] == ["second"]
    assert items[0]["source"].endswith("/row2")  # packing never mutates candidates


def test_prompt_and_retrieval_query_do_not_depend_on_any_target_label():
    e = example()
    changed = copy.deepcopy(e)
    changed["atoms"][0].update(value=2, earlier=2)
    assert messages_for(e, []) == messages_for(changed, [])
    assert target_text(e) == target_text(changed)


def test_incomplete_optional_history_matrix_is_not_falsely_aligned_or_dropped():
    history = [{"BlockName": "Personality", "Questions": [{
        "QuestionID": "Q29", "QuestionType": "Matrix", "QuestionText": "Optional history",
        "Rows": ["first", "second", "third"], "Columns": ["a", "b"],
        "Answers": {"SelectedByPosition": [1, 2], "SelectedText": ["a", "b"]}}]}]
    items = evidence_items(history)
    assert len(items) == 1
    assert items[0]["answer"] == history[0]["Questions"][0]["Answers"]
    assert items[0]["question"]["Rows"] == ["first", "second", "third"]
    assert "ambiguous" in items[0]["row_alignment"]


def test_donors_are_deterministic_bijective_and_not_self():
    ids = list(map(str, range(30)))
    a = donor_assignment(ids, 42)
    assert a == donor_assignment(list(reversed(ids)), 42)
    assert set(a.values()) == set(ids)
    assert all(k != v for k, v in a.items())
    assert a != donor_assignment(ids, 43)


def test_rrf_omits_zero_keyword_matches_and_keeps_negative_semantic_scores():
    scores = rrf_scores([0, 2, 1], [-.1, -.2, -.3])
    assert scores[0] == pytest.approx(1/61)
    assert scores[1] == pytest.approx(1/61+1/62)
    assert scores[2] == pytest.approx(1/62+1/63)


def test_random_control_matches_distinct_item_count_and_length():
    lengths = [100]*30
    chosen = matched_random(lengths, 1600, 16, 42)
    assert len(chosen) == len(set(chosen)) == 16
    assert sum(lengths[i] for i in chosen) == 1600
    assert chosen == matched_random(lengths, 1600, 16, 42)


def test_task_macro_not_dominated_by_large_task():
    rows = []
    for pid in ["1", "2"]:
        e = example(pid, "Small")
        rows.extend(score_prediction(e, {"values": [1], "probabilities": [[.9, .1]]}))
        for _ in range(40):
            rows.extend(score_prediction(example(pid, "Large"), {"values": [2], "probabilities": [[.1, .9]]}))
    result = summarize(rows, draws=20)
    assert result["exact"]["mean"] == .5
    assert result["nll"]["mean"] == pytest.approx((-np.log(.9)-np.log(.1))/2)


def test_invalid_numeric_outputs_count_as_zero_without_imputed_error():
    e = example(levels=None, value=20)
    rows = score_prediction(e, {"values": [200]})
    assert rows[0]["invalid"] == 1
    assert rows[0]["agreement"] == rows[0]["exact"] == 0
    assert rows[0]["absolute_error"] is None
    assert rows[0]["nll"] is None


def test_partial_credit_differs_from_exact_for_ordered_and_numeric_answers():
    e = example(value=3, levels=5)
    e["atoms"][0]["kind"] = "ordinal"
    row = score_prediction(e, {"values": [4], "probabilities": [[.1, .1, .2, .5, .1]]})[0]
    assert row["exact"] == 0 and row["agreement"] == .75
    e = example(value=20, levels=None)
    row = score_prediction(e, {"values": [30]})[0]
    assert row["exact"] == 0 and row["agreement"] == .9


def test_balanced_training_visits_people_and_tasks_without_reading_answers():
    examples = [example(str(pid), group) for pid in range(4) for group in ("one", "two", "three")]
    schedule = balanced_schedule(examples, 12)
    assert len(set(schedule)) == 12
    assert len({examples[i]["pid"] for i in schedule[:4]}) == 4
    changed = copy.deepcopy(examples)
    for e in changed:
        e["atoms"][0]["value"] = 2
    assert schedule == balanced_schedule(changed, 12)


def test_training_coverage_counts_only_consumed_schedule_prefix():
    examples = [example(str(pid), group) for pid in range(4) for group in ("one", "two", "three")]
    schedule = balanced_schedule(examples, 12)
    before = sampled_coverage(examples, schedule, 0)
    assert before["sampled_participants"] == before["processed_draws"] == 0
    assert before["planned_participants"] == 4
    partial = sampled_coverage(examples, schedule, 2)
    assert partial["sampled_participants"] == partial["unique_sampled_questions"] == 2
    assert partial["draws_relative_to_question_count"] == 2/12
    assert partial["planned_draws_relative_to_question_count"] == 1
    done = sampled_coverage(examples, schedule, 12)
    assert done["sampled_participants"] == 4 and done["sampled_response_rows"] == 12


def test_paired_scores_require_same_people_and_answers():
    e = example()
    a = score_prediction(e, {"values": [1], "probabilities": [[.8, .2]]})
    b = score_prediction(e, {"values": [2], "probabilities": [[.2, .8]]})
    result = paired_difference(a, b, "nll", draws=20)
    assert result["mean"] == pytest.approx(np.log(.2/.8))
    b[0]["pid"] = "other"
    with pytest.raises(ValueError, match="identical"):
        paired_difference(a, b, "nll")


def test_repeated_assignments_do_not_inflate_participant_count():
    e = example()
    a = score_prediction(e, {"values": [1], "probabilities": [[.8, .2]]})
    b = score_prediction(e, {"values": [2], "probabilities": [[.2, .8]]})
    averaged = average_assignments([a, b])
    report = summarize(averaged, draws=20)
    assert report["participants"] == report["responses"] == 1
    assert report["exact"]["mean"] == .5


def test_baseline_training_only_and_calibration_does_not_change_point_rule():
    training = [example(value=1), example(value=1), example(value=2)]
    baseline = Baselines(training)
    e = example(value=2)
    prediction = baseline.predict(e)
    assert prediction["probabilities"][0] == pytest.approx([.6, .4])
    rows = score_prediction(e, prediction)
    scaled = temperature_scale(rows, 2)
    assert scaled[0]["prediction"] == rows[0]["prediction"]
    assert scaled[0]["exact"] == rows[0]["exact"]
    assert scaled[0]["nll"] < rows[0]["nll"]


def test_prediction_resume_recovers_only_incomplete_final_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"a": 1}\n{"a":')
    assert read_predictions(path) == [{"a": 1}]
    assert path.read_text() == '{"a": 1}\n'
    path.write_text('{"a": 1}\ninvalid\n')
    with pytest.raises(json.JSONDecodeError):
        read_predictions(path)


def test_identical_prompt_reuse_scores_every_person_and_is_local_to_evaluation(tmp_path):
    from types import SimpleNamespace
    engine = Engine.__new__(Engine)
    engine.plan, engine.metadata, engine.tokenizer = {}, {}, None
    engine.question_prediction_reuse = True
    engine.torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None,
                                                        max_memory_allocated=lambda: 0))
    engine.reset_prefix_cache = lambda: None
    calls = []
    def predict(e, items):
        calls.append(e["pid"])
        return {"values": [1], "probabilities": [[.8, .2]], "raw_output": "A",
                "input_tokens": 10, "inference_seconds": .1, "processed_input_tokens": 10}
    engine.predict = predict
    examples = [example("1", value=1), example("2", value=2)]
    people = {"1": [], "2": []}
    rows, summary = engine.evaluate(people, examples, "question_only", tmp_path / "first")
    assert len(calls) == 2  # one warmup, one actual inference, one reuse
    assert len(rows) == 2 and [r["exact"] for r in rows] == [1, 0]
    assert summary["exact"]["mean"] == .5
    assert summary["timing"]["identical_prompt_predictions_reused"] == 1
    assert summary["timing"]["processed_input_tokens"] == 10
    engine.evaluate(people, examples, "question_only", tmp_path / "first")
    assert len(calls) == 2  # completed evaluation resumes without inference
    engine.evaluate(people, examples, "question_only", tmp_path / "second")
    assert len(calls) == 4  # prediction cache cannot cross checkpoints or conditions
