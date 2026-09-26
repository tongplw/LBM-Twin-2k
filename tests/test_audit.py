import copy
import json

import pytest

from twinlab.audit import audit_full_personas
from twinlab.data import target_atoms
from twinlab.download import assets


def persona(answer=1, qid="QID196"):
    return [{"BlockName": "Ratio bias", "Questions": [{
        "QuestionID": qid, "QuestionType": "MC", "QuestionText": "Choose a tray",
        "Options": ["Small", "Large"], "Answers": {"SelectedByPosition": answer},
    }]}]


def reference(early=1, late=2):
    a, b = target_atoms(persona(early))[0], target_atoms(persona(late))[0]
    return {a.key: (a, b)}


def row(pid, answer=2):
    return {"pid": pid, "persona_json": json.dumps(persona(answer))}


def test_full_audit_counts_changed_and_unchanged_and_skips_reserved_json():
    refs = {"a": reference(), "b": reference(2, 2)}
    rows = [row("a"), row("b"), {"pid": "test", "persona_json": "INVALID JSON"}]
    result = audit_full_personas(rows, refs, {"a": "train", "b": "dev", "test": "test"})
    assert result["expected_participants"] == result["participants"] == 2
    assert result["expected_responses"] == result["comparable_responses"] == 2
    assert result["changed_responses"] == result["changed_matches_wave4"] == 1
    assert result["unchanged_responses"] == result["unchanged_matches_wave4"] == 1
    assert result["matches_wave4"] == 2 and result["disagrees_with_wave4"] == 0
    assert result["reserved_test_participants_excluded"] == 1


def test_full_audit_keeps_missing_participants_and_questions_in_denominator():
    result = audit_full_personas([{"pid": "a", "persona_json": "[]"}],
                                {"a": reference(), "b": reference()}, {"a": "train", "b": "dev"})
    assert result["expected_responses"] == 2
    assert result["missing_participants"] == 1 and result["missing_responses"] == 2
    assert result["present_responses"] == result["comparable_responses"] == 0


def test_full_audit_reports_disagreements_invalid_and_missing_answers():
    refs = {pid: reference() for pid in ("a", "b", "c")}
    result = audit_full_personas([row("a", 1), row("b", 99), row("c", None)],
                                refs, {pid: "train" for pid in refs})
    assert result["present_responses"] == 3
    assert result["invalid_responses"] == 1 and result["missing_or_invalid_values"] == 2
    assert result["changed_atom_comparisons"] == result["changed_disagrees_with_wave4"] == 1
    assert result["matches_wave4"] == 0 and result["disagrees_with_wave4"] == 1


def test_full_audit_matches_question_context_and_checks_stimuli():
    changed = persona(2)
    changed[0]["Questions"][0]["QuestionText"] = "A different task"
    unrelated = copy.deepcopy(persona(1))
    unrelated[0]["BlockName"] = "Unrelated reused question ID"
    result = audit_full_personas([{"pid": "a", "persona_json": json.dumps(changed + unrelated)}],
                                {"a": reference()}, {"a": "dev"})
    assert result["present_responses"] == result["stimulus_mismatches"] == 1
    assert result["comparable_responses"] == result["matches_wave4"] == 0


def test_full_audit_rejects_duplicate_people_and_response_keys():
    with pytest.raises(ValueError, match="Duplicate full-persona participant"):
        audit_full_personas([row("a"), row("a")], {"a": reference()}, {"a": "train"})
    duplicate = {"pid": "a", "persona_json": json.dumps(persona(2) * 2)}
    with pytest.raises(ValueError, match="Duplicate target key"):
        audit_full_personas([duplicate], {"a": reference()}, {"a": "train"})


def test_full_audit_rejects_unknown_or_wrong_scope_references():
    with pytest.raises(ValueError, match="Unknown full-persona participant"):
        audit_full_personas([row("unknown")], {"a": reference()}, {"a": "train"})
    with pytest.raises(ValueError, match="exactly train/dev"):
        audit_full_personas([], {"test": reference()}, {"a": "train", "test": "test"})


def test_download_includes_complete_full_persona_source():
    names = [name for name, _ in assets() if name.startswith("persona_chunk_")]
    assert names == [f"persona_chunk_{i:03d}.parquet" for i in range(1, 8)]


def test_download_preserves_analysis_catalog_and_pinned_revisions():
    from twinlab.download import PLAN, TOKENIZER_FILE
    downloads = dict(assets())
    assert 'question_catalog.json' in downloads
    assert PLAN['dataset_revision'] in downloads['question_catalog.json']
    assert PLAN['model_revision'] in downloads[TOKENIZER_FILE.name]
    assert all('/resolve/main/' not in url for url in downloads.values())
