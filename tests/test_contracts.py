import copy
import json
import numpy as np
import pytest

from twinlab.data import ROOT, TASK_GROUPS, question_records, target_atoms, make_splits, stimulus, task_group
from twinlab.analyze import grouped_scores, macro_interval
from twinlab.metrics import agreement_score, categorical_metrics, bootstrap_mean
from twinlab.experiment_engine import language_lora_targets


def blocks(question, name="False consensus"):
    return [{"BlockName": name, "Questions": [question]}]


def test_question_ids_do_not_overwrite_each_other():
    q = {"QuestionID": "QID268", "QuestionType": "MC"}
    data = blocks(q, "Cognitive tests") + blocks({**q, "QuestionType": "TE"}, "Personality")
    result = list(question_records(data))
    assert len(result) == 2 and result[0][0] != result[1][0]


def test_nonconsecutive_matrix_row_ids():
    q = {"QuestionID": "QID287", "QuestionType": "Matrix", "Rows": ["a", "b"],
         "RowsID": ["7", "10"], "Columns": ["Disagree", "Neutral", "Agree"],
         "Answers": {"SelectedByPosition": [1, 3]}}
    atoms = target_atoms(blocks(q))
    assert [a.subid for a in atoms] == ["7", "10"]
    assert [a.value for a in atoms] == [1, 3]


def test_zero_is_valid_and_out_of_range_is_not_clipped():
    q = {"QuestionID": "QID182", "QuestionType": "TE", "Answers": {"Text": "0"}}
    atom = target_atoms(blocks(q, "Sunk cost - yes"))[0]
    assert atom.value == 0 and not atom.invalid
    q["Answers"]["Text"] = "21"
    atom = target_atoms(blocks(q, "Sunk cost - yes"))[0]
    assert atom.value is None and atom.invalid


def test_nominal_distance_is_never_numeric_distance():
    assert agreement_score(1, 2, 1, 5, "nominal") == 0
    assert agreement_score(1, 2, 1, 5, "ordinal") == .75
    assert agreement_score(10, 20, None, None, "unbounded_numeric") is None


def test_split_disjoint_and_inspected_people_not_test():
    # Use actual-like IDs with the two known preliminary records.
    splits = make_splits(range(1, 2059))
    train, dev, test = map(set, (splits["train"], splits["dev"], splits["test"]))
    assert [len(train),len(dev),len(test)] == [1440,309,309]
    assert not (train & dev or dev & test or train & test)
    assert {"574", "2001"} <= train


def test_persisted_split_matches_original_assignment():
    saved = json.loads((ROOT / "splits/participants.json").read_text())
    ids = [pid for group in saved.values() for pid in group]
    assert len(ids) == len(set(ids)) == 2058
    assert make_splits(ids) == saved


def test_main_task_mapping_preserves_all_diagnostic_families():
    assert len(TASK_GROUPS) == 19
    assert len(set(TASK_GROUPS.values())) == 17
    assert task_group("Anchoring: countries") == task_group("Anchoring: height") == "Anchoring"
    assert task_group("Less is more") == task_group("Proportion dominance") == "Less is More (combined)"
    assert task_group("Pricing") == "Pricing"
    with pytest.raises(KeyError):
        task_group("Unreviewed task")


def test_main_groups_pool_responses_not_subgroup_means():
    rows = [{"pid": "p", "task_group": "Less is More (combined)", "family": family, "score": score}
            for family, score in [("Less is more", 0), ("Proportion dominance", 1),
                                  ("Proportion dominance", 1)]]
    main = grouped_scores(rows, "score", ["p"], ["Less is More (combined)"])
    detail = grouped_scores(rows, "score", ["p"], ["Less is more", "Proportion dominance"], "family")
    assert main[0, 0] == pytest.approx(2 / 3)
    assert detail.mean() == .5


def test_groups_weight_people_equally_and_exclude_unbounded_scores():
    rows = [{"pid": "a", "task_group": "Pricing", "score": 1}] * 40
    rows += [{"pid": "b", "task_group": "Pricing", "score": 0},
             {"pid": "a", "task_group": "Anchoring", "score": 0},
             {"pid": "b", "task_group": "Anchoring", "score": 0},
             {"pid": "a", "task_group": "Anchoring", "score": None}]
    matrix = grouped_scores(rows, "score", ["a", "b"], ["Pricing", "Anchoring"])
    assert np.array_equal(matrix, [[1, 0], [0, 0]])
    assert macro_interval(matrix, draws=100)["mean"] == .25








def test_probability_scores_and_validation():
    scores = categorical_metrics([[.8,.2],[.3,.7]], [0,1])
    assert scores["accuracy"] == 1
    assert scores["nll"] == pytest.approx(-np.log([.8,.7]).mean())
    assert scores["brier"] == pytest.approx(.13)
    with pytest.raises(ValueError): categorical_metrics([[.8,.8]], [0])


def test_bootstrap_preserves_constant_participant_scores():
    result = bootstrap_mean([.7]*20, draws=200)
    assert result["lo"] == pytest.approx(.7)
    assert result["hi"] == pytest.approx(.7)






def test_product_price_is_preserved_including_free_offers():
    q = {"QuestionID": "QID500", "QuestionType": "MC", "QuestionText": "Product priced at: $0.00",
         "Options": ["Yes", "No"], "Answers": {"SelectedByPosition": 2}}
    first = target_atoms(blocks(q, "Product Preferences"))[0]
    assert first.price == 0 and first.value == 2
    q["QuestionText"] = "Product priced at: $12.50"
    second = target_atoms(blocks(q, "Product Preferences"))[0]
    assert second.price == 12.5 and first.stimulus_hash != second.stimulus_hash


def test_unreviewed_multiclass_scale_is_not_assumed_ordinal():
    q = {"QuestionID": "QID999", "QuestionType": "MC", "Options": ["Red", "Green", "Blue"],
         "Answers": {"SelectedByPosition": 2}}
    with pytest.raises(ValueError, match="Unreviewed categorical scale"):
        target_atoms(blocks(q, "Disease framing"))




def test_qwen_adapters_exclude_vision_and_output_layers():
    class Linear:
        pass
    class Model:
        def named_modules(self):
            return [("model.language_model.layers.0.linear_attn.in_proj_qkv", Linear()),
                    ("model.language_model.layers.3.self_attn.q_proj", Linear()),
                    ("model.visual.blocks.0.attn.qkv", Linear()), ("lm_head", Linear())]
    names = language_lora_targets(Model(), Linear)
    assert len(names) == 2 and all(n.startswith("model.language_model.layers.") for n in names)
