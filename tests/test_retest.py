import pytest

from twinlab.analyze import retest_agreement_comparison, retest_by_answer_type


def response(pid="a", group="ratings", exact=0, partial=.75, kind="ordinal"):
    return {"pid": pid, "task_group": group, "exact": exact, "human": partial, "kind": kind}


def test_retest_compares_same_bounded_answers_and_complements_intervals():
    rows = [response(), response(exact=1, partial=1),
            response(group="estimates", exact=0, partial=None, kind="unbounded_numeric")]
    result = retest_agreement_comparison(rows, ["a"], ["ratings"], draws=20)
    assert result["paired_responses"] == 2
    assert result["excluded_unbounded_responses"] == 1
    assert result["exact_agreement"]["mean"] == .5
    assert result["partial_credit_agreement"]["mean"] == .875
    assert result["normalized_disagreement"]["mean"] == .125
    assert result["weighted_exact_disagreement"]["mean"] == .5
    for source, opposite in [("exact_agreement", "weighted_exact_disagreement"),
                             ("partial_credit_agreement", "normalized_disagreement")]:
        assert result[opposite]["lo"] == 1 - result[source]["hi"]
        assert result[opposite]["hi"] == 1 - result[source]["lo"]


def test_retest_preserves_person_and_task_weights_for_both_scores():
    rows = [response(exact=1, partial=1)] * 40
    rows += [response(pid="b"), response(group="binary", partial=0, kind="binary"),
             response(pid="b", group="binary", partial=0, kind="binary")]
    result = retest_agreement_comparison(rows, ["a", "b"], ["ratings", "binary"], draws=20)
    assert result["exact_agreement"]["mean"] == .25
    assert result["partial_credit_agreement"]["mean"] == .4375


def test_retest_binary_scores_are_identical_under_both_rules():
    rows = [response(exact=value, partial=value, kind="binary") for value in (0, 1)]
    result = retest_agreement_comparison(rows, ["a"], ["ratings"], draws=20)
    assert result["exact_agreement"] == result["partial_credit_agreement"]


def test_retest_rejects_out_of_scope_people_and_incomplete_coverage():
    with pytest.raises(ValueError, match="outside"):
        retest_agreement_comparison([response(pid="test")], ["a"], ["ratings"], draws=20)
    with pytest.raises(ValueError, match="Missing participant/task"):
        retest_agreement_comparison([response()], ["a", "b"], ["ratings"], draws=20)
    with pytest.raises(ValueError, match="identical coverage"):
        retest_agreement_comparison([response(exact=None)], ["a"], ["ratings"], draws=20)


def test_answer_type_breakdown_keeps_task_weights_and_excludes_unbounded():
    rows = [response(group="pricing", exact=1, partial=1, kind="binary")] * 40
    rows += [response(group="choice", partial=0, kind="binary"), response(),
             response(group="ratings", exact=1, partial=1, kind="bounded_numeric"),
             response(group="height", partial=None, kind="unbounded_numeric")]
    result = retest_by_answer_type(rows, ["a"], draws=20)
    assert set(result) == {"binary", "ordinal", "bounded_numeric"}
    assert result["binary"]["paired_responses"] == 41
    assert result["binary"]["main_task_groups"] == 2
    assert result["binary"]["exact_agreement"]["mean"] == .5
    assert result["binary"]["exact_agreement"] == result["binary"]["partial_credit_agreement"]
    assert result["ordinal"]["partial_credit_agreement"]["mean"] == .75
    assert result["bounded_numeric"]["exact_agreement"]["mean"] == 1.
