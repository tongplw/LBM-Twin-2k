"""Publication tests use synthetic fixtures only and write exclusively to tmp_path."""
import copy
import json
from zipfile import ZipFile

import pytest

from twinlab import finalize_submission as publication


def interval(mean):
    return {"mean": mean, "lo": mean-.01, "hi": mean+.01}


def fixture_run():
    summary = {"participants": 309, "responses": 30000, "exact": interval(.5),
               "agreement": interval(.75), "nll": interval(.8), "brier": interval(.4),
               "invalid_response_rate": 0., "numeric_errors": {},
               "per_task": {f"Task {i}": {"exact": .5, "agreement": .75, "nll": .8} for i in range(17)},
               "timing": {"mean_input_tokens": 100, "total_prediction_seconds": 20,
                          "retrieval_seconds": 1, "query_encoding_seconds": 2}}
    names = ["common_answer", "price_aware", "human_retest", "frozen_question_only", "frozen_demographics",
             "frozen_full_history", "frozen_shuffled", "sft_question_only", "sft_full_history", "sft_shuffled",
             "frozen_random", "frozen_bm25", "frozen_semantic", "frozen_hybrid", "sft_hybrid",
             "test_common_answer", "test_price_aware", "test_human_retest", "test_selected_uncalibrated", "test_selected_calibrated"]
    names += [f"{mode}_{seed}" for mode in ("frozen_random", "frozen_shuffled", "sft_shuffled") for seed in (42, 43, 44)]
    stages = {n: copy.deepcopy(summary) for n in names}
    training = {"updates": 128, "selected_update": 32, "planned_draws": 4096,
                "processed_tokens": 1000, "training_seconds": 10, "peak_allocated_gib": 5,
                "checkpoints": [{"update": 32, "development": summary}],
                "loss_curve": [{"update": 1, "answer_loss": 1., "gradient_norm": .5, "learning_rate": 1e-4}]}
    comparisons = {name: {f: interval(0.) for f in ("exact", "agreement", "nll")} for _, name in publication.EFFECTS}
    return {"status": "complete", "final_test_opened": True,
            "coverage": {"task_groups": 17, "dev_participants": 309},
            "spec": {"max_updates": 128}, "stages": stages, "comparisons": comparisons,
            "selection": {"method": "sft_question_only", "calibration": {"temperature": 1.}},
            "training": {"question_only": copy.deepcopy(training), "full_history": copy.deepcopy(training)}}


def test_publication_rejects_incomplete_run(tmp_path):
    run = tmp_path / "report"
    run.mkdir()
    (run / "experiment.json").write_text(json.dumps({"status": "running", "final_test_opened": False}))
    with pytest.raises(ValueError, match="before reserved-test"):
        publication.finalize(tmp_path)


def test_complete_submission_contains_final_tables_only(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    monkeypatch.setattr(publication, "audit", lambda *args, **kwargs: {
        "development_predictions": {"example": {"complete": True}}, "test_predictions": {"final-test": {"complete": True}}})
    for name in ("requirements.txt", "requirements-training.txt", "pytest.ini", "configs/model_plan.json",
                 "configs/runtime_profile.json", "splits/participants.json"):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    (tmp_path / "docs/figures").mkdir(parents=True)
    (tmp_path / "docs/04_evaluation.md").write_text("# Evaluation\n\n## Experiment results\nPending\n## LLM judges\nExtension\n")
    (tmp_path / "docs/06_reproducibility.md").write_text(
        "# Reproduction\n\n## Completed run\nPending\n\n## Evaluation and run records\nRecords\n")
    for number in (1, 2, 3, 5):
        (tmp_path / f"docs/{number:02d}_chapter.md").write_text("# Chapter\n")
    (tmp_path / "docs/07_execution.md").write_text("Private live report")
    (tmp_path / ".env").write_text("DO_NOT_PACKAGE=private")
    (tmp_path / "README.md").write_text("# Fixture\n\n**Current status:** pending\n\nEnd\n")
    root = tmp_path / "checkpoints/fixture"
    (root / "report").mkdir(parents=True)
    (root / "report/experiment.json").write_text(json.dumps(fixture_run()))
    publication.finalize(root)
    report = (tmp_path / "docs/04_evaluation.md").read_text()
    assert "Pending" not in report and "50.00" in report and "75.00" in report
    assert "**128 optimizer updates**" in report
    assert all(f"E{i}:" in report for i in (1, 2, 3))
    assert "### 4.1.1 E1:" in report
    assert len(list((tmp_path / "results").glob("*.csv"))) == 6
    scores = (tmp_path / "results/model_scores.csv").read_text()
    assert all(f"sft_shuffled_{seed}," in scores for seed in (42, 43, 44))
    assert not list((tmp_path / "results").glob("*.json"))
    with ZipFile(tmp_path / "dist/submission.zip") as archive:
        names = archive.namelist()
        assert 'docs/04_evaluation.md' in names and 'results/model_scores.csv' in names
        assert 'docs/07_execution.md' not in names and '.env' not in names
        assert not any(n.startswith(('checkpoints/', 'data/')) for n in names)
    publication.finalize(root)  # idempotent, with no duplicated run section
    assert (tmp_path / "docs/06_reproducibility.md").read_text().count("Completed run") == 1


def test_publication_requires_all_three_experiments(tmp_path):
    (tmp_path / "report").mkdir()
    run = fixture_run()
    del run["stages"]["frozen_hybrid"]
    (tmp_path / "report/experiment.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="incomplete E1"):
        publication.finalize(tmp_path)


def test_publication_rejects_missing_control_seed(tmp_path):
    (tmp_path / "report").mkdir()
    run = fixture_run()
    del run["stages"]["frozen_random_43"]
    (tmp_path / "report/experiment.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="incomplete E1"):
        publication.finalize(tmp_path)


def test_publication_rejects_a_partial_cohort(tmp_path):
    (tmp_path / "report").mkdir()
    run = fixture_run()
    run["stages"]["test_selected_calibrated"]["participants"] = 308
    (tmp_path / "report/experiment.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Incomplete coverage"):
        publication.finalize(tmp_path)


def test_random_control_diagnostics_disclose_length_mismatch(tmp_path):
    for seed, error in ((42, .01), (43, .02), (44, .08)):
        path = tmp_path / f"frozen_random_{seed}/predictions.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps({"actual_history_length_relative_error": error})+"\n")
    result = publication.random_length_diagnostics(tmp_path)
    assert result['fraction_within_2_percent'] == pytest.approx(2/3)
    assert result['maximum_relative_error'] == .08


def test_ranked_scores_respect_direction_ties_and_human_reference():
    stages = fixture_run()['stages']
    names = ('price_aware', 'sft_question_only', 'sft_full_history', 'human_retest')
    for name, exact, nll in zip(names, (.5, .4, .40001, .9), (.9, .7, .8, .1)):
        stages[name]['exact'] = interval(exact)
        stages[name]['nll'] = interval(nll)
    assert publication.ranked_cell(stages, names, 'price_aware', 'exact', True) == '**50.00**'
    assert publication.ranked_cell(stages, names, 'sft_question_only', 'exact', True) == '<ins>40.00</ins>'
    assert publication.ranked_cell(stages, names, 'sft_full_history', 'exact', True) == '<ins>40.00</ins>'
    assert publication.ranked_cell(stages, names, 'human_retest', 'exact', True) == '90.00'
    assert publication.ranked_cell(stages, names, 'sft_question_only', 'nll') == '**0.700**'
    assert publication.ranked_cell(stages, names, 'sft_full_history', 'nll') == '<ins>0.800</ins>'


def test_revised_findings_use_agreement_and_disclose_test_reuse():
    from twinlab.selection import AGREEMENT_RULE
    run = fixture_run()
    run['revision'] = {'retrospective': True}
    run['manifest'] = {'plan': {'selection_metric': AGREEMENT_RULE}}
    run['selection']['method'] = 'sft_full_history'
    for a, b in [('sft_question_only', 'frozen_question_only'), ('sft_full_history', 'frozen_full_history')]:
        run['stages'][a]['agreement']['mean'] = .71
        run['stages'][b]['agreement']['mean'] = .60
        run['comparisons'][a+' minus '+b]['agreement'] = interval(.11)
        # Opposite NLL direction must not reverse the partial-credit finding.
        run['comparisons'][a+' minus '+b]['nll'] = interval(.11)
    text = '\n'.join(publication.final_result_text(run))
    assert 'SFT raises partial credit by 11.00 percentage points' in text
    assert 'highest development partial credit' in text
    assert 'Retrospective test result' in text
    assert 'original test results had already been examined' in text
    assert '2,048 examples seen' in text
    assert 'lowest uncalibrated development NLL selects' not in text
    for control in ('frozen_bm25', 'frozen_random'):
        run['comparisons']['frozen_hybrid minus '+control]['agreement'] = interval(-.02)
    text = '\n'.join(publication.final_result_text(run))
    assert 'comparisons favor BM25 and length-matched random retrieval over hybrid' in text
    assert text.index('**Selection revision:**') < text.index('### E1:')
