"""Audit a completed study, publish concise chapters, and package reproducible files."""
import csv
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from .data import ROOT
from .audit_run import audit, snapshot_records
from .report_experiment import plot_results, EXPERIMENTS
from .report_format import number_sections, validate_links, replace_section
from .selection import selection_metric


EFFECTS = (
    ("E1", "frozen_full_history minus frozen_question_only"),
    ("E1", "frozen_full_history minus frozen_demographics"),
    ("E1", "frozen_full_history minus frozen_shuffled"),
    ("E1", "sft_full_history minus sft_shuffled"),
    ("E2", "sft_question_only minus frozen_question_only"),
    ("E2", "sft_full_history minus frozen_full_history"),
    ("E2", "sft_full_history minus sft_question_only"),
    ("E2", "sft_full_history minus price_aware"),
    ("E3", "frozen_hybrid minus frozen_full_history"),
    ("E3", "frozen_hybrid minus frozen_random"),
    ("E3", "frozen_hybrid minus frozen_bm25"),
    ("E3", "frozen_hybrid minus frozen_semantic"),
    ("E3", "sft_hybrid minus sft_full_history"),
)
LABELS = {"common_answer": "Common answer / median", "price_aware": "Price-aware baseline",
          "human_retest": "Human retest", "question_only": "question only",
          "demographics": "demographics", "full_history": "full history",
          "shuffled": "shuffled history", "random": "random retrieval",
          "bm25": "BM25", "semantic": "semantic", "hybrid": "hybrid"}


def label(name):
    for prefix, title in (("frozen_", "Frozen"), ("sft_", "SFT")):
        if name.startswith(prefix):
            return title+" / "+LABELS[name.removeprefix(prefix)]
    return LABELS.get(name, name.replace("_", " "))


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point(summary, field, percent=False):
    result = summary.get(field)
    return f"{result['mean']*(100 if percent else 1):.{2 if percent else 3}f}" if result else "—"


def ranked_cell(stages, names, name, field, percent=False):
    """Rank displayed means within a table; human retest is a reference only."""
    scale, precision = (100, 2) if percent else (1, 3)
    result = stages[name].get(field)
    if not result:
        return "—"
    score = round(result['mean']*scale, precision)
    eligible = [n for n in names if not n.endswith('human_retest') and stages[n].get(field)]
    ranks = sorted({round(stages[n][field]['mean']*scale, precision) for n in eligible},
                   reverse=field in ('exact', 'agreement'))
    text = f"{score:.{precision}f}"
    if name in eligible:
        rank = ranks.index(score)
        if rank < 2:
            text = '**'+text+'**' if rank == 0 else '<ins>'+text+'</ins>'
    return text


def score_table(stages, names):
    lines = ["| Condition | Exact % ↑ | Partial credit % ↑ | NLL ↓ | Invalid % ↓ |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        cells = [ranked_cell(stages, names, name, field, field != 'nll')
                 for field in ('exact', 'agreement', 'nll')]
        lines.append(f"| {label(name)} | {' | '.join(cells)} | {100*stages[name]['invalid_response_rate']:.2f} |")
    return lines


def final_result_text(run):
    stages, comparisons = run['stages'], run['comparisons']
    partial = selection_metric(run.get('manifest', {}).get('plan', {})) == 'agreement'
    metric = 'agreement' if partial else 'nll'
    metric_name = 'partial credit' if partial else 'NLL'
    def lower(a, b):
        ci = comparisons[a+' minus '+b][metric]
        return ci['lo'] > 0 if partial else ci['hi'] < 0
    def inconclusive(a, b):
        ci = comparisons[a+' minus '+b][metric]
        return ci['lo'] <= 0 <= ci['hi']
    def bounds(name, field, percent=False):
        ci = stages[name][field]
        scale, precision = (100, 2) if percent else (1, 3)
        return f"{ci['lo']*scale:.{precision}f}–{ci['hi']*scale:.{precision}f}" + ('%' if percent else '')
    full, hybrid = timing_for(stages, 'frozen_full_history'), timing_for(stages, 'frozen_hybrid')
    reduction = 100*(1-hybrid['mean_input_tokens']/full['mean_input_tokens'])
    draws = run['training']['question_only']['planned_draws']
    steps = run['spec']['max_updates']
    gain = lower('sft_question_only', 'frozen_question_only') and lower('sft_full_history', 'frozen_full_history')
    no_identity_gain = all(inconclusive(a, b) for a, b in (
        ('frozen_full_history', 'frozen_shuffled'), ('sft_full_history', 'sft_shuffled')))
    lines = ['## Experiment results', '',
        'E1–E3 compare personal history, fine-tuning, and retrieval on the same **309 development participants and 17 tasks**. '
        'These comparisons explain what helps. Development scores then select one eligible Qwen configuration for test evaluation.', '',
        f'Each adapter completed **{steps} optimizer updates**, with an effective batch of {draws//steps} examples '
        f'({draws:,} examples in total). NLL covers the 15 categorical task groups.', '',
        '**Reading the tables:** **Bold** marks the best score and <ins>underline</ins> the second-best for exact match, partial credit, and NLL. '
        'Ties at the displayed precision share a rank. Human retest is an unranked reference. '
        'Highlighting shows numerical rank, not a statistically established difference.', '']
    if run.get('revision'):
        lines += ['**Selection revision:** Partial credit replaces NLL for checkpoint and configuration selection. '
                  'Both adapters peak at update 64 (2,048 examples seen). The original training completed 128 updates. '
                  'Because the original test results had already been examined, the revised test is retrospective.', '']
    lines += ['### E1: Personalization', '',
              "**Question:** Does the correct person's history help beyond absent, demographic, or shuffled histories?", '']
    lines += score_table(stages, ('common_answer', 'price_aware', 'human_retest', 'frozen_question_only',
        'frozen_demographics', 'frozen_full_history', 'frozen_shuffled', 'sft_full_history', 'sft_shuffled'))
    lines += ['', '**Findings**', '',
        '- **Answer:** '+(f'A reliable personalization benefit is not established. Correct and shuffled histories have similar {metric_name}, '
        'both before and after SFT. The small differences could reflect sampling variation.' if no_identity_gain else
        'At least one correct-versus-shuffled comparison differs reliably. The differences below show the direction and size; '
        'a history effect alone does not establish useful personalization.'),
        '- **Interpretation:** '+(f'Full history improves frozen-model {metric_name} over demographics alone, but this is not enough to show '
        'that the model uses information specific to the correct person.' if lower('frozen_full_history', 'frozen_demographics') else
        f'The demographic comparison does not establish better {metric_name} for full history.'),
        '- **Limit:** Shuffled histories average three donor assignments. Donors are not matched on demographics. '
        'Human retest has access to earlier target answers that the models do not receive.', '',
        '### E2: Fine-tuning', '',
        '**Question:** Does SFT improve prediction, and does history add value beyond learning common answers?', '']
    lines += score_table(stages, ('price_aware', 'frozen_question_only', 'frozen_full_history', 'sft_question_only', 'sft_full_history'))
    q_gain = (100*(stages['sft_question_only']['agreement']['mean']-stages['frozen_question_only']['agreement']['mean']) if partial else
              100*(1-stages['sft_question_only']['nll']['mean']/stages['frozen_question_only']['nll']['mean']))
    h_gain = (100*(stages['sft_full_history']['agreement']['mean']-stages['frozen_full_history']['agreement']['mean']) if partial else
              100*(1-stages['sft_full_history']['nll']['mean']/stages['frozen_full_history']['nll']['mean']))
    relative = -comparisons['sft_full_history minus price_aware']['nll']['mean']/stages['price_aware']['nll']['mean']
    accepted = relative >= .02 and comparisons['sft_full_history minus price_aware']['nll']['hi'] < 0
    lines += ['', '**Findings**', '',
        '- **Answer:** '+((f'Yes. SFT raises partial credit by {q_gain:.2f} percentage points with question-only input and '
        f'{h_gain:.2f} points with full history.' if partial else
        f'Yes. SFT lowers NLL by {q_gain:.1f}% with question-only input and {h_gain:.1f}% with full history.')
        + ' Both gains are supported by paired comparisons.' if gain else
        f'The paired comparisons do not establish a {metric_name} improvement for both adapters.'),
        '- **Value of history:** '+(f'Full-history and question-only SFT have similar {metric_name}; the paired interval includes zero. '
        'The numerical ranking does not establish a personalization benefit.' if inconclusive('sft_full_history', 'sft_question_only') else
        f'The paired comparison establishes a difference in {metric_name}. The shuffled control tests whether this reflects the correct person’s history.'),
        '- **Baseline comparison:** '+('Full-history SFT remains below the price-aware baseline on development partial credit.'
        if stages['sft_full_history']['agreement']['mean'] < stages['price_aware']['agreement']['mean'] else
        'Full-history SFT exceeds the price-aware baseline’s development partial-credit point estimate.')
        + (f" The original, secondary target of at least 2% lower NLL with a paired interval below zero {'is' if accepted else 'is not'} met."), '',
        '### E3: Retrieval', '',
        '**Question:** Can selected evidence preserve quality with less input, beyond random shortening?', '',
        '**Frozen-model comparison**', '']
    lines += score_table(stages, ('frozen_full_history', 'frozen_random', 'frozen_bm25', 'frozen_semantic', 'frozen_hybrid'))
    lines += ['', '**Same SFT adapter, different inputs**', '']
    lines += score_table(stages, ('sft_full_history', 'sft_hybrid'))
    best = min(('frozen_bm25', 'frozen_semantic', 'frozen_hybrid'), key=lambda n: stages[n]['nll']['mean'])
    best_agreement = max(('frozen_bm25', 'frozen_semantic', 'frozen_hybrid'), key=lambda n: stages[n]['agreement']['mean'])
    ranking = (f"{LABELS[best.removeprefix('frozen_')]} has the lowest NLL and highest partial-credit point estimates among the three retrievers. "
               if best == best_agreement else
               f"{LABELS[best.removeprefix('frozen_')]} has the lowest NLL point estimate; "
               f"{LABELS[best_agreement.removeprefix('frozen_')]} has the highest partial credit. ")
    full_better = all(lower(b, a) if b+' minus '+a in comparisons else
                      (comparisons[a+' minus '+b][metric]['hi'] < 0 if partial else
                       comparisons[a+' minus '+b][metric]['lo'] > 0)
                      for a, b in (('frozen_hybrid', 'frozen_full_history'), ('sft_hybrid', 'sft_full_history')))
    retrieval_effect = comparisons['sft_hybrid minus sft_full_history'][metric]
    sft_retrieval = ('improves' if lower('sft_hybrid', 'sft_full_history') else
                     'reduces' if not inconclusive('sft_hybrid', 'sft_full_history') else 'does not reliably change')
    if partial and all(comparisons['frozen_hybrid minus '+n]['agreement']['hi'] < 0
                       for n in ('frozen_bm25', 'frozen_random')) and inconclusive('frozen_hybrid', 'frozen_semantic'):
        retrieval_finding = ('Paired partial-credit comparisons favor BM25 and length-matched random retrieval over hybrid. '
                             'Hybrid and semantic retrieval do not differ clearly.')
    elif lower('frozen_hybrid', 'frozen_semantic') and inconclusive('frozen_hybrid', 'frozen_bm25') and inconclusive('frozen_hybrid', 'frozen_random'):
        retrieval_finding = (f'Hybrid improves {metric_name} over semantic retrieval, but does not show a reliable '
                             'advantage over BM25 or length-matched random selection.')
    else:
        retrieval_finding = f'The numerical ranking must be interpreted alongside the paired {metric_name} comparisons.'
    lines += ['', '**Findings**', '',
        f'- **Answer:** Hybrid retrieval reduces mean input length by {reduction:.1f}%. '
        + (f'It reduces {metric_name} quality relative to full history for both frozen and SFT models.' if full_better else
           f'For SFT, it {sft_retrieval} {metric_name} relative to full history.')
        + (f" The SFT partial-credit difference is {100*retrieval_effect['mean']:+.2f} percentage points."
           if partial else ''),
        '- **Retriever choice:** '+ranking+retrieval_finding,
        '- **Limit:** The study tests one retrieval size and encoder. The SFT adapter was trained on full history, '
        'so these results do not establish how an adapter trained specifically for retrieved input would perform.']
    match = run.get('publication_length_matching')
    if match:
        lines += ['', f"**Random-control check:** Item counts match hybrid retrieval. Token lengths are within 2% in "
                  f"{100*match['fraction_within_2_percent']:.1f}% of draws; the largest mismatch is {100*match['maximum_relative_error']:.1f}%."]
    selected = run['selection']['method']
    lines += ['', '### From development results to the final test', '',
        'The experiments compare input and training choices. Selection uses development scores only; test results do not enter the ranking.', '',
        '1. **Eligible models:** Six Qwen configurations enter selection: frozen and SFT models, each with question-only, full-history, or hybrid input. '
        'Demographics, shuffled histories, random retrieval, BM25, and semantic retrieval serve as controls. Statistical baselines remain separate references.',
        f"2. **Selection:** {'The highest development partial credit' if partial else 'The lowest uncalibrated development NLL'} selects **{label(selected)}** "
        f"({point(stages[selected], metric, partial)}{'%' if partial else ' NLL'}). Scores use unrounded means; exact ties retain the listed candidate order. "
        'This numerical choice does not prove superiority over every alternative.', 
        f"3. **Final test:** A probability temperature of **{run['selection']['calibration']['temperature']:.3f}** is fitted on development data. "
        'Temperature is fitted by NLL to improve probabilities; it does not change predicted answers or partial credit. '
        + ('The revised choice is fixed before this new evaluation on the same 309 test participants. Their original test results were already known.'
           if run.get('revision') else 'The choices are fixed before opening the 309-person reserved test.'), '',
        '**Decision:** '+('The selected Qwen model still has lower development partial credit than the price-aware baseline. '
        'It is the best eligible Qwen configuration by the selection rule, not the best predictor overall.'
        if stages[selected]['agreement']['mean'] < stages['price_aware']['agreement']['mean'] else
        'The selected Qwen model is evaluated alongside the statistical baselines on the reserved test.'), '',
        '![Development model comparisons with participant intervals](figures/13_model_results.png)', '',
        '**Evidence behind the findings:** Paired comparisons score the same people under both conditions. '
        'A 95% interval containing zero does not establish a difference. Intervals are exploratory and are not adjusted for multiple comparisons. '
        'The complete [paired comparisons](../results/paired_comparisons.csv) and [score intervals](../results/model_scores.csv) remain available for inspection.', '',
        '### Retrospective test result' if run.get('revision') else '### Reserved-test result', '',
        '**Question:** Does the selected Qwen configuration outperform the statistical baselines on new participants?', '',
        '| Method | Exact % ↑ | Partial credit % ↑ | NLL ↓ |',
        '| --- | ---: | ---: | ---: |']
    names = ('test_common_answer', 'test_price_aware', 'test_human_retest', 'test_selected_uncalibrated', 'test_selected_calibrated')
    for name in names:
        base = name.removeprefix('test_')
        title = label(base) if not base.startswith('selected_') else (
            'Selected model, calibrated' if base == 'selected_calibrated' else 'Selected model, uncalibrated')
        cells = [ranked_cell(stages, names, name, field, field != 'nll') for field in ('exact', 'agreement', 'nll')]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    test, baseline = stages['test_selected_calibrated'], stages['test_price_aware']
    below_baseline = (test['nll']['mean'] > baseline['nll']['mean'] and
                      all(test[f]['mean'] < baseline[f]['mean'] for f in ('exact', 'agreement')))
    lines += ['', '**Findings**', '',
        '- **Answer:** '+('No. The price-aware baseline has better observed exact match, partial credit, and NLL. '
        'The selected Qwen model has not surpassed this simple reference under the available training budget.' if below_baseline else
        'The table compares the selected model with both statistical references; metric-specific results determine the conclusion.'),
        f"- **Calibration:** Temperature scaling changes NLL from {point(stages['test_selected_uncalibrated'], 'nll')} "
        f"to {point(test, 'nll')} without changing the predicted answers. Exact match and partial credit therefore remain unchanged. "
        f"The selected model has {100*test['invalid_response_rate']:.2f}% invalid responses.", '',
        f"**Uncertainty:** The calibrated model's 95% intervals are {bounds('test_selected_calibrated', 'exact', True)} for exact match, "
        f"{bounds('test_selected_calibrated', 'agreement', True)} for partial credit, and {bounds('test_selected_calibrated', 'nll')} for NLL. "
        'These bootstrap intervals can be asymmetric, so they are retained as ranges rather than a symmetric ± margin. '
        'Full intervals are in [model scores](../results/model_scores.csv).', '',
        '[Task scores](../results/task_scores.csv) cover all 17 groups. [Numerical errors](../results/numeric_errors.csv) retain the original units.', '',
        '### Quality and compute', '',
        'These times cover the full development set with frozen Qwen. Random-control times average three runs. '
        '**Total time includes model inference, history selection, and question embedding.** The last two columns break down that total; they are not extra costs.', '',
        '| Input | Mean input tokens | Total time (min) | History selection (min) | Question embedding (min) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ('frozen_full_history', 'frozen_random', 'frozen_bm25', 'frozen_semantic', 'frozen_hybrid'):
        t = timing_for(stages, name)
        lines.append(f"| {LABELS[name.removeprefix('frozen_')]} | {t['mean_input_tokens']:,.0f} | "
                     f"{t['total_prediction_seconds']/60:.2f} | {t['retrieval_seconds']/60:.2f} | {t['query_encoding_seconds']/60:.2f} |")
    lines += ['', '- **History selection:** Time spent selecting and assembling the history items, including cache reads and token-length matching for random inputs.',
        '- **Question embedding:** Time spent converting target questions into vectors for semantic matching. Cached encoding costs are included. '
        'BM25 does not require question embeddings.', '',
        f"**Finding:** Hybrid retrieval cuts input length by {reduction:.1f}%, but total time changes only from "
        f"{full['total_prediction_seconds']/60:.2f} to {hybrid['total_prediction_seconds']/60:.2f} minutes. "
        'Prefix caching reduces repeated full-history computation, while retrieval adds selection and embedding work.', '',
        '**Timing scope:** Warmup, model loading, and one-time history embedding are excluded from this table. '
        'These are measurements of this implementation, not general hardware benchmarks. '
        '[Compute records](../results/compute.csv) retain the separate history-embedding cost.', '',
        '### Limits and next experiments', '',
        f'- **Training budget:** Each adapter processed {draws:,} examples. The completed run is one pass through the selected subset, '
        'not an epoch over all 90,720 available examples. Partial credit peaks at update 64, while NLL continues to improve through update 128. '
        'The metrics favor different checkpoints; more training does not guarantee better agreement.',
        '- **Further training:** A follow-up could use more examples and stop after a predefined number of development checks without improvement. '
        'The best checkpoint would be retained. Numerical divergence would indicate a training failure, not a desired stopping point.',
        '- **Larger models:** More capacity may help, but this was not tested. Any gains would need to justify the additional memory, latency, and cost.',
        '- **Generalization:** One model size and one training seed limit the conclusions. The three control seeds vary history assignments, not model training. '
        'Further tuning requires a fresh final test, and aggregate scores do not establish individual fidelity.', '']
    return lines


def timing_for(stages, name):
    if "timing" in stages[name]:
        return stages[name]["timing"]
    runs = [stages[f"{name}_{seed}"]["timing"] for seed in (42, 43, 44)]
    return {key: sum(r[key] for r in runs)/len(runs) for key in
            ("mean_input_tokens", "total_prediction_seconds", "retrieval_seconds", "query_encoding_seconds")}


def random_length_diagnostics(run_directory):
    errors = []
    for seed in (42, 43, 44):
        path = run_directory / f"frozen_random_{seed}/predictions.jsonl"
        if path.exists():
            errors.extend(r["actual_history_length_relative_error"] for r in snapshot_records(path))
    if not errors:
        return None
    return {"mean_relative_error": sum(errors)/len(errors), "maximum_relative_error": max(errors),
            "fraction_within_2_percent": sum(e <= .02 for e in errors)/len(errors)}


def package_submission():
    """An allowlist excludes source data, weights, diagnostics, credentials and Git history."""
    required = ("README.md", "requirements.txt", "requirements-training.txt", "pytest.ini",
                "configs/model_plan.json", "configs/runtime_profile.json", "splits/participants.json")
    missing = [name for name in required if not (ROOT/name).is_file()]
    chapters = sorted((ROOT/'docs').glob('0[1-6]_*.md'))
    if missing or [p.name[:2] for p in chapters] != ['01', '02', '03', '04', '05', '06']:
        raise ValueError(f"Submission is missing required reproduction files or chapters: {missing}")
    files = [ROOT/name for name in (*required, ".gitignore")]
    for directory, pattern in (("docs", "0[1-6]_*.md"), ("docs/figures", "*.png"),
                               ("configs", "*.json"), ("splits", "*.json"), ("results", "*.csv"),
                               ("src/twinlab", "*.py"), ("tests", "*.py")):
        files.extend((ROOT/directory).glob(pattern))
    files = sorted({p for p in files if p.is_file()})
    validate_links(files)
    output = ROOT / "dist/submission.zip"
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    temporary.replace(output)
    return output


def finalize(run_directory):
    run_directory = Path(run_directory)
    run = json.loads((run_directory / "report/experiment.json").read_text())
    if run["status"] != "complete" or not run["final_test_opened"]:
        raise ValueError("A submission cannot be finalized before reserved-test evaluation completes")
    if run["coverage"]["task_groups"] != 17 or run["coverage"]["dev_participants"] != 309:
        raise ValueError("Final submission requires complete task/development coverage")
    if run.get('revision'):
        from .reselect_experiment import verify_source, file_hash
        source = ROOT/'checkpoints'/run['revision']['source_run']
        verify_source(source, run['revision']['source_sha256'])
        for mode, training in run['training'].items():
            if not training['selected_adapter_available']:
                continue
            adapter = f"sft_{mode}/update-{training['selected_update']}-adapter"
            if not (source/adapter).exists():
                adapter = f'sft_{mode}/best_adapter'
            expected = run['revision']['source_sha256'][adapter+'/adapter_model.safetensors']
            if file_hash(run_directory/f'sft_{mode}/best_adapter/adapter_model.safetensors') != expected:
                raise ValueError('Revised adapter weights differ from the selected source checkpoint')
    stages = run["stages"]
    required = {name for _, _, names in EXPERIMENTS for name in names}
    required.update({"test_common_answer", "test_price_aware", "test_human_retest", "test_selected_uncalibrated", "test_selected_calibrated"})
    required.update(f"{mode}_{seed}" for mode in ('frozen_random', 'frozen_shuffled', 'sft_shuffled')
                    for seed in (42, 43, 44))
    missing = required-stages.keys()
    if missing:
        raise ValueError(f"Cannot finalize incomplete E1–E3/test results: {sorted(missing)}")
    if any(name not in run["comparisons"] for _, name in EFFECTS):
        raise ValueError("Cannot finalize missing paired experiment comparisons")
    for name in required:
        s = stages[name]
        if s['participants'] != 309 or len(s['per_task']) != 17:
            raise ValueError(f"Incomplete coverage in {name}")
        if not 0 <= s['exact']['mean'] <= s['agreement']['mean'] <= 1:
            raise ValueError(f"Inconsistent agreement scores in {name}")
    checked = audit(run_directory, include_test=True)
    if any(not r['complete'] for group in ('development_predictions', 'test_predictions') for r in checked[group].values()):
        raise ValueError("Final source audit found incomplete predictions")
    (run_directory / 'report/final-audit.json').write_text(json.dumps(checked, indent=2)+'\n')
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    rows, task_rows, numeric_rows, compute_rows = [], [], [], []
    length_matching = random_length_diagnostics(run_directory)
    run["publication_length_matching"] = length_matching
    for name, summary in stages.items():
        record = {"condition": name, "participants": summary["participants"], "responses": summary["responses"]}
        for field in ("exact", "agreement", "nll", "brier"):
            for part in ("mean", "lo", "hi"):
                record[f"{field}_{part}"] = summary[field][part] if summary.get(field) else ""
        record["invalid_response_rate"] = summary["invalid_response_rate"]
        rows.append(record)
        for group, metrics in summary["per_task"].items():
            task_rows.append({"condition": name, "task": group, **{f: metrics.get(f, "") for f in ("exact", "agreement", "nll", "brier", "invalid")}})
        for item, error in summary.get('numeric_errors', {}).items():
            numeric_rows.append({'condition': name, 'item': item, **{k: error[k] for k in ('valid', 'attempted', 'mae', 'median_absolute_error')}})
        if 'timing' in summary or name in ('frozen_random', 'frozen_shuffled', 'sft_shuffled'):
            t = timing_for(stages, name)
            compute_rows.append({'condition': name, **{k: t[k] for k in ('mean_input_tokens', 'total_prediction_seconds', 'retrieval_seconds', 'query_encoding_seconds')},
                                 'one_time_history_encoding_seconds': '',
                                 **(length_matching if name == 'frozen_random' and length_matching else {})})
    for split in ('dev', 'test'):
        path = run_directory / f'retrieval-{split}/metadata.json'
        if path.exists():
            metadata = json.loads(path.read_text())
            compute_rows.append({'condition': f'cached_{split}_history_embeddings', 'mean_input_tokens': '',
                                 'total_prediction_seconds': '', 'retrieval_seconds': '', 'query_encoding_seconds': '',
                                 'one_time_history_encoding_seconds': metadata['total_history_encoding_seconds']})
    write_csv(results / "model_scores.csv", rows)
    write_csv(results / "task_scores.csv", task_rows)
    write_csv(results / "numeric_errors.csv", numeric_rows, ['condition', 'item', 'valid', 'attempted', 'mae', 'median_absolute_error'])
    write_csv(results / "compute.csv", compute_rows,
              ['condition', 'mean_input_tokens', 'total_prediction_seconds', 'retrieval_seconds',
               'query_encoding_seconds', 'one_time_history_encoding_seconds', 'mean_relative_error',
               'maximum_relative_error', 'fraction_within_2_percent'])
    comparisons = [{"comparison": name, "metric": field, **{k: interval[k] for k in ("mean", "lo", "hi")}}
                   for name, metrics in run["comparisons"].items() for field, interval in metrics.items()]
    write_csv(results / "paired_comparisons.csv", comparisons)
    curves = []
    for name, training in run['training'].items():
        checkpoints = {c['update']: c['development'] for c in training['checkpoints']}
        for row in training['loss_curve']:
            curves.append({'condition': name, **{k: row[k] for k in ('update', 'answer_loss', 'gradient_norm', 'learning_rate')},
                           **row.get('runtime', {'microbatch': 1, 'cpu_threads': 8, 'checkpointing': True}),
                           'development_nll': checkpoints.get(row['update'], {}).get('nll', {}).get('mean', ''),
                           'development_partial_credit': checkpoints.get(row['update'], {}).get('agreement', {}).get('mean', '')})
    write_csv(results / "training_curves.csv", curves)
    plot_results(run_directory / 'report', run)
    shutil.copyfile(run_directory / 'report/comparison.png', ROOT / 'docs/figures/13_model_results.png')
    plot_training(run)
    doc = ROOT / "docs/04_evaluation.md"
    old = doc.read_text()
    doc.write_text(replace_section(old, "Experiment result", "LLM judges", '\n'.join(final_result_text(run))))
    lines = ["## Completed run", "", "| Adapter | Updates | Selected | Samples | Processed tokens | Training hours | Peak GiB |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in run['training'].items():
        lines.append(f"| {LABELS[name]} | {r['updates']} | {r['selected_update']} | {r['planned_draws']:,} | {r['processed_tokens']:,} | {r['training_seconds']/3600:.3f} | {r['peak_allocated_gib']:.2f} |")
    lines += ["", "- **Timing scope:** Training times exclude evaluation and setup. CPU throttling affected training and was corrected during evaluation. "
              "These times are not normal-speed hardware benchmarks.",
              "- **Learning progress:** Both adapters peak on development partial credit at update 64, while NLL reaches its lowest value at update 128. "
              + ("The selected update-64 checkpoints have seen 2,048 examples each (2.26% of the full pool), covering all 1,440 people and 17 tasks. "
                 if run.get('revision') else '')
              + "The fixed budget does not establish convergence. [Training curves](../results/training_curves.csv) retain every update and checkpoint value.", "",
              "![Training losses and development checkpoint scores](figures/14_training_curves.png)", ""]
    doc = ROOT / 'docs/06_reproducibility.md'
    old = doc.read_text()
    heading = 'Training time' if re.search(r'(?m)^## (?:\d+(?:\.\d+)+ )?Training time', old) else 'Completed run'
    if re.search(rf'(?m)^## (?:\d+(?:\.\d+)+ )?{heading}', old):
        old = replace_section(old, heading, 'Evaluation and run records', '\n'.join(lines))
    else:
        old += '\n'+'\n'.join(lines)
    doc.write_text(old)
    doc = ROOT / 'README.md'
    old = doc.read_text()
    if run.get('revision'):
        old = re.sub(r'\*\*Evaluation update:\*\*[^\n]*\n\n', '', old)
    marker = '**Current status:**' if '**Current status:**' in old else '**Study summary:**'
    a = old.index(marker)
    b = old.index('\n\n', a)
    old = old[:a]+"**Study summary:** Two Qwen3.5-0.8B adapters and three experiments examine personalization, fine-tuning, and retrieval. "\
        "Each adapter used 4,096 training examples (4.51% of a full pass). "\
        "[Results](docs/04_evaluation.md) distinguish fine-tuning gains from evidence of personalization; "\
        "[chapter 06](docs/06_reproducibility.md) gives the complete reproduction steps."+old[b:]
    doc.write_text(old)
    number_sections(ROOT)
    print(f"Final submission: {package_submission()}", flush=True)


def plot_training(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .figures import style, header, save, BLUE, TEAL, GREY
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.09, right=.96, top=.77, bottom=.27, wspace=.35)
    header(fig, "Training and checkpoint performance",
           f"Two LoRA adapters · {run['spec']['max_updates']} optimizer updates each · lower loss and higher partial credit are better")
    for name, training in run["training"].items():
        color, line, marker = (BLUE, "-", "o") if name == "question_only" else (TEAL, "--", "s")
        label = "Question-only SFT" if name == "question_only" else "Full-history SFT"
        curve = training["loss_curve"]
        y = np.array([p["answer_loss"] for p in curve])
        window = min(8, len(y))
        axes[0].plot([p["update"] for p in curve][window-1:],
                     np.convolve(y, np.ones(window)/window, mode="valid"),
                     color=color, linestyle=line, lw=2, label=label)
        checks = training["checkpoints"]
        axes[1].plot([p["update"] for p in checks], [100*p["development"]["agreement"]["mean"] for p in checks],
                     color=color, linestyle=line, marker=marker, ms=6, lw=2, label=label)
    axes[0].set_title("Training loss", loc="left", pad=12)
    axes[1].set_title("Development checkpoints", loc="left", pad=12)
    axes[0].set_ylabel("Answer-token loss")
    axes[1].set_ylabel("Partial credit (%)")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(axis="y", color="#dce3eb", alpha=.6)
        ax.set_axisbelow(True)
        ax.set_xticks(sorted({0, *(c['update'] for t in run['training'].values() for c in t['checkpoints'])}))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .12),
               ncol=2, frameon=False, fontsize=10, handlelength=3)
    fig.text(.05, .065, "Training curves show an 8-update moving mean. Development partial credit gives equal weight to all 17 task groups.",
             fontsize=10, color=GREY)
    selected = "; ".join(f"{LABELS[n]}: update {t['selected_update']}" for n, t in run['training'].items())
    fig.text(.05, .025, f"Selected checkpoints — {selected}. The fixed budget does not establish convergence.", fontsize=10, color=GREY)
    save(fig, "14_training_curves.png", directory=ROOT / "docs/figures")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    finalize(parser.parse_args().run_directory)
