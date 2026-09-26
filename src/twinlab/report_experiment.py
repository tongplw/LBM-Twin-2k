"""Render aggregate results only; pending stages never become invented numbers."""
import argparse
import json
from pathlib import Path


EXPERIMENTS = (
    ("E1: Personalization", "Does the correct person's history improve prediction?", (
        "common_answer", "price_aware", "human_retest", "frozen_question_only", "frozen_demographics",
        "frozen_full_history", "frozen_shuffled", "sft_full_history", "sft_shuffled")),
    ("E2: Fine-tuning", "Does full-history SFT outperform question-only SFT and the frozen models?", (
        "price_aware", "frozen_question_only", "frozen_full_history", "sft_question_only", "sft_full_history")),
    ("E3: Retrieval", "Can hybrid retrieval preserve quality with less input, beyond random shortening?", (
        "frozen_full_history", "frozen_random", "frozen_bm25", "frozen_semantic", "frozen_hybrid",
        "sft_full_history", "sft_hybrid")),
)


def experiment_tables(stages):
    lines = []
    for title, question, names in EXPERIMENTS:
        lines += [f"### {title}", "", question, "",
            "| Condition | Exact % [95% CI] | Partial credit % [95% CI] | NLL [95% CI] | Brier | Invalid % |",
            "| --- | --- | --- | --- | --- | --- |"]
        for name in names:
            summary = stages.get(name)
            if summary:
                brier = f"{summary['brier']['mean']:.4f}" if summary.get("brier") else "—"
                invalid = f"{summary['invalid_response_rate']*100:.3f}"
                lines.append(f"| {name.replace('_', ' ')} | {value(summary, 'exact', True)} | "
                    f"{value(summary, 'agreement', True)} | {value(summary, 'nll')} | {brier} | {invalid} |")
            else:
                lines.append(f"| {name.replace('_', ' ')} | Pending | Pending | Pending | Pending | Pending |")
        lines.append("")
    return lines


def value(summary, field, percent=False):
    result = summary.get(field)
    if not result:
        return "—"
    scale = 100 if percent else 1
    return f"{result['mean']*scale:.3f} [{result['lo']*scale:.3f}, {result['hi']*scale:.3f}]"


def write_report(directory):
    directory = Path(directory)
    run = json.loads((directory / "experiment.json").read_text())
    stages = run["stages"]
    scope = run["spec"]["scope"]
    lines = ["# Measured experiment results", "", f"Status: **{run['status']}**. Updated: {run['updated']}.", "",
        "All numbers below come from saved model predictions or observed human responses. "
        "Intervals resample participants 2,000 times, preserving all their task responses.", ""]
    coverage = run.get("coverage", {})
    if coverage:
        lines += [f"Coverage: **{coverage['task_groups']} task groups**, "
            f"{coverage['train_participants']} training and {coverage['dev_participants']} development participants; "
            f"{coverage['training_responses']:,} training and {coverage['development_responses']:,} development responses.", ""]
    if scope == "pilot" or coverage.get("train_participants", 0) < 1440:
        lines += ["**This is a limited pilot. It does not establish performance on the complete 17-task, "
                  "1,440-participant training benchmark.**", ""]
    lines += ["Final-test outcomes: " + ("opened only after selection was sealed." if run["final_test_opened"] else "**unopened**."), ""]
    if run.get("error"):
        lines += ["Execution stopped: `"+run["error"]+"`.", ""]
    lines += ["## Development scores", "",
        "Exact and partial-credit agreement use identical point predictions and bounded responses. "
        "Categorical NLL and Brier use the categorical task groups; free numerical estimates are excluded. "
        "Matrix probabilities use earlier model-generated answers from the same question. Human row answers are not supplied. "
        "All three experiments are reported separately below. Shared reference rows reuse the same evaluation. "
        "The selected model's reserved-test result is an additional result, not a replacement for E1–E3.", ""]
    desired = ["common_answer", "price_aware", "human_retest", "frozen_question_only", "frozen_demographics",
        "frozen_full_history", "frozen_shuffled", "frozen_random", "frozen_bm25", "frozen_semantic", "frozen_hybrid",
        "sft_question_only", "sft_full_history", "sft_shuffled", "sft_hybrid", "selected_calibrated_development"]
    lines += experiment_tables(stages)
    lines += ["", "Human retest uses the previous target answer, which is excluded from model inputs. "
              "Its missing NLL/Brier reflect the absence of a probability model. "
              "Development calibration is fitted on these same development answers and is not independent validation.", ""]
    if run.get("comparisons"):
        lines += ["## Paired comparisons", "", "Differences are left minus right. Negative NLL favors the left; "
            "positive agreement favors the left. Repeated controls are averaged within each participant before bootstrapping.", "",
            "| Comparison | Δ exact, percentage points [95% CI] | Δ partial credit, pp [95% CI] | Δ NLL [95% CI] |",
            "| --- | --- | --- | --- |"]
        for name, comparison in run["comparisons"].items():
            lines.append(f"| {name.replace('_', ' ')} | {value(comparison, 'exact', True)} | "
                         f"{value(comparison, 'agreement', True)} | {value(comparison, 'nll')} |")
        e2 = run["comparisons"]["sft_full_history minus sft_question_only"]["nll"]
        shuffled = run["comparisons"]["sft_full_history minus sft_shuffled"]["nll"]
        baseline = run["comparisons"]["sft_full_history minus price_aware"]["nll"]
        if e2["hi"] < 0 and shuffled["hi"] < 0:
            lines += ["", "The development comparisons support a history benefit over question-only SFT and shuffled histories "
                "within the measured scope. Demographic-matched donors were not evaluated, so this does not isolate information beyond demographics."]
        else:
            lines += ["", "These development results do not establish a reliable personalization benefit: "
                "the full-history model does not beat both question-only SFT and shuffled histories with a paired NLL interval wholly below zero."]
        relative = -baseline["mean"]/stages["price_aware"]["nll"]["mean"]
        lines += [f"Full-history SFT's relative NLL improvement over the price-aware reference is {100*relative:.2f}%. "
            + ("It meets" if relative >= .02 and baseline["hi"] < 0 else "It does not meet")
            + " the prespecified 2% improvement plus paired-interval criterion.", ""]
    lines += ["## Compute and retrieval", "",
        "| Condition | Mean input tokens | Prediction seconds | Retrieval + query encoding seconds |",
        "| --- | --- | --- | --- |"]
    for name in desired:
        timing = stages.get(name, {}).get("timing")
        if timing:
            lines.append(f"| {name.replace('_', ' ')} | {timing['mean_input_tokens']:.0f} | "
                f"{timing['total_prediction_seconds']:.1f} | {timing['retrieval_seconds']+timing['query_encoding_seconds']:.1f} |")
    lines += ["", "Prediction time includes input processing and within-question generation. Batch size is one; "
        "one same-condition warmup precedes scoring. Immutable history-prefix states are copied before independent target questions. "
        "Question-only evaluation can reuse deterministic outputs for identical answer-free prompts within the same checkpoint; "
        "each participant is still scored separately, and reused predictions are counted in timing metadata. "
        "History embedding is a one-time cached preparation cost, recorded separately in the private retrieval metadata. "
        "Random controls match item count and target a 2% difference in actual packed history tokens; residuals are recorded.", "",
        "| Training condition | Selected update | Updates | Processed / planned examples | Processed tokens | Training hours | Peak allocated GPU GiB |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, training in run.get("training", {}).items():
        accum = training.get("plan", {}).get("training", {}).get("gradient_accumulation", 32)
        consumed = training["updates"]*accum
        lines.append(f"| {name.replace('_', ' ')} | {training.get('selected_update') or 'Pending'} | {training['updates']} | "
            f"{consumed:,} / {training['planned_draws']:,} | "
            f"{training['processed_tokens']:,} | {training['training_seconds']/3600:.3f} | {training['peak_allocated_gib']:.2f} |")
    lines += ["", "The example counts above are completed optimizer updates × gradient accumulation. "
              "Eligible pool size is not the amount already trained. The 4,096-example plan is 4.51% of "
              "90,720 available person–question examples; it is not one epoch. "
              "A checkpoint under evaluation is not a completed model result.", ""]
    if run.get("selection"):
        selected = run["selection"]
        lines += ["", f"Selected method: **{selected['method']}** using {selected['rule']}. "
                  f"Development-fitted temperature: {selected['calibration']['temperature']:.4f}.", ""]
        summary = stages[selected["method"]]
        reference = stages["price_aware"]
        lines += ["## Selected method by task (development)", "",
            "| Task | Exact % | Partial credit % | NLL | Reference NLL |",
            "| --- | --- | --- | --- | --- |"]
        for group, metrics in summary["per_task"].items():
            nll = f"{metrics['nll']:.3f}" if "nll" in metrics else "—"
            base_nll = reference["per_task"].get(group, {}).get("nll")
            base_nll = f"{base_nll:.3f}" if base_nll is not None else "—"
            lines.append(f"| {group} | {100*metrics.get('exact', 0):.2f} | "
                f"{100*metrics.get('agreement', 0):.2f} | {nll} | {base_nll} |")
        if summary["numeric_errors"]:
            lines += ["", "Numerical errors remain in each question's native units; errors from different units are not pooled.", "",
                "| Numerical item | Valid / attempted | MAE | Median absolute error |",
                "| --- | --- | --- | --- |"]
            for key, errors in summary["numeric_errors"].items():
                mae = f"{errors['mae']:.3f}" if errors["mae"] is not None else "—"
                median = f"{errors['median_absolute_error']:.3f}" if errors["median_absolute_error"] is not None else "—"
                lines.append(f"| {key.replace('|', ' / ')} | {errors['valid']} / {errors['attempted']} | {mae} | {median} |")
    test = {k: v for k, v in stages.items() if k.startswith("test_")}
    if test:
        lines += ["## Reserved test", "", "| Method | Exact % [95% CI] | Partial credit % [95% CI] | NLL [95% CI] |",
                  "| --- | --- | --- | --- |"]
        for name, summary in test.items():
            lines.append(f"| {name} | {value(summary, 'exact', True)} | {value(summary, 'agreement', True)} | {value(summary, 'nll')} |")
    if run["status"] in {"complete", "pilot_complete_final_test_unopened"}:
        plot_results(directory, run)
        lines += ["", "![Measured development comparison with participant bootstrap intervals](comparison.png)", ""]
    lines += ["", "## Reproduction", "", "Model and embedding revisions, software versions, source/data hashes, "
        "seeds, hyperparameters, coverage and per-task results are saved in [experiment.json](experiment.json). "
        "Participant-level predictions, prompts and adapters remain in ignored local storage.", ""]
    temp = directory / "REPORT.md.tmp"
    temp.write_text("\n".join(lines))
    temp.replace(directory / "REPORT.md")


def plot_results(directory, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .figures import style, header, BLUE, TEAL, GREY, NAVY
    style()
    names = [s for s in ("price_aware", "frozen_question_only", "frozen_full_history",
                         "frozen_hybrid", "sft_question_only", "sft_full_history", "sft_hybrid") if s in run["stages"]]
    labels = {"price_aware": "Price-aware baseline", "frozen_question_only": "Frozen · question only",
              "frozen_full_history": "Frozen · full history", "frozen_hybrid": "Frozen · hybrid",
              "sft_question_only": "SFT · question only", "sft_full_history": "SFT · full history",
              "sft_hybrid": "SFT · hybrid"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), sharey=True)
    fig.subplots_adjust(left=.23, right=.96, top=.78, bottom=.23, wspace=.30)
    coverage = run["coverage"]
    header(fig, "How do fine-tuning and history affect prediction?",
           f"Development set · {coverage['dev_participants']} people · {coverage['task_groups']} tasks · 95% participant-bootstrap intervals")
    positions = np.arange(len(names))
    for ax, field, factor, title in zip(axes, ("exact", "agreement", "nll"), (100, 100, 1),
                                       ("Exact match ↑", "Partial credit ↑", "Categorical NLL ↓")):
        stats = [run["stages"][n][field] for n in names]
        values = np.array([s["mean"] for s in stats])*factor
        lower = np.array([s["lo"] for s in stats])*factor
        upper = np.array([s["hi"] for s in stats])*factor
        for i, name in enumerate(names):
            color = TEAL if name.startswith("sft_") else BLUE if name.startswith("frozen_") else GREY
            ax.errorbar(values[i], positions[i],
                        xerr=np.maximum(0, [[values[i]-lower[i]], [upper[i]-values[i]]]),
                        fmt="o", color=color, capsize=3, ms=6, lw=1.5, zorder=3)
            text = f"{values[i]:.2f}%" if factor == 100 else f"{values[i]:.3f}"
            ax.annotate(text, (upper[i], positions[i]), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=9.5, color=NAVY)
        margin = max(float(upper.max()-lower.min())*.16, .02)
        ax.set_xlim(lower.min()-margin, upper.max()+2.5*margin)
        ax.set_title(title, loc="left", pad=14)
        ax.set_xlabel("Agreement (%)" if factor == 100 else "Negative log-likelihood")
        ax.tick_params(axis="y", length=0, pad=12)
        ax.grid(axis="x", color="#dce3eb", alpha=.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, [labels[n] for n in names])
    axes[0].set_ylim(len(names)-.5, -.5)
    fig.text(.05, .085, "SFT = supervised fine-tuning. Higher agreement and lower NLL are better.", fontsize=10, color=GREY)
    fig.text(.05, .045, "Agreement covers 17 task groups; NLL covers the 15 categorical groups. Both agreement scores use the same predictions.",
             fontsize=10, color=GREY)
    fig.savefig(directory / "comparison.png", dpi=180, bbox_inches="tight", pad_inches=.2,
                metadata={"Software": "TwinLab / Matplotlib"})
    fig.savefig(directory / "comparison.svg", bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    write_report(args.directory)


if __name__ == "__main__":
    main()
