"""Explicit development selection rules, preserving historical run protocols."""
import math


NLL_RULE = "development_task_group_macro_categorical_nll_all_15_applicable_groups"
AGREEMENT_RULE = "development_task_group_macro_partial_credit_all_17_groups"


def selection_metric(plan, checkpoint=False):
    rule = plan.get("selection_metric", NLL_RULE)
    if checkpoint:
        rule = plan.get("execution_budget", {}).get("checkpoint_selection", rule)
    aliases = {
        NLL_RULE: "nll", "lowest_development_macro_categorical_nll": "nll",
        AGREEMENT_RULE: "agreement", "highest_development_macro_partial_credit": "agreement",
    }
    if rule not in aliases:
        raise ValueError(f"Unsupported selection rule: {rule}")
    return aliases[rule]


def selection_loss(summary, metric):
    """Lower is better, including the negative of partial-credit agreement."""
    if metric not in {"nll", "agreement"}:
        raise ValueError(f"Unsupported selection metric: {metric}")
    value = summary[metric]["mean"]
    if not math.isfinite(value):
        raise ValueError(f"Non-finite development {metric}")
    return -value if metric == "agreement" else value


def select_best(summaries, metric):
    """Select by unrounded means; exact ties retain the supplied candidate order."""
    if not summaries:
        raise ValueError("No development candidates")
    return min(summaries, key=lambda name: selection_loss(summaries[name], metric))


def selection_description(metric):
    return ("highest development task-macro partial-credit agreement" if metric == "agreement"
            else "lowest uncalibrated development task-macro categorical NLL")
