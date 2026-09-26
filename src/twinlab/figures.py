"""Create publication-style static figures from recorded analysis outputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from .data import ROOT

NAVY, BLUE, TEAL, ORANGE, GREY = "#17324d", "#3268ad", "#12877f", "#d4764b", "#788799"
OUT = ROOT / "docs/figures"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelcolor": NAVY,
        "text.color": NAVY, "xtick.color": GREY, "ytick.color": NAVY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#dce3eb",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white"})


def save(fig, name, directory=None):
    fig.savefig((OUT if directory is None else directory) / name, dpi=180, bbox_inches="tight", pad_inches=.2,
                metadata={"Software": "TwinLab / Matplotlib"})
    plt.close(fig)


def header(fig, title, subtitle):
    fig.suptitle(title, x=.05, y=.985, ha="left", fontsize=19, fontweight="bold", color=NAVY)
    fig.text(.05, .925, subtitle, fontsize=10.5, color=GREY, va="top")


def cohort(s):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(top=.83, bottom=.16, hspace=.65, wspace=.72, left=.22, right=.95)
    header(fig, "Who is represented?", "Observed cohort composition · 2,058 US adults · unweighted percentages")
    configs = [("QID13", "Age", ["18-29", "30-49", "50-64", "65+"]),
        ("QID14", "Education", ["Less than high school", "High school graduate", "Some college, no degree",
                                 "Associate's degree", "College graduate/some postgrad", "Postgraduate"]),
        ("QID15", "Race / origin (survey categories)", ["White", "Black", "Asian", "Hispanic", "Other"]),
        ("QID12", "Sex assigned at birth", ["Female", "Male"])]
    labels = {"College graduate/some postgrad": "College / some postgrad",
              "Some college, no degree": "Some college", "High school graduate": "High school"}
    for ax, (qid, title, order) in zip(axes.flat, configs):
        vals = [s["demographics"][qid][v]/2058*100 for v in order]
        yy = np.arange(len(order))
        ax.barh(yy, vals, color=BLUE if qid != "QID13" else TEAL, height=.58)
        ax.set_yticks(yy, [labels.get(v,v) for v in order], fontsize=10)
        ax.invert_yaxis(); ax.set_xlim(0, max(vals)*1.3)
        ax.set_title(title, loc="left", pad=12); ax.set_xlabel("Share (%)")
        ax.tick_params(axis="y", length=0)
        for y,v in zip(yy, vals): ax.text(v+1, y, f"{v:.1f}%", va="center", fontsize=10)
    fig.text(.05,.035,"Only 4 participants report non-US citizenship. These marginals do not establish population representativeness.\n"
             "Race/origin and sex categories retain the questionnaire's definitions; no Census weighting is asserted.", fontsize=10, color=GREY)
    save(fig, "01_cohort.png")


def structure(s, lengths):
    fig, axes = plt.subplots(1,2,figsize=(12,5.4))
    fig.subplots_adjust(top=.78,bottom=.24,wspace=.5,left=.09,right=.96)
    header(fig,"Context length and outcome structure", "Personas: all 2,058 people · outcomes: 1,749 training/development people")
    ax=axes[0]
    tokens=np.array([r["tokens"] for r in lengths])/1000
    ax.hist(tokens,bins=24,color=BLUE,edgecolor="white")
    ax.axvline(np.median(tokens),color=TEAL,lw=2)
    ax.set_title("Safe persona length",loc="left")
    ax.set_xlabel("Qwen3.5 tokens (thousands)"); ax.set_ylabel("Participants")
    ax.text(.97,.92,f"Median {np.median(tokens):.1f}k\nUse full safe history",transform=ax.transAxes,ha="right",va="top",fontsize=11)
    ax=axes[1]
    order=["binary","ordinal","bounded_numeric","unbounded_numeric"]
    vals=[s["target_types"][k] for k in order]
    names=["Binary choices","Ordered choices","Bounded numeric","Unbounded numeric"]
    ax.barh(range(4),np.array(vals)/sum(vals)*100,color=[TEAL,BLUE,ORANGE,GREY],height=.55)
    ax.set_yticks(range(4),names,fontsize=10); ax.invert_yaxis()
    ax.set_xlim(0,70); ax.set_xlabel("Share of response slots (%)")
    ax.set_title("167,894 paired response slots",loc="left")
    for i,v in enumerate(vals):ax.text(v/sum(vals)*100+1,i,f"{v/sum(vals):.1%}",va="center",fontsize=10)
    fig.text(.06,.07,"A participant has 94 or 98 repeated response slots, depending on the assigned probability-matching task.\n"
             "Forty pricing choices account for 41–43% of slots. Family-level averaging prevents them dominating the score.",fontsize=10,color=GREY)
    save(fig,"02_structure.png")


def reliability(s, diagnostic=False):
    if diagnostic:
        rows = [r for r in s["families"] if r["task_group"] in {"Anchoring", "Less is More (combined)"}]
    else:
        rows = s["task_groups"]
    rows=sorted(rows,key=lambda x:x["human"]["mean"])
    label_key = "family" if diagnostic else "task_group"
    fig, ax=plt.subplots(figsize=(12,6 if diagnostic else 9))
    fig.subplots_adjust(left=.25,right=.96,top=.78 if diagnostic else .83,
                        bottom=.25 if diagnostic else .14)
    title = "Inside the two combined task groups" if diagnostic else "What must a behavior model beat?"
    subtitle = ("309 development people · 95% participant-bootstrap intervals · diagnostic views only" if diagnostic else
                "17 main task groups · 309 development people · 95% participant-bootstrap intervals")
    header(fig,title,subtitle)
    for i,r in enumerate(rows):
        h,b=r["human"],r["price_aware"]
        ax.plot([b["mean"]*100,h["mean"]*100],[i,i],color="#d8e0e8",lw=2,zorder=1)
        for value,offset,color in [(h,-.11,TEAL),(b,.11,BLUE)]:
            ax.errorbar(value["mean"]*100,i+offset,
                xerr=np.array([[value["mean"]-value["lo"]],[value["hi"]-value["mean"]]])*100,
                fmt="o",color=color,capsize=2,ms=5,lw=1.3)
    ax.set_yticks(range(len(rows)),[r[label_key]+(" *" if r[label_key].startswith("Anchoring") else "") for r in rows])
    ax.set_xlim(45,96);ax.set_xlabel("Agreement score on bounded answers (%)")
    ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis="y",length=0)
    ax.plot([],[],"o",color=TEAL,label="Human retest reference")
    ax.plot([],[],"o",color=BLUE,label="Training-only baseline (price-aware)")
    ax.legend(loc="upper left" if diagnostic else "lower right",frameon=False,fontsize=10)
    fig.text(.06,.055,"Binary: exact agreement. Ordered/bounded: 1 − absolute error / scale range. This is not exact-match accuracy.\n"
             "* Anchoring rows score only the higher/lower binary probes; numerical estimates are reported separately.",fontsize=10,color=GREY)
    save(fig,"07_subgroups.png" if diagnostic else "03_reliability.png")


def transitions(s):
    fig,axes=plt.subplots(1,3,figsize=(14,6),gridspec_kw={"width_ratios":[1,1.3,1.05]})
    fig.subplots_adjust(top=.73,bottom=.30,wspace=.75,left=.09,right=.97)
    header(fig,"Three ways that repeated answers differ",
           "Development set: 309 matched people · choice reversals, rating shifts, and extreme numerical changes")
    labels=[["Small tray","Large tray"],["Definitely no","Probably no","Probably yes","Definitely yes"]]
    for ax,qid,title,labs in zip(axes[:2],["QID196","QID291"],
                               ["1. Ratio bias\nMarble choice","2. Omission bias\nVaccine-choice rating"],labels):
        counts=np.array(s["transitions"][qid]);probs=counts/counts.sum(axis=1,keepdims=True)
        ax.imshow(probs,cmap="Blues",vmin=0,vmax=1,aspect="auto")
        ax.set_xticks(range(len(labs)),labs,rotation=25,ha="right",fontsize=9)
        ax.set_yticks(range(len(labs)),labs,fontsize=9)
        ax.set_title(title,loc="left",pad=12);ax.set_xlabel("Wave 4");ax.set_ylabel("Earlier answer")
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j,i,f"{probs[i,j]:.0%}\n(n={counts[i,j]})",ha="center",va="center",fontsize=9,
                        color="white" if probs[i,j]>.58 else NAVY)
    numeric=s["native_numeric_errors"]["Anchoring: height"]
    values=[numeric["retest_absolute_error"]["median"],numeric["retest_mae"]]
    ax=axes[2]
    ax.barh([0,1],values,height=.42,color=[TEAL,ORANGE])
    ax.set_yticks([0,1],["Median","Mean"])
    ax.set_ylim(1.8,-.6);ax.set_xlim(0,max(values)*1.27)
    ax.set_title("3. Anchoring\nRedwood-height estimate",loc="left",pad=12)
    ax.set_xlabel("Absolute earlier–later difference (feet)",fontsize=9)
    ax.set_xticks([0,100,200,300]);ax.tick_params(axis="y",length=0)
    for y,value in enumerate(values):
        label=f"{value:,.0f}" if value==int(value) else f"{value:,.2f}"
        ax.text(value+7,y,label,va="center",fontsize=10,fontweight="bold")
    largest=numeric["retest_absolute_error"]["max"]
    ax.text(0,.03,f"Largest difference: {largest:,.0f} feet",transform=ax.transAxes,fontsize=9,color=GREY)
    fig.text(.06,.09,"Left and middle: cells show row percentages and counts; off-diagonal cells are changed answers.\n"
             "Right: extreme numerical changes pull the mean above the median. These are answer differences, not errors against a factual height.",
             fontsize=10,color=GREY)
    fig.text(.06,.025,"These descriptive comparisons do not establish causal changes in beliefs.",fontsize=10,color=GREY)
    save(fig,"04_transitions.png")


def pricing(rows,s):
    selected=[r for r in rows if r["family"]=="Pricing"]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    fig.subplots_adjust(top=.78,bottom=.23,wspace=.5,left=.1,right=.95)
    header(fig,"A stronger baseline uses the actual offered price", "Development set: 309 people × 40 products · price-bin edges estimated from training data only")
    ax=axes[0]
    xx=np.arange(5)
    observed=[np.mean([r["later"]==1 for r in selected if r["training_price_bin"]==i]) for i in xx]
    predicted=[np.mean([r["predicted_purchase_probability"] for r in selected if r["training_price_bin"]==i]) for i in xx]
    ax.plot(xx,np.array(observed)*100,"o-",color=TEAL,label="Observed wave-4 purchases")
    ax.plot(xx,np.array(predicted)*100,"s--",color=BLUE,label="Training frequency prediction")
    ax.set_xticks(xx,["Lowest","2","3","4","Highest"]);ax.set_ylim(0,100)
    ax.set_xlabel("Within-product price quintile");ax.set_ylabel("Purchase share (%)")
    ax.set_title("Price alone explains substantial variation",loc="left");ax.legend(frameon=False,fontsize=9)
    pricing=next(r for r in s["families"] if r["family"]=="Pricing")
    ax=axes[1]
    vals=[pricing[n]["mean"]*100 for n in ["baseline","price_aware","human"]]
    bars=ax.bar(range(3),vals,color=[GREY,BLUE,TEAL],width=.6)
    ax.set_xticks(range(3),["Product\nfrequency","Product +\nprice bin","Human\nretest"])
    ax.set_ylim(0,100);ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("Same 12,360 development choices",loc="left")
    for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+2,f"{val:.1f}%",ha="center",fontsize=12,fontweight="bold")
    fig.text(.06,.04,"Price-aware predictions use five training-derived bins and shrink each bin toward its product's response distribution.\n"
             "This is a descriptive baseline, not a personalized model or evidence of real-world demand elasticity.",fontsize=10,color=GREY)
    save(fig,"05_pricing.png")


def architecture():
    from .diagrams import architecture as render_architecture
    render_architecture()


def data_flow(s, splits, development_rows):
    """Keep paper-level questions, parsed responses, and analysis scope distinct."""
    audit = s["full_persona_audit"]
    total = audit["expected_responses"]
    changed = audit["changed_responses"]
    unchanged = audit["unchanged_responses"]
    dev_count = len(development_rows)
    variants = {int(k): v for k, v in s["target_atoms_per_person"].items()}
    assert sum(k * v for k, v in variants.items()) == total == changed + unchanged
    assert sum(variants.values()) == len(splits["train"]) + len(splits["dev"])
    fig, ax = plt.subplots(figsize=(12, 10.5))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10.5); ax.axis("off")
    ax.text(.35, 10.1, "From survey waves to the analyzed responses", fontsize=20, fontweight="bold")
    ax.text(.35, 9.7, "Question counts, individual responses and participant splits describe different units.",
            fontsize=11, color=GREY)

    def box(x, y, w, h, title, body, color=BLUE, fill="#f4f7fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.035,rounding_size=.10",
                                   linewidth=1.2, edgecolor=color, facecolor=fill))
        ax.text(x+.16, y+h-.19, title, fontsize=13, fontweight="bold", va="top", color=color)
        ax.text(x+.16, y+h-.57, body, fontsize=10.5, va="top", linespacing=1.5)

    def arrow(start, end, color=GREY):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   color=color, lw=1.4))

    box(.35, 8.0, 3.0, 1.25, f"{s['scope']['cohort']:,} participants", "Same people across all waves\nUS adults completing waves 1–4")
    box(3.8, 8.0, 3.35, 1.25, "Waves 1–3", "~500 questions per person\nPersonal history + earlier answers")
    box(7.6, 8.0, 4.0, 1.25, "Wave 4", "88 repeated questions (paper count)\nSame assigned experimental conditions")
    arrow((3.4, 8.63), (3.75, 8.63))
    arrow((7.2, 8.63), (7.55, 8.63))
    arrow((9.6, 7.95), (9.6, 7.5))
    box(3.8, 6.1, 7.8, 1.35, "94 or 98 recorded responses per person", 
        "Count each answer part or rating row separately; task variants differ by four responses.\n"
        "Pair each wave-4 response with that person's earlier answer to the same item.\n"
        "The paper's 88-question count is not the response-count denominator.", TEAL, "#f0f8f6")

    ax.text(.35, 5.7, "Fixed participant split: approximately 70% / 15% / 15%", fontsize=12, fontweight="bold")
    ax.text(.35, 5.4, "All waves from a person stay in the same set.", fontsize=10.5, color=GREY)
    # One branch from the common source into three participant-disjoint sets.
    ax.plot([7.7, 7.7], [6.06, 5.25], color=GREY, lw=1.3)
    ax.plot([2.05, 9.9], [5.25, 5.25], color=GREY, lw=1.3)
    for x in (2.05, 5.9, 9.9):
        arrow((x, 5.25), (x, 5.05))
    box(.35, 3.7, 3.4, 1.3, f"Training · {len(splits['train']):,} people",
        f"{total-dev_count:,} paired responses\nEarlier answers are training labels")
    box(4.2, 3.7, 3.4, 1.3, f"Development · {len(splits['dev']):,} people",
        f"{dev_count:,} paired responses\nUsed for baseline comparisons", TEAL, "#f0f8f6")
    box(8.05, 3.7, 3.55, 1.3, f"Final test · {len(splits['test']):,} people",
        "Reserved for one final evaluation\nOutcomes excluded from this analysis", GREY, "#f5f6f8")
    ax.plot([2.05, 2.05, 5.9, 5.9], [3.66, 3.4, 3.4, 3.66], color=GREY, lw=1.3)
    arrow((3.975, 3.4), (3.975, 3.08))
    box(.35, 1.93, 7.25, 1.1, f"Outcome exploration · {audit['participants']:,} people",
        f"{total:,} earlier–later response pairs\n{variants[94]:,} people × 94 + {variants[98]:,} people × 98 = {total:,}", TEAL, "#f0f8f6")
    ax.text(8.05, 2.97, "One pair contains", fontsize=11, fontweight="bold", va="top")
    ax.text(8.05, 2.61, "one earlier answer +\none wave-4 answer.\nOther waves 1–3 answers\nare outside this paired count.",
            fontsize=10.5, va="top", linespacing=1.5)
    arrow((3.975, 1.88), (3.975, 1.53))
    ax.text(.35, 1.61, "Exact answer comparison", fontsize=11, fontweight="bold", va="bottom")
    width = 7.25
    for start, fraction, color, label in [
        (0, changed/total, ORANGE, f"Changed · {changed/total:.1%}\n{changed:,} responses"),
        (changed/total, unchanged/total, TEAL, f"Unchanged · {unchanged/total:.1%}\n{unchanged:,} responses"),
    ]:
        ax.add_patch(plt.Rectangle((.35+width*start, .63), width*fraction, .81,
                                  facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(.35+width*(start+fraction/2), 1.035, label, ha="center", va="center",
                fontsize=11, color="white", linespacing=1.5)
    ax.text(8.05, 1.26, "37.2% counts any change,\nnot its size. It is not the\npaper's partial-credit error rate.",
            fontsize=10.5, va="top", linespacing=1.5)
    ax.text(.35, .20, "Sources: Twin-2K-500 paper (question counts); dataset audit (response counts and changes). No model trained.",
            fontsize=9.5, color=GREY)
    save(fig, "08_data_flow.png")


def human_consistency(s):
    comparison = s["train_dev_retest_comparison"]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(left=.23, right=.78, top=.74, bottom=.27)
    header(fig, "Identical answers versus close answers",
           f"{comparison['participants']:,} training/development participants · "
           f"{comparison['paired_responses']:,} response pairs · same 17-task weighting")
    for y, key, color in [(1, "exact_agreement", BLUE), (0, "partial_credit_agreement", TEAL)]:
        score = comparison[key]
        value, lo, hi = (score[k] * 100 for k in ("mean", "lo", "hi"))
        ax.barh(y, value, height=.4, color=color)
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt="none",
                    ecolor=NAVY, elinewidth=1.6, capsize=6, capthick=1.6)
        ax.text(1.04, y+.08, f"{value:.2f}%", transform=ax.get_yaxis_transform(),
                fontsize=15, fontweight="bold", va="center")
        ax.text(1.04, y-.19, f"95% CI {lo:.2f}–{hi:.2f}%", transform=ax.get_yaxis_transform(),
                fontsize=10, va="center", color=GREY)
    ax.set_yticks([1, 0], ["Exact agreement", "Partial-credit\nagreement"], fontsize=12)
    ax.set_ylim(-.55, 1.55)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Agreement score (%)", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.05, .075, "Whiskers: 95% confidence intervals from 2,000 participant-bootstrap samples.\n"
             "Both scores exclude unbounded numerical estimates. Partial credit measures closeness, not identical answers.",
             fontsize=10, color=GREY, linespacing=1.5)
    save(fig, "09_human_consistency.png")


def main():
    style();OUT.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/"data/analysis/summary.json").read_text())
    lengths=json.loads((ROOT/"data/processed/context_lengths.json").read_text())
    rows=json.loads((ROOT/"data/processed/development_pairs.json").read_text())
    cohort(s);structure(s,lengths);reliability(s);transitions(s);pricing(rows,s);architecture()
    reliability(s, diagnostic=True)
    splits=json.loads((ROOT/"splits/participants.json").read_text())
    data_flow(s, splits, rows)
    human_consistency(s)
    from .diagrams import training_pipeline, judge_extension, maintenance
    training_pipeline(); judge_extension(); maintenance()
    print(f"Rendered twelve figures in {OUT}")


if __name__=="__main__":main()
