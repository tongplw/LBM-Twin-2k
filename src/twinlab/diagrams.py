"""Report-native diagrams with orthogonal connectors and semantic symbols."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path

from .data import ROOT

NAVY, BLUE, TEAL, ORANGE, GREY = "#17324d", "#3268ad", "#12877f", "#d4764b", "#788799"


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 7.3))
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.3); ax.axis("off")
    ax.text(.3, 6.95, title, fontsize=19, fontweight="bold", color=NAVY)
    ax.text(.3, 6.5, subtitle, fontsize=10.5, color=GREY)
    return fig, ax


def node(ax, x, y, w, h, title, body, icon=None, color=BLUE):
    fill = "#f0f8f6" if color == TEAL else "#f4f7fb"
    if icon == "database":
        # A compact database symbol sits above the text, not around it.
        cx, bottom, width, height, rim = x+w/2, y+h-.40, .54, .32, .12
        # Open outline: two sides and only the visible front curve of the base.
        # A rectangle edge or complete bottom ellipse adds an unwanted line.
        left, right, top = cx-width/2, cx+width/2, bottom+height
        outline = Path([(left, top), (left, bottom),
                        (left, bottom-rim*2/3), (right, bottom-rim*2/3),
                        (right, bottom), (right, top)],
                       [Path.MOVETO, Path.LINETO, Path.CURVE4,
                        Path.CURVE4, Path.CURVE4, Path.LINETO])
        ax.add_patch(PathPatch(outline, facecolor=fill, edgecolor=color, lw=1.3))
        ax.add_patch(Ellipse((cx, bottom+height), width, rim,
                             facecolor=fill, edgecolor=color, lw=1.3))
        ax.text(cx, y+h-.64, title, fontsize=11.5, fontweight="bold", color=color,
                ha="center", va="center")
        ax.text(cx, y+h-.87, body, fontsize=10, color=NAVY,
                ha="center", va="top", linespacing=1.6)
        return
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.1",
                                   facecolor=fill, edgecolor=color, lw=1.3))
    title_x = x+.14
    if icon == "model":
        cx, cy, size = x+.14, y+h-.40, .23
        ax.add_patch(Rectangle((cx, cy), size, size, facecolor="white", edgecolor=color, lw=1.1))
        for offset in (.06, .17):
            ax.plot([cx-.05, cx], [cy+offset]*2, color=color, lw=1)
            ax.plot([cx+size, cx+size+.05], [cy+offset]*2, color=color, lw=1)
            ax.plot([cx+offset]*2, [cy-.05, cy], color=color, lw=1)
            ax.plot([cx+offset]*2, [cy+size, cy+size+.05], color=color, lw=1)
        title_x += .38
    title_y = y+h-.32
    ax.text(title_x, title_y, title, fontsize=11.5, fontweight="bold", color=color, va="center")
    ax.text(x+.14, y+h-.64, body, fontsize=10, color=NAVY, va="top", linespacing=1.6)


def connect(ax, points, color=GREY, dashed=False):
    """Every segment is horizontal or vertical, including the arrow segment."""
    for a, b in zip(points, points[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError("Diagram connectors must be orthogonal")
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=1.5, linestyle="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
                               color=color, lw=1.5, linestyle="--" if dashed else "-"))


def save(fig, name):
    path = ROOT / "docs/figures" / name
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.2,
                metadata={"Software": "TwinLab / Matplotlib"})
    plt.close(fig)


def architecture():
    fig, ax = canvas("System overview: one person, one question, one prediction",
                     "One shared Qwen model; original target answers never enter its input.")
    node(ax,.3,4.45,2.5,1.55,"Input history","Waves 1–3 responses\nTarget answers excluded\nFull history by default", "database")
    node(ax,3.25,4.45,2.5,1.55,"Build messages","System role + history\nCurrent question + options\nNo answer in the input")
    node(ax,6.2,4.45,2.45,1.55,"Qwen 0.8B","Frozen or SFT adapter\nSame shared parameters\nFresh context per question", "model")
    node(ax,9.1,4.45,2.6,1.55,"Prediction","Valid-option probabilities\nPredicted choice\nRecord time and failures", color=TEAL)
    for a,b in [(2.8,3.25),(5.75,6.2),(8.65,9.1)]: connect(ax,[(a,5.22),(b,5.22)])
    node(ax,3.25,1.75,2.5,1.55,"Question store","Exact wording and price\nAssigned condition\nTarget answer removed", "database")
    connect(ax,[(4.5,3.3),(4.5,4.43)])
    node(ax,6.2,1.75,2.45,1.55,"Evaluate","Exact / partial credit\nProbability quality\nCompare on same people", color=TEAL)
    node(ax,9.1,1.75,2.6,1.55,"Held-out labels","Development: 309 people\nFinal test: 309 reserved\nUsed only for scoring", "database", TEAL)
    connect(ax,[(10.4,4.43),(10.4,3.83),(7.425,3.83),(7.425,3.32)],TEAL)
    connect(ax,[(9.08,2.52),(8.67,2.52)],TEAL)
    ax.text(.3,.73,"Training labels: earlier repeated answers from 1,440 training people. Development labels: wave-4 answers.",fontsize=10,color=GREY)
    ax.text(.3,.35,"Core study: E1 personalization controls · E2 question-only vs history SFT · E3 hybrid retrieval. Development selects the model; test evaluation follows.",fontsize=10,color=GREY)
    save(fig,"06_architecture.png")


def training_pipeline():
    fig, ax = canvas("Training pipeline: question-only versus history-conditioned SFT",
                     "SFT is the objective; LoRA updates two adapters on the same training examples.")
    node(ax,.3,4.45,2.5,1.55,"Prepared data","Train: 1,440 people\nDevelopment: 309 people\nOriginal human answers", "database")
    node(ax,3.25,4.45,2.5,1.55,"Frozen references","Question / demographics\nCorrect / shuffled history\nStatistical baselines", "model")
    node(ax,6.2,4.45,2.45,1.55,"SFT + LoRA","Question-only vs history\nTwo separate adapters\nSame labels and updates", "model", TEAL)
    node(ax,9.1,4.45,2.6,1.55,"Development","E1 controls + E3 retrieval\nCompare quality and cost\nSelect and freeze recipe", color=TEAL)
    for a,b in [(2.8,3.25),(5.75,6.2),(8.65,9.1)]:connect(ax,[(a,5.22),(b,5.22)])
    node(ax,3.25,1.65,2.5,1.7,"Optional QLoRA","Quantized frozen base\nAlternative adapter setup\nLess weight memory\nNot another training stage",color=GREY)
    connect(ax,[(5.77,2.5),(5.97,2.5),(5.97,3.85),(7.425,3.85),(7.425,4.43)],GREY,True)
    node(ax,6.2,1.65,2.45,1.7,"Deferred DPO / GRPO","Need reliable preferences\nor a validated reward\nExtra optimization cost\nNot used in the main run",color=ORANGE)
    connect(ax,[(8.1,4.43),(8.1,3.37)],ORANGE,True)
    node(ax,9.1,1.65,2.6,1.7,"Final evaluation","309 reserved people\nOne fixed recipe\nNo tuning on these labels", "database", TEAL)
    connect(ax,[(10.4,4.43),(10.4,3.37)],TEAL)
    ax.text(.3,.72,"LoRA reduces trainable parameters and optimizer state; speed gains and long-context memory fit must be measured.",fontsize=10,color=GREY)
    ax.text(.3,.34,"All 17 tasks; 4,096 sampled questions and 128 updates per adapter. The fixed budget is 4.51% of a full pass.",fontsize=10,color=GREY)
    save(fig,"10_training_pipeline.png")


def judge_extension():
    fig, ax = canvas("Optional extension: learning persona-consistent review writing",
                     "Not part of the survey-answer benchmark. Judge rewards need human validation.")
    node(ax,.3,4.45,2.5,1.6,"Training evidence","Consented writing/profile\nReal ratings or preferences\nTraining people only", "database")
    node(ax,3.25,4.45,2.5,1.6,"Candidate reviews","Same task and evidence\nSeveral generated drafts\nNo invented experiences", "model")
    node(ax,6.2,4.45,2.45,1.6,"Training judge","Fixed rubric\nCalibrate against humans\nRandomize draft order", "model")
    node(ax,9.1,4.45,2.6,1.6,"Optional update","DPO: preference pairs\nGRPO: rubric rewards\nLog rejected/uncertain cases",color=ORANGE)
    for a,b in [(2.8,3.25),(5.75,6.2),(8.65,9.1)]:connect(ax,[(a,5.25),(b,5.25)])
    node(ax,.3,1.5,2.5,1.7,"Evaluation input","Unseen people and topics\nPermitted profile + request\nNo reference review in input", "database", TEAL)
    node(ax,3.25,1.5,2.5,1.7,"Adapted generator","Frozen evaluation version\nNo evaluation-label access\nProduce new reviews", "model", TEAL)
    node(ax,6.2,1.5,2.45,1.7,"Independent judge","Separate configuration\nSame disclosed criteria\nBlind to model identity", "model", TEAL)
    node(ax,9.1,1.5,2.6,1.7,"Human validation","Held-out writing and ratings\nJudge–human agreement\nSubgroup and privacy checks",color=TEAL)
    for a,b in [(2.8,3.25),(5.75,6.2),(8.65,9.1)]:connect(ax,[(a,2.35),(b,2.35)],TEAL)
    connect(ax,[(10.4,4.43),(10.4,3.8),(4.5,3.8),(4.5,3.22)],GREY)
    ax.text(.3,.65,"Training-judge scores are not the final evidence of success. Independent judges can still share biases with the generator.",fontsize=10,color=GREY)
    ax.text(.3,.28,"Rubric: supported preferences · no invented experience · style consistency · relevance · privacy.",fontsize=10,color=GREY)
    save(fig,"11_judge_extension.png")


def maintenance():
    fig, ax = canvas("Maintain the model with fresh human evidence",
                     "A versioned lifecycle with a release decision, monitoring, and a route back to review.")
    node(ax,.4,4.4,3.2,1.65,"1  Collect and version","Consented human observations\nRecord scope and provenance\nSeparate synthetic responses", "database")
    node(ax,4.4,4.4,3.2,1.65,"2  Prepare and adapt","Fixed data and prompt versions\nRecheck leakage and mappings\nTrain on approved observations", "model")
    node(ax,8.4,4.4,3.2,1.65,"3  Validate and approve","Fresh labels + old-model comparison\nQuality, subgroup and safety checks\nHuman owner decides promotion",color=TEAL)
    connect(ax,[(3.6,5.22),(4.38,5.22)]);connect(ax,[(7.62,5.22),(8.38,5.22)])
    node(ax,8.4,1.5,3.2,1.65,"4  Release with rollback","Preserve the previous version\nLog configuration and predictions\nLimit to validated applications",color=TEAL)
    node(ax,4.4,1.5,3.2,1.65,"5  Monitor real outcomes","New wording, populations, prices\nFresh human labels and calibration\nLatency, cost and incidents")
    node(ax,.4,1.5,3.2,1.65,"6  Review or retire","Diagnose before retraining\nRollback on severe failures\nUpdate, revalidate, or stop",color=ORANGE)
    connect(ax,[(10,4.38),(10,3.17)],TEAL)
    connect(ax,[(8.38,2.32),(7.62,2.32)]);connect(ax,[(4.38,2.32),(3.62,2.32)])
    connect(ax,[(2,3.17),(2,4.38)],ORANGE)
    ax.text(.4,.65,"Input drift triggers investigation, not automatic proof of failure. Reuse neither synthetic answers as human labels nor a repeatedly tuned final test.",fontsize=9.7,color=GREY)
    save(fig,"12_maintenance.png")
