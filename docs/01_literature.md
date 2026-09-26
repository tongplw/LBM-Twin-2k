# 1. Literature review

The literature suggests two things: personal evidence can help, but a convincing simulated answer is not enough. Evaluation must test whether the model predicts the **right person**, not just a typical respondent.

**Scope:** The published results below use different models, datasets, and evaluation settings. They inform this study but are not directly comparable scores.

## 1.1 Closest work on behavior prediction

### 1.1.1 Twin-2K-500 — Toubia et al. (2025)

- **What it does:** Collects roughly 500 survey answers from **2,058 US adults across four waves**. Its main baseline prompts **GPT-4.1-mini** with a person's non-target answers to predict held-out wave-1–3 responses. Across 17 tasks, it reports 71.72% agreement versus 81.72% human retest agreement, measured using wave 4. These scores include partial credit, not just exact matches.
- **Design implication:** Separating histories from target answers prevents leakage. Preserving question wording and conditions supports valid comparisons with human retest. Full histories, summaries, and input formats require empirical comparison.
- **Limit:** Our evaluation follows the [dataset guide's wave-4 prediction setup](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500#2-wave-split-folder). The [paper's reported experiment](https://arxiv.org/html/2505.17479v1#S4) instead scores predictions against waves 1–3 answers and uses wave 4 for human retest. Different target answers, participant splits, and numerical scoring prevent direct score comparison.

[Read the paper](https://arxiv.org/abs/2505.17479).

### 1.1.2 Generative Agent Simulations of 1,000 People — Park et al. (2024)

- **What it does:** Builds **GPT-4o-based agents for 1,052 people** using two-hour interview transcripts and model-generated observations from four social-science perspectives. It evaluates survey answers, personality, economic games, and experiments. On the General Social Survey, agents reach **85% of human two-week retest performance**—a relative score, not 85% exact accuracy.
- **Design implication:** Actual personal evidence provides a stronger basis than a demographic description alone. Demographics-only inputs and human consistency provide useful references. Short summaries would need tests of whether they preserve useful evidence.
- **Limit:** Interview narratives differ from the structured survey data used here, and generated observations can introduce unsupported assumptions. Its GPT-4o results do not establish what Qwen3.5-0.8B will achieve or guarantee similar performance across tasks.

[Read the original version](https://arxiv.org/abs/2411.10109v1).

### 1.1.3 Digital Twins as Funhouse Mirrors: Five Key Distortions — Peng et al. (2025)

- **What it does:** Evaluates **1,784 returning Twin-2K participants** across **19 preregistered studies and 164 outcomes**. Its main twins use **GPT-4.1** prompted with earlier survey histories. Predictions correlate weakly with human responses (average **r = 0.20**), and rich histories provide only modest gains over generic or demographic inputs.
- **Design implication:** Correct, absent, and shuffled histories test whether a model uses person-specific information. Response variation, subgroup errors, stereotypes, and unrealistically rational choices expose failures that high average agreement can hide.
- **Limit:** This is a test on new studies, not the wave-4 benchmark used here. The authors also test alternative models and fine-tuning, but their results do not rule out every future method. The findings motivate failure checks without establishing that personalization is impossible.

[Read the study (2026 revision)](https://arxiv.org/html/2509.19088v5).

### 1.1.4 Centaur: A foundation model to predict and capture human cognition — Binz et al. (2025)

- **What it does:** Fine-tunes **Llama 3.1 70B** on **Psych-101**, containing over 10 million choices from 60,092 people across 160 psychological experiments. It predicts choices from experiment instructions and previous trials. Fine-tuning improves prediction over the base model on held-out participants and tested new experimental settings, measured by negative log-likelihood.
- **Design implication:** Adapter training and answer-token loss support efficient adaptation. Frozen-model controls separate training gains from base-model capability. Held-out people and new experiments test different forms of generalization.
- **Limit:** Learning from a person's choices within an experiment differs from using a rich personal profile across surveys. Its much larger model and training dataset do not establish expected performance for Qwen3.5-0.8B or the wave-4 task in this assignment.

[Read the Nature paper](https://www.nature.com/articles/s41586-025-09215-4).

### 1.1.5 Socrates: Finetuning LLMs for Human Behavior Prediction in Social Science Experiments — Kolluri et al. (2025)

- **What it does:** Fine-tunes **Llama-3-8B-Instruct and Qwen2.5-14B-Instruct** on **SocSci210**, containing 2.9 million responses from 400,491 participants across 210 experiments. Inputs combine demographics, the assigned experimental condition, and the question. It compares SFT, SFT with generated explanations, and contrastive DPO. Its unseen-study evaluation trains on 170 studies and tests on 40.
- **Design implication:** Individual prediction and population response distributions require separate evaluation. On unseen studies, SFT better matched distributions, while DPO achieved better individual agreement. This motivates SFT as a starting point and DPO as a possible later comparison.
- **Limit:** Its demographic profiles and larger models differ from the rich histories and small model studied here. Its DPO pairs contrast different people's answers, which does not prove that one person rejects the other's choice.

[Read the EMNLP paper](https://aclanthology.org/2025.emnlp-main.1530.pdf).

## 1.2 Relevant techniques

| Paper | What it does | What to borrow | Limitation here |
| --- | --- | --- | --- |
| [LaMP — Salemi et al., 2024](https://aclanthology.org/2024.acl-long.399/) | Tests personalized classification and text generation using retrieved profile entries | Retrieve relevant facts from the person's own history | Better text personalization does not guarantee better human-behavior prediction |
| [Lost in the Middle — Liu et al., 2023/24](https://arxiv.org/abs/2307.03172) | Studies how models use evidence at different positions in long inputs | Compare selected facts with full histories and test evidence order | Results depend on the model and task |
| [Whose Opinions Do Language Models Reflect? — Santurkar et al., 2023](https://arxiv.org/abs/2303.17548) | Compares model opinions with human population and group distributions | Measure subgroup errors and response distributions | Demographic prompting alone does not validate individual predictions |
| [LoRA — Hu et al., 2021/22](https://arxiv.org/abs/2106.09685) | Adapts models with small trainable weight additions | Use one shared adapter to reduce training cost | A smaller update can also limit learning |
| [DPO — Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | Learns from preferred/rejected response pairs without a separate reward-training loop | Consider it if reliable preference pairs become available | Survey choices are noisy observations, not complete preference rankings |
| [DeepSeekMath / GRPO — Shao et al., 2024](https://arxiv.org/abs/2402.03300) | Improves mathematical reasoning using reward-based training | Borrow the requirement for a clearly defined, verifiable reward | Factual correctness is the wrong reward for reproducing human mistakes |
| [Judging LLM-as-a-Judge — Zheng et al., 2023](https://arxiv.org/abs/2306.05685) | Evaluates model judges and their biases | Validate any judge against blinded human ratings | A judge adds unnecessary subjectivity to multiple-choice scoring |
| [Calibration — Guo et al., 2017](https://arxiv.org/abs/1706.04599) | Studies confidence errors and temperature scaling | Adjust probability confidence using development data | Better confidence estimates cannot recover missing personal information |

## 1.3 Implications for this study

- **Simple baselines:** Question wording, conditions, and price can explain substantial performance without personal information.
- **Personalization controls:** Correct, absent, and shuffled histories distinguish person-specific evidence from common answer patterns.
- **Separate SFT conditions:** Question-only training provides a control for learning answer patterns without personal history.
- **Beyond averages:** Aggregate agreement can hide weak individual prediction and uneven subgroup performance.
- **New-question testing:** The [Twin-2K-500 Mega-Study](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study) is a possible future benchmark, subject to linkage and leakage checks. It was not evaluated here.

**Previous: [← Report overview](../README.md)**

**Next: [2. Data exploration →](02_data_exploration.md)**
