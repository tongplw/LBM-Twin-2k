# Building a large behavior model with Twin-2K-500

## Problem statement

**Question:** Can a model use a person's past answers to predict how that person will answer a new question?

This assignment uses survey responses from four waves. The goal is to predict answers in wave 4 from information available by the end of wave 3. The model should predict **what the person would choose**, not what a perfectly rational person should choose.

A large behavior model adapts a language model to this task. It should capture differences between people, including uncertainty, preferences, and inconsistent decisions.

## Motivation

- **Survey research:** Researchers could screen survey questions and experiments before recruiting participants.
- **Product research:** Product teams could shortlist concepts for human testing.
- **Behavioral science:** Behavioral scientists could study which personal information helps predict decisions.

These uses depend on evidence. A model that gives everyone the most common answer may score well without understanding any individual. This project therefore asks whether **the correct person's history improves prediction beyond simple baselines**.

## Dataset overview

[Twin-2K-500 on Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) contains **2,058 US adults**, surveyed across **four waves** on roughly **500 questions**. Topics cover demographics, psychological traits, cognitive tasks, economic preferences, behavioral experiments, and pricing.

Wave 4 repeats earlier questions. These repeated responses allow comparison with model predictions and show how often people agree with their own earlier answers. The dataset is released under CC BY 4.0. [Dataset paper](https://arxiv.org/abs/2505.17479).

There are two useful prediction settings:

1. **Main setting:** Predictions use earlier history but exclude the person's previous answer to the target question. This tests whether broader personal information helps.
2. **Possible extension:** Predictions could also include the previous target answer. This was not tested. Copying that answer would be the required baseline.

Human retest is a reference for the main setting. It uses the person's previous target answer, which the model does not receive.

The evaluation keeps **1,440 training, 309 development, and 309 test participants** in fixed, separate sets. A test person's permitted history is available at prediction time, but none of their answers train the shared model. Main results use **17 task groups**, with finer breakdowns for anchoring and Less is More.

## Report structure

1. [Literature review](docs/01_literature.md) — what related papers do, what to borrow, and their limits.
2. [Data exploration](docs/02_data_exploration.md) — the dataset, charts, leakage risk, and measured findings.
3. [Model and experimental method](docs/03_method.md) — data preparation, Qwen, prompts, and the core study of personalization, SFT, and hybrid retrieval.
4. [Evaluation](docs/04_evaluation.md) — metric definitions, baselines, aggregation, and result tables.
5. [Applications, guardrails, and maintenance](docs/05_applications.md).
6. [Reproducibility](docs/06_reproducibility.md) — code, settings, commands, and study limitations.

The chapters form one report. Code, configuration files, and analysis outputs support the report. Readers do not need to open JSON files.

**Study summary:** Two Qwen3.5-0.8B adapters and three experiments examine personalization, fine-tuning, and retrieval. Each adapter used 4,096 training examples (4.51% of a full pass). [Results](docs/04_evaluation.md) distinguish fine-tuning gains from evidence of personalization; [chapter 06](docs/06_reproducibility.md) gives the complete reproduction steps.

**Next: [1. Literature review →](docs/01_literature.md)**
