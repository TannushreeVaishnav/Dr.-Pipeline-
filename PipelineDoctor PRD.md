# PipelineDoctor PRD

Oct 21, 2026 · @TANNU

## Overview

PipelineDoctor is an assistant that diagnoses ML-pipeline failures from their logs, cites the evidence behind each diagnosis, and declines to answer when it is not confident.

**Problem.** My own MLOps projects (CryptoMLOps: GitHub Actions, Docker Compose, DVC, MLflow, Evidently) fail in a small number of repeating ways. Each time, the fix lives in a different place: official docs, an old commit, or my memory. Enterprise platform teams face the same pattern at larger scale: repeated incidents, scattered knowledge, slow diagnosis.

**Question to answer.** Can an assistant diagnose a failure from its log, point to the right fix with a source, and say "I'm not sure" when it doesn't know?

**Why I'm building it.** It closes the GenAI/RAG/evaluation gap on my resume for the Enterprise Platform AI Engineer role, using problems I have actually hit rather than a generic chatbot demo.

**Personal angle.**

- NeuralFlow taught me to treat model uncertainty as a feature (Monte Carlo Dropout). Here it becomes calibrated confidence and abstention.
- CryptoMLOps taught me to monitor drift. Here a new, unseen error type is treated as drift in log space and flagged as a runbook gap.

## Goals and non-goals

The project succeeds if the full pipeline measurably beats an LLM-only baseline on a labeled failure set, and the repo explains honestly where it fails.

**Goals**

- Diagnose the root cause of a failure log and propose a fix with citations to retrieved sources.
- Abstain on logs it should not answer, and say what extra information it would need.
- Compare three setups (LLM only, keyword search, full pipeline) on one labeled set.
- Ship a public GitHub repo with a README that tells the problem story and a short failure analysis.

**Non-goals**

- Auto-applying fixes or touching live systems.
- Supporting every tool; scope is the CryptoMLOps stack (GitHub Actions, Docker, DVC, MLflow, Evidently).
- A polished product UI; Streamlit is enough.

**Success criteria** (targets are placeholders until the baseline is measured)

| Metric | Target |
| --- | --- |
| Root-cause accuracy (top-1), full pipeline vs LLM only | Full pipeline higher; report both |
| Retrieval hit@k (k=3) | Report; investigate if below keyword baseline |
| Groundedness: claims backed by a retrieved chunk | Report as a percentage on the test set |
| Abstention: correct declines on out-of-scope logs | Report as X of Y |
| Time to MVP | 2 days; stretch by day 4 |

## Users and key scenarios

The primary user is an engineer who just saw a pipeline fail and wants the likely cause and a sourced fix in under a minute. I am that user first; a platform engineer on an enterprise team is the user I am designing for.

| Scenario | Input | Expected behavior |
| --- | --- | --- |
| Known failure | GitHub Actions log with a DVC pull error | Classifies as a data-versioning failure, cites the matching post-mortem and doc section, proposes the fix |
| Ambiguous failure | Docker Compose log with several errors | Ranks likely causes, shows confidence, asks for the one missing detail |
| Unseen failure | An error type not in the knowledge base | Abstains, flags it as a runbook gap, suggests writing a new entry |
| Drift alert | Evidently report summary | Routes to a drift-handling runbook rather than a code-bug fix |

## Functional requirements

The MVP covers requirements 1 to 4 and the three-way comparison; abstention, the drift flag and the larger dataset are stretch.

| # | Requirement | Priority |
| --- | --- | --- |
| 1 | Accept a pasted or uploaded failure log (Actions, Docker, DVC, MLflow, Evidently) | MVP |
| 2 | Classify the failure type with a TF-IDF + logistic regression baseline and show its confidence | MVP |
| 3 | Retrieve the top-k relevant chunks from a knowledge base of my post-mortems and official docs | MVP |
| 4 | Generate a diagnosis and fix via an LLM, with every claim citing a retrieved chunk | MVP |
| 5 | Run the evaluation script comparing LLM only, keyword search and the full pipeline | MVP |
| 6 | Abstain when classifier confidence or retrieval similarity falls below a threshold, and say what is missing | Stretch |
| 7 | Flag logs unlike anything in the knowledge base as a runbook gap | Stretch |
| 8 | Expand the labeled set from about 20 to 40+ failures | Stretch |
| 9 | Run the evaluation in GitHub Actions CI and publish the results table | Stretch |
| 10 | Streamlit UI showing diagnosis, confidence, and cited sources side by side | Stretch |

## System design

A failure log passes through five stages, and the last stage can end in a refusal instead of a diagnosis.

1. **Ingest and clean.** Strip timestamps and run IDs, keep the error lines and a few lines of context.
2. **Classify.** TF-IDF + logistic regression assigns a failure category and a confidence score. This is also the cheap baseline.
3. **Retrieve.** Embed the cleaned log, search the knowledge base, return the top-k chunks with similarity scores.
4. **Diagnose.** An LLM writes the root cause and fix using only the retrieved chunks, with a citation per claim.
5. **Decide.** If classifier confidence or retrieval similarity is below the threshold, output "not sure" plus the missing information; otherwise output the diagnosis.

**Stack**

| Layer | Choice |
| --- | --- |
| Language and ML | Python, scikit-learn |
| Embeddings and search | sentence-transformers with FAISS or Chroma |
| LLM | Any LLM API; the choice is an open question below |
| UI | Streamlit |
| Versioning and CI | Git, GitHub Actions |

**Data model.** Each knowledge-base chunk stores: source (post-mortem or doc), tool, symptom, root cause, fix, and a link or file path. Each labeled log stores: raw text, true category, true root cause, the chunk that answers it, and a flag for real or injected.

## Data plan

The labeled set will mix real and deliberately injected failures, and the README will state the split plainly (X real, Y injected).

**Knowledge base**

- My own short post-mortems: symptom, root cause, fix, one per real failure.
- Official documentation sections for the tools involved (Actions, Docker Compose, DVC, MLflow, Evidently), chunked by heading.

**Labeled failure set**

- Target: about 20 logs for the MVP, 40+ for the stretch.
- **Real:** start logging every genuine failure from now on and write each up as a post-mortem.
- **Injected:** break things on purpose, such as a bad config value, a pinned incompatible dependency, a corrupted DVC remote, or a missing secret. Treat it as lightweight chaos testing.
- Hold out a test split that the classifier training and knowledge-base writing never see. Include a few out-of-scope logs to test abstention.

**Rules**

- Injected failures are never described as production incidents.
- No confidential material from the internship goes into logs, the knowledge base, or the repo.

## Evaluation plan

The evaluation compares three setups on the same held-out set, so the value of retrieval and abstention is measured rather than claimed.

| Setup | What it isolates |
| --- | --- |
| LLM only, no retrieval | What the model knows without evidence |
| Keyword search over the docs | A cheap retrieval baseline |
| Full pipeline: classify, retrieve, cite, abstain | The system being built |

**Metrics**

- Root-cause accuracy (top-1) against the labeled cause.
- Retrieval hit@k and MRR: was the right chunk in the top k, and how high did it rank.
- Groundedness: the share of claims in the answer supported by a retrieved chunk. I will check a sample by hand before trusting any automated scorer.
- Abstention: correct declines on out-of-scope logs, and wrongful declines on in-scope ones.
- Calibration: reliability diagram or expected calibration error, to justify the abstention threshold.

**Error analysis (required deliverable).** A short written section covering where the system failed, why, and what I would change. A candid failure analysis is worth more than a high number.

**Research to do first** (verify details yourself before citing):

- RAG evaluation: the RAGAS framework and retrieval metrics.
- Log analysis: the Drain log parser and Loghub datasets, and how my logs differ from those benchmarks.
- Calibration: reliability diagrams and expected calibration error.

## Risks and mitigations

The biggest risk is a test set too small or too artificial to support any claim, so every result must be reported with its sample size and its real-versus-injected split.

| Risk | Mitigation |
| --- | --- |
| Too few labeled failures for meaningful metrics | Combine real and injected failures; report counts; avoid claiming statistical significance |
| Injected failures look unlike real ones | Keep them separate in reporting; log real failures from now on |
| LLM invents fixes not in the sources | Force citations, check groundedness by hand on a sample, abstain on weak retrieval |
| Evaluation leakage (test logs seen in training or in the knowledge base) | Freeze a held-out split before writing post-mortems |
| Overclaiming on the resume | Add bullets only after numbers exist; leave out RAG or agent claims until built |
| Confidential internship content leaks in | Use only my own project logs and public docs |
| Scope creep past the 2-day MVP | Cut stretch items first; the three-way comparison is the one thing that must ship |

## Timeline

The MVP fits in two days if the data work is front-loaded; days 3 and 4 are optional polish.

**Day 1: data and baselines**

1. Inject and collect about 20 failures; label each with category and root cause; freeze the test split.
2. Write the knowledge base: post-mortems plus chunked official docs.
3. Train the TF-IDF + logistic regression classifier and record its accuracy.
4. Build the keyword-search baseline.

**Day 2: pipeline and comparison**

1. Add embedding retrieval and measure hit@k.
2. Add LLM diagnosis with citations.
3. Run the three-way comparison and record results.
4. Write the first draft of the README with the problem story and results table.

**Day 3 (stretch): abstention and drift flag**

1. Add confidence and similarity thresholds; measure abstention and calibration.
2. Add the runbook-gap flag for unseen error types.

**Day 4 (stretch): scale and polish**

1. Grow the labeled set toward 40+ and rerun the comparison.
2. Run the evaluation in GitHub Actions; add the Streamlit UI.
3. Write the failure analysis and finalize the README.

## Deliverables

The repo README is the main deliverable, since recruiters read the README before the code.

- **GitHub repo** with code, the evaluation script, and a pinned, clearly named README.
- **README** with the problem story, architecture, results table (three setups), real-versus-injected split, and the failure analysis.
- **Resume bullets**, added only after real numbers exist:
  - Built PipelineDoctor, a RAG-based failure-triage assistant for ML pipelines (Python, FAISS, LLM API), evaluated on \_\_ labeled failures (\_\_ real, \_\_ injected).
  - Improved root-cause accuracy from \_\_% to \_\_% over an LLM-only baseline, with cited sources and confidence-based abstention.
- **Interview story (STAR):** Situation: my pipeline failed in repeating ways with scattered fixes. Task: diagnose from logs, cite sources, know when not to answer. Action: classical baseline, then retrieval, then LLM diagnosis with abstention, measured on a labeled set. Result: my real numbers. Reflection: what failed, what surprised me, what I would do next.

## Open questions

- [ ] Which LLM API will I use, and what is the budget cap for evaluation runs?
- [ ] Vector store: FAISS (simpler, local) or Chroma (easier metadata filtering)?
- [ ] How many real failures can I log in the first two days, versus injecting them?
- [ ] Who labels the root causes, and can a second person spot-check a sample?
- [ ] Which abstention threshold is acceptable: favor fewer wrong answers or fewer refusals?
- [ ] Should the repo include the Streamlit UI at MVP, or only after the evaluation is done?
