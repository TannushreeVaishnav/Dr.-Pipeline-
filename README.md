# PipelineDoctor

PipelineDoctor classifies ML pipeline failure logs and matches them to sourced runbook guidance. The demo includes a synthetic DVC pull failure in a realistic log format, clearly marked as a demo rather than a production incident.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-api-key"
$env:PIPELINEDOCTOR_OPENAI_MODEL = "model-id-available-to-your-account"
streamlit run app.py
```

Select **DVC demo log** to try the sample. PipelineDoctor cleans the log, predicts its failure category, retrieves matching entries from `data/knowledge_base.json`, and shows their source links. Classification and retrieval run locally without API credentials. Choose **Generate diagnosis** only after reviewing the log and confirming that it can be sent to OpenAI. The app sends the cleaned log and retrieved evidence only after you check the consent box. The output uses source IDs from those retrieved entries; unsupported IDs are rejected.

Logs are limited to 2 MiB. Diagnosis requests are limited to 12,000 characters and 500 output tokens. This version does not save logs to disk. Review logs for secrets before sharing them; timestamp and run ID redaction is pattern-based and does not detect secrets.

## Failure classification

The classifier uses TF-IDF word unigrams and bigrams with logistic regression from the existing scikit-learn dependency. It covers five categories: CI configuration, container build or runtime, data versioning, experiment tracking, and data drift. Category labels describe a broad area for investigation; they do not establish a root cause. The knowledge base currently covers only DVC, even when the classifier recognizes another category.

`data/labeled_logs.json` contains 30 hand-authored synthetic examples: 20 training logs and 10 fixed test logs, with four training and two test examples per category. There are **0 real incidents and 0 executed fault injections**. Each record includes its tool, category, expected root cause, provenance, and split. The model uses only the training log text and category; metadata and held-out examples are not used as features. Data loading rejects duplicate IDs and identical cleaned logs, including overlap across splits.

The app displays all five class probabilities. These probabilities are uncalibrated and must not be interpreted as diagnosis accuracy. Empty input and text with no learned vocabulary receive no category. Other unfamiliar logs can still receive a prediction; confidence thresholds and general out-of-scope detection are not implemented.

Run the fixed classifier evaluation without API calls:

```powershell
python evaluate_classifier.py
```

The JSON report includes sample counts, provenance, accuracy, per-category counts, and every held-out prediction. The small test set was authored alongside the training data and mostly has explicit tool names. It is a development smoke check, not evidence of real-world accuracy. Do not tune against these ten examples and continue calling them an untouched test set. The three-way comparison of LLM only, keyword search, and the full pipeline remains to be built.

The initial fixed holdout result is **9 correct out of 10 synthetic logs**. The missing-executable Docker case, `container-06`, was classified as CI configuration. This shows that overlapping startup and step-failure language can confuse the small baseline. The holdout result has not been used to tune the model. Collect independent real logs before drawing conclusions about operational usefulness or choosing confidence thresholds.

## Verify

```powershell
python -m unittest discover -s tests
```
