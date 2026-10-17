import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from log_cleaner import clean_log


CATEGORIES = {
    "ci_configuration": "CI configuration",
    "container": "Container build or runtime",
    "data_versioning": "Data versioning",
    "experiment_tracking": "Experiment tracking",
    "data_drift": "Data drift",
}


def load_labeled_logs(path: Path) -> list[dict[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Labeled logs must be a non-empty JSON array.")
    required = {"id", "split", "provenance", "tool", "category", "root_cause", "log"}
    ids, logs = set(), set()
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("Each labeled log must contain all required fields.")
        if not all(isinstance(record[key], str) and record[key].strip() for key in required):
            raise ValueError("Labeled log fields must be non-empty strings.")
        if record["category"] not in CATEGORIES or record["split"] not in {"train", "test"}:
            raise ValueError("Unknown category or split in labeled logs.")
        if record["provenance"] not in {"synthetic", "real", "injected"}:
            raise ValueError("Unknown labeled log provenance.")
        normalized = " ".join(clean_log(record["log"]).lower().split())
        if record["id"] in ids or normalized in logs:
            raise ValueError("Duplicate ID or cleaned log; train/test overlap is not allowed.")
        ids.add(record["id"])
        logs.add(normalized)
    for category in CATEGORIES:
        for split, minimum in (("train", 2), ("test", 1)):
            count = sum(r["category"] == category and r["split"] == split for r in records)
            if count < minimum:
                raise ValueError(f"Category {category} requires at least {minimum} {split} logs.")
    return records


def train_classifier(records: list[dict[str, str]]) -> Pipeline:
    training = [record for record in records if record["split"] == "train"]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(C=4, max_iter=1000, random_state=42)),
    ])
    model.fit([clean_log(record["log"]) for record in training],
              [record["category"] for record in training])
    return model


def classify_log(log: str, model: Pipeline) -> dict[str, float]:
    cleaned = clean_log(log)
    if not cleaned.strip() or model.named_steps["tfidf"].transform([cleaned]).nnz == 0:
        return {}
    probabilities = model.predict_proba([cleaned])[0]
    return dict(sorted(zip(model.classes_, map(float, probabilities)),
                       key=lambda item: (-item[1], item[0])))
