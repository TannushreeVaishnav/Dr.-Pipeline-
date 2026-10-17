"""Report classifier results on the fixed synthetic holdout, without API calls."""

import json
from collections import Counter
from pathlib import Path

from classifier import CATEGORIES, classify_log, load_labeled_logs, train_classifier


def evaluate(records: list[dict[str, str]]) -> dict:
    model = train_classifier(records)
    outcomes = []
    for record in records:
        if record["split"] == "test":
            probabilities = classify_log(record["log"], model)
            predicted = next(iter(probabilities), None)
            outcomes.append({
                "id": record["id"], "expected": record["category"],
                "predicted": predicted, "correct": predicted == record["category"],
            })
    correct = sum(outcome["correct"] for outcome in outcomes)
    return {
        "training_count": sum(record["split"] == "train" for record in records),
        "test_count": len(outcomes),
        "provenance_counts": dict(Counter(record["provenance"] for record in records)),
        "correct": correct,
        "accuracy": correct / len(outcomes) if outcomes else None,
        "per_category": {
            category: {
                "correct": sum(o["correct"] for o in outcomes if o["expected"] == category),
                "total": sum(o["expected"] == category for o in outcomes),
            } for category in CATEGORIES
        },
        "predictions": outcomes,
        "limitations": "Small synthetic holdout authored alongside training data. "
                        "Probabilities are uncalibrated; this does not establish real-world accuracy.",
    }


if __name__ == "__main__":
    records = load_labeled_logs(Path(__file__).resolve().parent / "data" / "labeled_logs.json")
    print(json.dumps(evaluate(records), indent=2))
