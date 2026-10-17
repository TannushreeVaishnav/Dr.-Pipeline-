import copy
import json
import tempfile
import unittest
from pathlib import Path

from classifier import CATEGORIES, classify_log, load_labeled_logs, train_classifier
from evaluate_classifier import evaluate


ROOT = Path(__file__).resolve().parents[1]


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_labeled_logs(ROOT / "data" / "labeled_logs.json")
        cls.model = train_classifier(cls.records)

    def load_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            return load_labeled_logs(path)

    def test_dvc_demo_is_classified_with_a_probability_distribution(self):
        log = (ROOT / "examples" / "dvc_pull_failed.log").read_text(encoding="utf-8")
        probabilities = classify_log(log, self.model)
        self.assertEqual(next(iter(probabilities)), "data_versioning")
        self.assertEqual(set(probabilities), set(CATEGORIES))
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))

    def test_blank_and_unknown_vocabulary_do_not_get_a_category(self):
        for log in ("", "  \n", "qzxwvplm"):
            with self.subTest(log=log):
                self.assertEqual(classify_log(log, self.model), {})

    def test_changing_test_data_does_not_change_predictions(self):
        changed = copy.deepcopy(self.records)
        for record in changed:
            if record["split"] == "test":
                record["category"] = "data_drift"
                record["log"] = "Evidently drift " * 100
        other_model = train_classifier(changed)
        log = "dvc pull: remote cache missing"
        self.assertEqual(classify_log(log, self.model), classify_log(log, other_model))

    def test_rejects_duplicate_logs_across_splits_after_cleaning(self):
        records = copy.deepcopy(self.records)
        records[4]["log"] = records[0]["log"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.load_records(records)

    def test_rejects_invalid_labels_and_missing_fields(self):
        for field, value in (("category", "unknown"), ("split", "validation"),
                             ("provenance", "production-demo"), ("log", "")):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                with self.assertRaises(ValueError):
                    self.load_records(records)
        with self.assertRaisesRegex(ValueError, "required"):
            self.load_records([{"id": "bad"}])

    def test_requires_training_and_test_coverage_per_category(self):
        records = [record for record in self.records
                   if not (record["category"] == "data_drift" and record["split"] == "test")]
        with self.assertRaisesRegex(ValueError, "test logs"):
            self.load_records(records)

    def test_evaluation_reports_every_holdout_and_synthetic_provenance(self):
        result = evaluate(self.records)
        self.assertEqual(result["training_count"], 20)
        self.assertEqual(result["test_count"], 10)
        self.assertEqual(result["provenance_counts"], {"synthetic": 30})
        self.assertEqual({row["id"] for row in result["predictions"]},
                         {record["id"] for record in self.records if record["split"] == "test"})
        self.assertEqual(result["correct"], sum(row["correct"] for row in result["predictions"]))
        self.assertEqual(result["accuracy"], result["correct"] / 10)


if __name__ == "__main__":
    unittest.main()
