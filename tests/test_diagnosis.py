import unittest
from types import SimpleNamespace

from diagnosis import Diagnosis, DiagnosisError, SupportedClaim, generate_diagnosis


MATCH = {
    "chunk": {
        "chunk_id": "dvc-pull-cloud-failed",
        "tool": "DVC",
        "title": "Failed to pull data from the cloud",
        "symptom": "dvc pull cannot fetch tracked data.",
        "guidance": "Run dvc push from the source project, then retry dvc pull.",
    }
}


class FakeResponses:
    def __init__(self, diagnosis):
        self.diagnosis = diagnosis
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_parsed=self.diagnosis)


class DiagnosisTests(unittest.TestCase):
    def make_client(self, diagnosis):
        client = SimpleNamespace()
        client.responses = FakeResponses(diagnosis)
        return client

    def test_returns_source_cited_diagnosis_and_limits_output(self):
        diagnosis = Diagnosis(
            status="diagnosis",
            root_cause=SupportedClaim(text="The remote may not contain the tracked data.", citation_ids=["dvc-pull-cloud-failed"]),
            recommended_fix=SupportedClaim(text="Run dvc push from the source project.", citation_ids=["dvc-pull-cloud-failed"]),
            missing_information=[],
        )
        client = self.make_client(diagnosis)

        result = generate_diagnosis(client, "test-model", "ERROR: failed to pull data", [MATCH])

        self.assertEqual(result.root_cause.text, diagnosis.root_cause.text)
        self.assertEqual(client.responses.request["max_output_tokens"], 500)

    def test_rejects_citation_not_in_retrieved_evidence(self):
        diagnosis = Diagnosis(
            status="diagnosis",
            root_cause=SupportedClaim(text="Wrong source.", citation_ids=["invented-source"]),
            recommended_fix=SupportedClaim(text="Run a command.", citation_ids=["dvc-pull-cloud-failed"]),
            missing_information=[],
        )
        with self.assertRaisesRegex(DiagnosisError, "not retrieved"):
            generate_diagnosis(self.make_client(diagnosis), "test-model", "failure", [MATCH])

    def test_supports_insufficient_evidence_without_claims(self):
        diagnosis = Diagnosis(
            status="insufficient_evidence",
            root_cause=None,
            recommended_fix=None,
            missing_information=["Provide the configured remote name."],
        )
        result = generate_diagnosis(self.make_client(diagnosis), "test-model", "failure", [MATCH])
        self.assertEqual(result.status, "insufficient_evidence")

    def test_rejects_invalid_structured_provider_output(self):
        with self.assertRaisesRegex(DiagnosisError, "did not match"):
            generate_diagnosis(self.make_client({"status": "diagnosis"}), "test-model", "failure", [MATCH])

    def test_rejects_oversized_log_before_calling_provider(self):
        client = self.make_client(None)
        with self.assertRaisesRegex(DiagnosisError, "too long"):
            generate_diagnosis(client, "test-model", "x" * 12_001, [MATCH])
        self.assertIsNone(client.responses.request)


if __name__ == "__main__":
    unittest.main()
