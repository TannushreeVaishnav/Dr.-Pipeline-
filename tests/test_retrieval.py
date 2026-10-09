import unittest
from pathlib import Path

from retrieval import load_knowledge_base, retrieve


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_knowledge_base(ROOT / "data" / "knowledge_base.json")
        cls.demo_log = (ROOT / "examples" / "dvc_pull_failed.log").read_text(encoding="utf-8")

    def test_demo_log_retrieves_documented_dvc_cloud_failure_first(self):
        results = retrieve(self.demo_log, self.chunks)
        self.assertEqual(results[0]["chunk"]["chunk_id"], "dvc-pull-cloud-failed")
        self.assertTrue(results[0]["chunk"]["source_url"].startswith("https://"))

    def test_unrelated_log_returns_no_false_match(self):
        self.assertEqual(retrieve("React button failed to render in browser", self.chunks), [])

    def test_limits_result_count_and_rejects_invalid_top_k(self):
        self.assertLessEqual(len(retrieve(self.demo_log, self.chunks, top_k=1)), 1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            retrieve(self.demo_log, self.chunks, top_k=0)


if __name__ == "__main__":
    unittest.main()
