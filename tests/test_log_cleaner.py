import unittest

from log_cleaner import MAX_LOG_BYTES, clean_log, decode_uploaded_log, validate_log_size


class CleanLogTests(unittest.TestCase):
    def test_redacts_timestamp_and_run_id(self):
        raw = "2026-10-21T10:11:12.123Z run_id=4812 ERROR failed"
        self.assertEqual(
            clean_log(raw),
            "<timestamp> run_id=<redacted> ERROR failed",
        )

    def test_removes_ansi_and_normalizes_line_endings(self):
        raw = "before\r\n\x1b[31mERROR\x1b[0m\r\nafter"
        self.assertEqual(clean_log(raw), "before\nERROR\nafter")

    def test_preserves_non_error_context(self):
        raw = "step 1\nstep 2\nstep 3"
        self.assertEqual(clean_log(raw), raw)

    def test_redacts_bracketed_timestamp(self):
        self.assertEqual(
            clean_log("[2026-10-21 10:11:12] ERROR"),
            "<timestamp> ERROR",
        )

    def test_accepts_utf8_text_upload_with_bom(self):
        self.assertEqual(decode_uploaded_log("run.LOG", b"\xef\xbb\xbfhello"), "hello")

    def test_rejects_unsupported_file_type(self):
        with self.assertRaisesRegex(ValueError, r"\.log or \.txt"):
            decode_uploaded_log("run.csv", b"hello")

    def test_rejects_binary_upload(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            decode_uploaded_log("run.log", b"hello\0world")

    def test_rejects_invalid_utf8_upload(self):
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            decode_uploaded_log("run.log", b"\xff")

    def test_rejects_oversized_input(self):
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            validate_log_size(MAX_LOG_BYTES + 1)


if __name__ == "__main__":
    unittest.main()
