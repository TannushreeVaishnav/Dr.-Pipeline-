import re
from pathlib import Path


MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_LOG_SIZE = "2 MiB"
ANSI_ESCAPE = re.compile(r"\x1B(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
TIMESTAMP_PREFIX = re.compile(
    r"^(\s*)\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?\s?"
)
RUN_ID = re.compile(r"\b((?:workflow[_ -]?)?run[_ -]?id)\s*([:=])\s*[\w.-]+", re.I)


def validate_log_size(size_bytes: int) -> None:
    if size_bytes > MAX_LOG_BYTES:
        raise ValueError(f"The log exceeds the {MAX_LOG_SIZE} limit. Reduce its size and try again.")


def decode_uploaded_log(filename: str, data: bytes) -> str:
    if Path(filename).suffix.lower() not in {".log", ".txt"}:
        raise ValueError("Upload a .log or .txt file.")
    validate_log_size(len(data))
    if b"\0" in data:
        raise ValueError("The uploaded file appears to be binary. Upload a UTF-8 text log.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The uploaded file is not valid UTF-8 text.") from error


def clean_log(raw: str) -> str:
    """Remove terminal formatting and redact common volatile log identifiers."""
    text = ANSI_ESCAPE.sub("", raw).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = []
    for line in text.split("\n"):
        line = TIMESTAMP_PREFIX.sub(r"\1<timestamp> ", line)
        cleaned.append(RUN_ID.sub(r"\1\2<redacted>", line))
    return "\n".join(cleaned).strip("\n")
