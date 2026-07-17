"""Console logging primitives for pipeline output.

Merged port of queering-nlp-bias `src/common/logging/log_primitives.py` and
`section_headers.py`, trimmed to the pieces this pipeline uses.
"""

from __future__ import annotations

HEADER_WIDTH = 60


def log(msg: str = "", end: str = "\n", gap: int = 0) -> None:
    """Print with immediate flush.

    Args:
        msg: Message to print
        end: Line ending (default newline)
        gap: Number of blank lines to print before the message
    """
    for _ in range(gap):
        print(flush=True)
    print(msg, end=end, flush=True)


def log_header(title: str, gap: int = 0) -> None:
    """Log a section header with double-line border."""
    log("═" * HEADER_WIDTH, gap=gap)
    log(title)
    log("═" * HEADER_WIDTH)


def log_stage(step: int, total: int, title: str) -> None:
    """Log a pipeline stage separator."""
    log("▓" * HEADER_WIDTH, gap=2)
    log(f"▓  STAGE {step}/{total}: {title}")
    log("▓" * HEADER_WIDTH)


def log_kv(fields: dict, indent: str = "  ") -> None:
    """Log key-value fields, skipping None values."""
    for key, value in fields.items():
        if value is not None:
            log(f"{indent}{key}: {value}")
