"""Tokenizer that keeps security identifiers intact for BM25."""

import re


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text while preserving cybersecurity identifiers.

    Examples:
        CVE-2025-1001 -> cve-2025-1001
        10.0.0.10     -> 10.0.0.10
        Asset:        -> asset
        Server?       -> server
    """

    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-._:/][a-zA-Z0-9]+)*",
        text,
    )

    return [token.lower() for token in tokens]
