"""Shared data structure for extracted document text and metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentRecord:
    """
    Standard representation of an extracted document.

    Attributes:
        text: Extracted textual content.
        metadata: Information about the original source.
    """

    text: str
    metadata: dict[str, Any]
