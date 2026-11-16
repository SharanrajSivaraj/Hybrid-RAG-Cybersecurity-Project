"""Stable document identity and reciprocal-rank fusion helpers."""

import hashlib
import json
from collections import defaultdict

from langchain_core.documents import Document


def get_document_key(document: Document) -> str:
    """
    Generate a stable key to identify the same chunk
    across different retrieval systems.
    """

    content = {
        "page_content": document.page_content,
        "metadata": document.metadata,
    }

    serialised = json.dumps(
        content,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[Document, float]]:
    """
    Combine multiple ranked document lists using RRF.

    Returns documents sorted by their combined RRF score.
    """

    scores = defaultdict(float)
    documents = {}

    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            document_key = get_document_key(document)

            scores[document_key] += 1 / (k + rank)

            documents[document_key] = document

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [(documents[key], scores[key]) for key in ranked_keys[:top_n]]
