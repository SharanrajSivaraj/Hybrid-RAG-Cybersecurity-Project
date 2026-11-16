"""Structural checks for citation labels, identifiers, and uncited claims."""

import re

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
NIST_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2}\.[A-Z]{2}(?:-\d{2})?\b")

# These are framing statements, headings, or abstention statements.
# They are not treated as uncited factual claims.
FRAMING_PATTERNS = [
    re.compile(
        r"^the seven steps in the incident response process.*:$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the severity and vulnerability type of .* are as follows:$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^each step is explicitly described in the provided sources\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the documents do not provide additional details about this specific cve\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the available documents do not provide sufficient information "
        r"about this\.?$",
        re.IGNORECASE,
    ),
]


def extract_citations(text: str) -> set[str]:
    """Extract citation labels such as S1 and S2."""
    return set(CITATION_PATTERN.findall(text))


def extract_nist_identifiers(text: str) -> set[str]:
    """Extract NIST identifiers from text."""
    return set(NIST_IDENTIFIER_PATTERN.findall(text))


def is_non_claim_line(line: str) -> bool:
    """Identify headings, list labels, and common framing statements."""

    cleaned = line.strip()

    if not cleaned:
        return True

    # Markdown headings.
    if cleaned.startswith("#"):
        return True

    # Horizontal separators.
    if cleaned in {"---", "***", "___"}:
        return True

    # Numbered or bulleted bold labels, such as:
    # 3. **Incident Response Reporting and Communication (RS.CO)**
    if re.match(
        r"^\s*(?:\d+\.|[-*])\s+\*\*.+\*\*\s*$",
        cleaned,
    ):
        return True

    # Common framing or abstention statements.
    return bool(any(pattern.match(cleaned) for pattern in FRAMING_PATTERNS))


def find_uncited_claim_lines(answer: str) -> list[str]:
    """
    Identify potentially uncited factual statements.

    This is a heuristic warning system, not a semantic claim detector.
    """

    uncited_lines = []

    for line in answer.splitlines():
        cleaned = line.strip()

        if is_non_claim_line(cleaned):
            continue

        # Ignore short lines because they are often labels or fragments.
        if len(cleaned) < 50:
            continue

        if not CITATION_PATTERN.search(cleaned):
            uncited_lines.append(cleaned)

    return uncited_lines


def validate_answer(answer: str, context: str) -> dict:
    """
    Perform deterministic structural validation.

    Hard checks:
    1. Citation labels must exist in the supplied context.
    2. NIST identifiers must exist in the supplied context.

    Warning:
    3. Substantial answer lines without citations are reported
       for manual review.

    These checks do not establish semantic faithfulness.
    """

    available_citations = extract_citations(context)
    answer_citations = extract_citations(answer)

    unknown_citations = sorted(answer_citations - available_citations)

    available_identifiers = extract_nist_identifiers(context)
    answer_identifiers = extract_nist_identifiers(answer)

    unsupported_identifiers = sorted(answer_identifiers - available_identifiers)

    uncited_claim_lines = find_uncited_claim_lines(answer)

    # Only hard structural failures determine this status.
    structural_valid = not unknown_citations and not unsupported_identifiers

    return {
        "valid": structural_valid,
        "available_citations": sorted(available_citations),
        "answer_citations": sorted(answer_citations),
        "unknown_citations": unknown_citations,
        "available_identifiers": sorted(available_identifiers),
        "answer_identifiers": sorted(answer_identifiers),
        "unsupported_identifiers": unsupported_identifiers,
        "uncited_claim_lines": uncited_claim_lines,
    }
