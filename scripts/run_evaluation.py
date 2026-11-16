"""Command-line runner for the project retrieval and answer evaluation dataset."""

from pathlib import Path

from hybrid_rag_cybersecurity.evaluation.evaluator import RAGEvaluator
from hybrid_rag_cybersecurity.evaluation.test_dataset import (
    EVALUATION_DATASET,
)


def print_question_result(result: dict) -> None:
    """Display the evaluation result for one question."""

    print("\n" + "-" * 65)
    print(f"{result['id']}: {result['question']}")
    print("-" * 65)

    print(f"Retrieved sources: {result['retrieved_sources']}")

    if result["keyword_coverage"] is not None:
        print(f"Keyword coverage: {result['keyword_coverage']:.2%}")

    if result["source_coverage"] is not None:
        print(f"Source coverage: {result['source_coverage']:.2%}")

    validation = result["validation"]

    print("Structural validation: " + ("PASSED" if validation["valid"] else "FAILED"))

    if validation["unknown_citations"]:
        print(f"Unknown citations: {validation['unknown_citations']}")

    if validation["unsupported_identifiers"]:
        print(f"Unsupported NIST identifiers: {validation['unsupported_identifiers']}")

    warnings = validation["uncited_claim_lines"]

    if warnings:
        print(f"Citation review warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nAnswer:\n{result['answer']}")


def print_summary(results: list[dict]) -> None:
    """Display aggregate evaluation metrics."""

    answerable_results = [result for result in results if result["answerable"]]

    unanswerable_results = [result for result in results if not result["answerable"]]

    keyword_scores = [
        result["keyword_coverage"]
        for result in answerable_results
        if result["keyword_coverage"] is not None
    ]

    source_scores = [
        result["source_coverage"]
        for result in answerable_results
        if result["source_coverage"] is not None
    ]

    structural_passes = sum(
        result["validation"]["valid"] for result in answerable_results
    )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    if keyword_scores:
        average = sum(keyword_scores) / len(keyword_scores)
        print(f"Average keyword coverage: {average:.2%}")

    if source_scores:
        average = sum(source_scores) / len(source_scores)
        print(f"Average source coverage: {average:.2%}")

    if answerable_results:
        print(
            "Structural validation: "
            f"{structural_passes}/{len(answerable_results)} passed"
        )

    print(f"Total questions: {len(results)}")
    print(f"Answerable questions: {len(answerable_results)}")
    print(f"Unanswerable questions: {len(unanswerable_results)}")

    print(
        "\nNote: Structural validation checks citation labels and "
        "NIST identifier presence. Citation warnings are advisory. "
        "These checks do not establish factual correctness or "
        "prove that a citation supports its associated claim."
    )


# Run the workflow described in this module overview.
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    print("=" * 65)
    print("SENTINEL CYBERSECURITY - HYBRID RAG EVALUATION")
    print("=" * 65)

    evaluator = RAGEvaluator(project_root)
    results = []

    for test_case in EVALUATION_DATASET:
        print(f"\nEvaluating {test_case['id']}...")

        result = evaluator.evaluate_question(test_case)
        results.append(result)

        print_question_result(result)

    print_summary(results)


if __name__ == "__main__":
    main()
