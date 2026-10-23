
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.okf_linker import (
    find_related_concepts,
    load_concept_catalogue,
)
from src.validation.relationship_quality import (
    evaluate_linker_predictions,
    load_relationship_benchmark,
)


def main() -> int:
    benchmark_path = (
        PROJECT_ROOT / "tests" / "fixtures" / "relationship_benchmark.json"
    )
    examples = load_relationship_benchmark(benchmark_path)

    # Create a catalogue from the benchmark's source and candidate concepts.
    concepts_by_id = {}
    for example in examples:
        for field in ("source", "candidate"):
            concept = example[field]
            concepts_by_id[concept["id"]] = concept

    catalogue = [
        {
            **concept,
            "tags": [],
            "content": concept.get("description", ""),
            "filename": f"{concept['id']}.md",
            "path": "",
        }
        for concept in concepts_by_id.values()
    ]

    related_by_id = find_related_concepts(catalogue, max_related=3)
    result = evaluate_linker_predictions(examples, related_by_id)

    print(f"Benchmark examples: {len(examples)}")
    print(f"True positives:     {result['true_positives']}")
    print(f"False positives:    {result['false_positives']}")
    print(f"False negatives:    {result['false_negatives']}")
    print(f"Precision:          {result['precision']:.2%}")
    print(f"Recall:             {result['recall']:.2%}")
    print(f"F1 score:           {result['f1']:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
