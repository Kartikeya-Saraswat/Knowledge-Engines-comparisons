
import json
from pathlib import Path


def load_relationship_benchmark(
    path: str | Path,
) -> list[dict]:
    """Load labelled source-candidate relationship examples."""
    benchmark_path = Path(path)

    with benchmark_path.open(encoding="utf-8") as file:
        examples = json.load(file)

    if not isinstance(examples, list):
        raise ValueError("Benchmark must be a JSON list.")

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Example {index} must be an object.")

        if not isinstance(example.get("related"), bool):
            raise ValueError(
                f"Example {index} must have a boolean 'related' label."
            )

        for field in ("source", "candidate"):
            concept = example.get(field)
            if not isinstance(concept, dict) or not concept.get("id"):
                raise ValueError(
                    f"Example {index} must contain {field} metadata "
                    "with an ID."
                )

    return examples


def evaluate_relationships(
    examples: list[dict],
    predicted_related: set[tuple[str, str]],
) -> dict:
    """Evaluate predicted directed concept pairs against labelled examples."""
    expected_related = {
        (example["source"]["id"], example["candidate"]["id"])
        for example in examples
        if example["related"]
    }

    labelled_pairs = {
        (example["source"]["id"], example["candidate"]["id"])
        for example in examples
    }

    unknown_predictions = predicted_related - labelled_pairs
    if unknown_predictions:
        raise ValueError(
            "Predictions contain pairs absent from the benchmark: "
            f"{sorted(unknown_predictions)}"
        )

    true_positives = len(predicted_related & expected_related)
    false_positives = len(predicted_related - expected_related)
    false_negatives = len(expected_related - predicted_related)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_linker_predictions(
    examples: list[dict],
    related_by_id: dict[str, list[str]],
) -> dict:
    """Evaluate linker output against labelled benchmark pairs."""
    labelled_pairs = {
        (example["source"]["id"], example["candidate"]["id"])
        for example in examples
    }

    predicted_pairs = {
        (source_id, target_id)
        for source_id, target_ids in related_by_id.items()
        for target_id in target_ids
    }

    # Only score predictions whose direction is explicitly labelled.
    benchmark_predictions = predicted_pairs & labelled_pairs

    return evaluate_relationships(examples, benchmark_predictions)
