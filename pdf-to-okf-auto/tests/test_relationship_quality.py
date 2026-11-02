
import pytest
from src.validation.relationship_quality import evaluate_linker_predictions
from src.validation.relationship_quality import (
    evaluate_relationships,
    load_relationship_benchmark,
)


BENCHMARK_PATH = "tests/fixtures/relationship_benchmark.json"


def test_loads_five_benchmark_examples():
    examples = load_relationship_benchmark(BENCHMARK_PATH)

    assert len(examples) == 5
    assert sum(example["related"] for example in examples) == 3


def test_perfect_predictions_score_one():
    examples = load_relationship_benchmark(BENCHMARK_PATH)
    predictions = {
        (example["source"]["id"], example["candidate"]["id"])
        for example in examples
        if example["related"]
    }

    result = evaluate_relationships(examples, predictions)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_incorrect_prediction_is_penalized():
    examples = load_relationship_benchmark(BENCHMARK_PATH)
    predictions = {
        ("KC-SOIL-001", "KC-SOIL-002"),
        ("KC-SOIL-001", "KC-ECON-001"),
    }

    result = evaluate_relationships(examples, predictions)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1 / 3)


def test_rejects_predictions_outside_benchmark():
    examples = load_relationship_benchmark(BENCHMARK_PATH)

    with pytest.raises(ValueError, match="absent from the benchmark"):
        evaluate_relationships(
            examples,
            {("KC-SOIL-001", "KC-UNKNOWN-001")},
        )


def test_evaluates_actual_linker_output():
    examples = load_relationship_benchmark(BENCHMARK_PATH)

    related_by_id = {
        "KC-SOIL-001": ["KC-SOIL-002", "KC-ECON-001"],
        "KC-RESOURCE-001": ["KC-RESOURCE-002"],
    }

    result = evaluate_linker_predictions(examples, related_by_id)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
