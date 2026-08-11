"""Tests for src.observability.metrics."""

import time

from src.observability.metrics import PipelineMetrics, StageTiming, timed_stage


def test_timed_stage_records_a_positive_duration():
    stages: list[StageTiming] = []

    with timed_stage(stages, "test_stage"):
        time.sleep(0.01)

    assert len(stages) == 1
    assert stages[0].name == "test_stage"
    assert stages[0].duration_ms > 0


def test_pipeline_metrics_total_latency_sums_all_stages():
    metrics = PipelineMetrics(
        model_name="gemini-flash-latest",
        stages=[StageTiming(name="a", duration_ms=10.0), StageTiming(name="b", duration_ms=25.0)],
    )

    assert metrics.total_latency_ms == 35.0


def test_pipeline_metrics_defaults():
    metrics = PipelineMetrics(model_name="gemini-flash-latest")

    assert metrics.stages == []
    assert metrics.validation_passed is True
    assert metrics.retried is False
    assert metrics.total_latency_ms == 0.0
