"""Lightweight timing/observability utilities for the AI pipeline.

Captures per-stage latency (OCR extraction, vision model call, etc.), the
model name used, and structured-output validation status, so every feature
response can show a small "Pipeline Metrics" panel. This is intentionally a
plain dataclass + context manager, not a tracing/telemetry stack - enough to
demonstrate the pipeline is observable without pulling in infrastructure a
single-process Streamlit app doesn't need.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    """Wall-clock duration of one named pipeline stage."""

    name: str
    duration_ms: float


@dataclass
class PipelineMetrics:
    """Observability summary for a single feature invocation."""

    model_name: str
    stages: list[StageTiming] = field(default_factory=list)
    validation_passed: bool = True
    retried: bool = False

    @property
    def total_latency_ms(self) -> float:
        return sum(stage.duration_ms for stage in self.stages)


@contextmanager
def timed_stage(stages: list[StageTiming], name: str):
    """Record the wall-clock duration of the enclosed block as a named stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        stages.append(StageTiming(name=name, duration_ms=(time.perf_counter() - start) * 1000))
