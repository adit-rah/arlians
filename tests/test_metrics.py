"""
Phase-0 MetricsLogger tests.

Covers:
  - record_step accumulates correctly.
  - year_summary returns dict with all required keys.
  - JSONL line is written and parses as valid JSON with expected keys.
  - Accumulators reset after year_summary so a second summary is independent.
  - Zero-step year_summary does not crash.
"""
import json
import os
import tempfile

import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation
from sim.metrics import MetricsLogger


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_sim():
    world = World.generate(WorldConfig(width=64, height=64, seed=3))
    cfg   = SimConfig(max_agents=128, init_agents=32)
    sim   = Simulation(world, cfg)
    sim.reset(seed=1)
    return sim


# ---------------------------------------------------------------------------
# record_step + year_summary dict shape
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "population_mean",
    "population_min",
    "population_max",
    "births",
    "deaths_by_cause",
}

_DEATH_CAUSES = {"starve", "dehydrate", "exposure", "predator", "conflict", "age"}


def test_year_summary_has_required_keys(small_sim):
    logger = MetricsLogger()
    for _ in range(10):
        logger.record_step(small_sim)
    summary = logger.year_summary()

    assert _REQUIRED_KEYS <= set(summary.keys()), (
        f"Missing keys: {_REQUIRED_KEYS - set(summary.keys())}"
    )


def test_deaths_by_cause_has_all_causes(small_sim):
    logger = MetricsLogger()
    logger.record_step(small_sim)
    summary = logger.year_summary()

    assert _DEATH_CAUSES <= set(summary["deaths_by_cause"].keys()), (
        f"Missing death causes: {_DEATH_CAUSES - set(summary['deaths_by_cause'].keys())}"
    )


def test_population_stats_are_consistent(small_sim):
    """population_min <= population_mean <= population_max."""
    logger = MetricsLogger()
    for _ in range(20):
        logger.record_step(small_sim)
    s = logger.year_summary()

    assert s["population_min"] <= s["population_mean"] <= s["population_max"]


def test_population_values_match_store(small_sim):
    """Recorded values should reflect the actual living-agent count."""
    logger = MetricsLogger()
    n = small_sim.store.n_living_agents()
    logger.record_step(small_sim)
    s = logger.year_summary()

    assert s["population_min"] == n
    assert s["population_max"] == n
    import math
    assert math.isclose(s["population_mean"], n)


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def test_jsonl_line_written_and_parses(small_sim):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as fh:
        path = fh.name

    try:
        logger = MetricsLogger(out_path=path)
        for _ in range(5):
            logger.record_step(small_sim)
        summary = logger.year_summary()

        # File should exist and have content
        assert os.path.getsize(path) > 0

        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        assert len(lines) == 1, f"Expected 1 JSONL line, got {len(lines)}"
        parsed = json.loads(lines[0])
        assert _REQUIRED_KEYS <= set(parsed.keys())
    finally:
        os.unlink(path)


def test_jsonl_appends_multiple_years(small_sim):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as fh:
        path = fh.name

    try:
        logger = MetricsLogger(out_path=path)
        for year in range(3):
            for _ in range(5):
                logger.record_step(small_sim)
            logger.year_summary()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        assert len(lines) == 3, f"Expected 3 JSONL lines, got {len(lines)}"
        for line in lines:
            json.loads(line)   # each line must be valid JSON
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Accumulator reset
# ---------------------------------------------------------------------------

def test_accumulators_reset_after_year_summary(small_sim):
    """Second year_summary without any record_step should give zero/empty stats."""
    logger = MetricsLogger()
    for _ in range(5):
        logger.record_step(small_sim)
    logger.year_summary()  # consumes the 5 steps

    # No additional record_step calls — should return zero stats
    s2 = logger.year_summary()
    assert s2["population_mean"] == 0.0
    assert s2["population_min"] == 0
    assert s2["population_max"] == 0
    assert s2["births"] == 0
    assert all(v == 0 for v in s2["deaths_by_cause"].values())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_year_summary_with_no_steps_does_not_crash():
    logger = MetricsLogger()
    s = logger.year_summary()
    assert s["population_mean"] == 0.0


def test_logger_without_out_path_does_not_crash(small_sim):
    logger = MetricsLogger()   # no out_path
    for _ in range(3):
        logger.record_step(small_sim)
    s = logger.year_summary()
    assert "population_mean" in s
