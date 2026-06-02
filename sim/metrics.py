"""
MetricsLogger — per-year emergence metrics (build-spec §4.1).

Phase 0 scaffold: population tracking and deaths-by-cause counter structure.
Later phases add: pct_calories_farmed, winter_survival_rate, settlement_clustering,
signal<->action MI, specialization index, gene-frequency drift, etc.

Outputs JSONL (one JSON object per year, newline-delimited), optionally to a file.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


# Death causes tracked from Phase 1 onwards.  Phase-0 counters stay at zero
# until the relevant damage logic is wired in.
_DEATH_CAUSES = ("starve", "dehydrate", "exposure", "predator", "conflict", "age")


class MetricsLogger:
    """Accumulate per-step simulation statistics and emit per-year summaries.

    Parameters
    ----------
    out_path : str or None
        If provided, ``year_summary`` appends one JSON line per year to this
        file (creates/appends, never truncates).

    Usage
    -----
    ::

        logger = MetricsLogger(out_path="runs/metrics.jsonl")
        # inside training loop:
        logger.record_step(sim)
        if step % season_period == 0:
            summary = logger.year_summary()

    Later phases increment ``logger.deaths["starve"] += 1`` etc. directly after
    a kill event before calling ``record_step``, so the per-year dict aggregates
    correctly.
    """

    def __init__(self, out_path: Optional[str] = None) -> None:
        self.out_path = out_path
        self._reset_year()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(self, sim) -> None:
        """Accumulate statistics for the current simulation step.

        Parameters
        ----------
        sim : sim.simulation.Simulation
            The live simulation object.  Only reads from ``sim.store``.
        """
        n = sim.store.n_living_agents()
        self._pop_samples.append(n)

    def year_summary(self) -> Dict[str, Any]:
        """Aggregate the accumulated steps into a per-year statistics dict.

        Appends a single JSON line to ``out_path`` if set, then resets
        per-year accumulators.

        The returned dict always contains:

        - ``population_mean`` — mean living-agent count over accumulated steps.
        - ``population_min``  — minimum.
        - ``population_max``  — maximum.
        - ``births``          — total births this year (Phase 5+).
        - ``deaths_by_cause`` — dict keyed by cause string, all zero until the
          relevant phase wires in the increment calls.

        The dict is intentionally open: later phases add keys
        (``pct_calories_farmed``, ``winter_survival_rate``,
        ``settlement_clustering``, ...) without reshaping existing structure.

        Returns
        -------
        dict
        """
        if self._pop_samples:
            import numpy as np
            arr = np.asarray(self._pop_samples, dtype=np.float64)
            pop_mean = float(arr.mean())
            pop_min  = int(arr.min())
            pop_max  = int(arr.max())
        else:
            pop_mean = 0.0
            pop_min  = 0
            pop_max  = 0

        summary: Dict[str, Any] = {
            "population_mean":  pop_mean,
            "population_min":   pop_min,
            "population_max":   pop_max,
            "births":           self.births,
            "deaths_by_cause":  dict(self.deaths),
        }

        if self.out_path is not None:
            with open(self.out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")

        self._reset_year()
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_year(self) -> None:
        """Reset all per-year accumulators."""
        self._pop_samples: list[int] = []
        self.births: int = 0
        self.deaths: Dict[str, int] = {cause: 0 for cause in _DEATH_CAUSES}
