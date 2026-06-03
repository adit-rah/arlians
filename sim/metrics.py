"""
MetricsLogger — per-year emergence metrics (build-spec §4.1).

Phase 0 scaffold: population tracking and deaths-by-cause counter structure.
Phase 1 additions: mean_displacement (settlement/nomadism signal),
  wild_food_mean (patch-depletion signal).
Later phases add: pct_calories_farmed, winter_survival_rate, settlement_clustering,
signal<->action MI, specialization index, gene-frequency drift, etc.

Outputs JSONL (one JSON object per year, newline-delimited), optionally to a file.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np


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

    Phase 1 new keys in year_summary()
    ------------------------------------
    ``mean_displacement``
        Mean Manhattan distance between each living agent's current (y, x) and
        the (y, x) it had when it was first observed alive this year (or its
        spawn position recorded on first ``record_step`` of each slot).
        Settlement → low value; nomadism → high value.

    ``wild_food_mean``
        Mean ``state.wild_remaining`` value across all land tiles (elevation >=
        sea_level) recorded each step and averaged over the year.  Drops as
        patches are depleted; recovers during regrowth steps.
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
            The live simulation object.  Reads from ``sim.store`` and
            ``sim.state``.
        """
        store = sim.store
        state = sim.state

        # --- population ---
        n = store.n_living_agents()
        self._pop_samples.append(n)

        # --- displacement ---
        # For each currently-alive (non-predator) slot, record spawn position
        # the first time we see it alive this year, then compute displacement
        # from that reference each subsequent step.
        living_mask = store.alive & ~store.is_predator
        live_idx = np.flatnonzero(living_mask)

        if live_idx.size > 0:
            # Register spawn positions for newly-seen alive slots
            new_slots = live_idx[~np.isin(live_idx, list(self._spawn_y.keys()))]
            for s in new_slots:
                self._spawn_y[int(s)] = int(store.y[s])
                self._spawn_x[int(s)] = int(store.x[s])

            # Compute mean Manhattan displacement for all currently-alive slots
            # that have a registered spawn position (should be all of live_idx now)
            sy = np.array([self._spawn_y.get(int(s), int(store.y[s])) for s in live_idx], dtype=np.float64)
            sx = np.array([self._spawn_x.get(int(s), int(store.x[s])) for s in live_idx], dtype=np.float64)
            dy = np.abs(store.y[live_idx].astype(np.float64) - sy)
            dx = np.abs(store.x[live_idx].astype(np.float64) - sx)
            mean_disp = float((dy + dx).mean())
        else:
            mean_disp = 0.0
        self._displacement_samples.append(mean_disp)

        # --- wild food mean across land tiles ---
        world = sim.world
        land_mask = world.elevation >= world.cfg.sea_level
        if land_mask.any():
            wild_mean = float(state.wild_remaining[land_mask].mean())
        else:
            wild_mean = 0.0
        self._wild_food_samples.append(wild_mean)

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
        - ``mean_displacement`` — mean Manhattan displacement of living agents
          from their year-start position (settlement/nomadism signal).
        - ``wild_food_mean``  — mean wild_remaining on land tiles averaged over
          steps this year (patch-depletion signal).

        The dict is intentionally open: later phases add keys
        (``pct_calories_farmed``, ``winter_survival_rate``,
        ``settlement_clustering``, ...) without reshaping existing structure.

        Returns
        -------
        dict
        """
        if self._pop_samples:
            arr = np.asarray(self._pop_samples, dtype=np.float64)
            pop_mean = float(arr.mean())
            pop_min  = int(arr.min())
            pop_max  = int(arr.max())
        else:
            pop_mean = 0.0
            pop_min  = 0
            pop_max  = 0

        if self._displacement_samples:
            mean_displacement = float(np.mean(self._displacement_samples))
        else:
            mean_displacement = 0.0

        if self._wild_food_samples:
            wild_food_mean = float(np.mean(self._wild_food_samples))
        else:
            wild_food_mean = 0.0

        summary: Dict[str, Any] = {
            "population_mean":  pop_mean,
            "population_min":   pop_min,
            "population_max":   pop_max,
            "births":           self.births,
            "deaths_by_cause":  dict(self.deaths),
            "mean_displacement": mean_displacement,
            "wild_food_mean":    wild_food_mean,
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
        # Phase 1 accumulators
        self._displacement_samples: list[float] = []
        self._wild_food_samples: list[float] = []
        # Spawn-position registry: slot_index -> (y, x) at first observation this year.
        # Cleared on year rollover so displacement resets each year.
        self._spawn_y: Dict[int, int] = {}
        self._spawn_x: Dict[int, int] = {}
