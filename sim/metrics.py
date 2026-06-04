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

    Phase 2 new keys in year_summary()
    ------------------------------------
    ``water_occupancy``
        Fraction of living-agent-steps spent on tiles with
        ``world.base_resources["water_proximity"] >= 0.6``.  Settlement near
        water → high value; random/inland wandering → low value.  In [0, 1].

    Phase 3 new keys in year_summary()
    ------------------------------------
    ``pct_calories_farmed``
        Fraction of total food calories (foraged + harvested) that came from
        harvested crops this year.  0.0 if no food was consumed at all.
        In [0, 1].

    ``fertile_occupancy``
        Fraction of living-agent-steps spent on tiles with
        ``world.base_resources["soil_fertility"] >= 0.6``.  Settlement-on-
        fertile signal; mirrors how ``water_occupancy`` is computed.
        In [0, 1].

    Phase 4 new keys in year_summary()
    ------------------------------------
    ``structures_built``
        Dict with keys ``total``, ``shelters``, ``storage``, and ``max_total``
        describing structure counts observed during the year.  ``total`` is the
        count at the final recorded step; ``max_total`` is the peak seen at any
        step (useful when structures decay).

    ``stored_food_total``
        Dict with keys ``mean`` (mean of ``sum(state.stored_food)`` over steps
        this year) and ``max`` (peak sum seen at any step — should peak in fall
        and be drawn down in winter if agents are banking food strategically).

    ``mean_thermal``
        Mean ``thermal`` drive value over living (non-predator) agents, averaged
        across all steps this year.  Low in winter for unsheltered populations,
        higher for sheltered ones.  In [0, 1].

    ``deaths_by_cause["exposure"]``
        Already tracked from Phase 1 — confirmed surfaced here.  Incremented by
        callers via ``logger.deaths["exposure"] += 1`` before ``record_step``.

    Phase 6 new keys in year_summary()
    ------------------------------------
    ``n_predators``
        Mean predator count alive per step this year.  Tracks predator-prey
        population dynamics; should oscillate without total collapse.

    ``walls_built``
        Dict with keys ``end_of_year`` (WALL tile count at final recorded step)
        and ``max`` (peak WALL tile count at any step this year).

    ``weapons_held``
        Mean count of living (non-predator) agents holding ``weapon==True``
        per step this year.

    ``predation_deaths`` and ``conflict_deaths``
        Convenience top-level aliases for ``deaths_by_cause["predator"]`` and
        ``deaths_by_cause["conflict"]``; surfaced for easy access without
        traversing the nested dict.
    """

    def __init__(self, out_path: Optional[str] = None) -> None:
        self.out_path = out_path
        self._reset_year()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(self, sim, info=None) -> None:
        """Accumulate statistics for the current simulation step.

        Parameters
        ----------
        sim : sim.simulation.Simulation
            The live simulation object.  Reads from ``sim.store`` and
            ``sim.state``.
        info : dict or None
            Optional per-step info dict from ``StepOut.info``.  When provided,
            ``info.get("foraged", 0)`` and ``info.get("harvested", 0)`` are
            accumulated into the calorie-source-split totals for the year.
            Existing single-argument call sites pass nothing and are unaffected.
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

        # --- water occupancy ---
        # Fraction of living-agent-steps spent on tiles with
        # world.base_resources["water_proximity"] >= 0.6.
        # Accumulate a (n_steps,) running numerator and denominator so we
        # can average correctly over the year even if population varies.
        living_mask = store.alive & ~store.is_predator
        live_idx = np.flatnonzero(living_mask)
        if live_idx.size > 0:
            water_prox = world.base_resources["water_proximity"]
            ys = store.y[live_idx]
            xs = store.x[live_idx]
            on_water = float((water_prox[ys, xs] >= 0.6).sum())
            total = float(live_idx.size)
        else:
            on_water = 0.0
            total = 0.0
        self._water_on_steps += on_water
        self._water_total_steps += total

        # --- fertile occupancy (Phase 3) ---
        # Fraction of living-agent-steps spent on tiles with soil_fertility >= 0.6.
        # Settlement-on-fertile signal (mirrors water_occupancy pattern).
        if live_idx.size > 0:
            soil_fert = world.base_resources["soil_fertility"]
            ys = store.y[live_idx]
            xs = store.x[live_idx]
            on_fertile = float((soil_fert[ys, xs] >= 0.6).sum())
            fertile_total = float(live_idx.size)
        else:
            on_fertile = 0.0
            fertile_total = 0.0
        self._fertile_on_steps += on_fertile
        self._fertile_total_steps += fertile_total

        # --- calorie-source split (Phase 3) ---
        # Accumulate foraged and harvested food totals from the per-step info dict
        # so we can compute pct_calories_farmed in year_summary.
        if info is not None:
            self._foraged_total  += float(info.get("foraged",  0))
            self._harvested_total += float(info.get("harvested", 0))

        # --- Phase 4: structure counts ---
        # Count structures on the map: total, shelters (type==1), storage (type==2).
        struct = state.structure_type  # (H, W) int8
        total_structs   = int(np.count_nonzero(struct))
        shelter_count   = int(np.sum(struct == 1))
        storage_count   = int(np.sum(struct == 2))
        self._struct_total_samples.append(total_structs)
        self._struct_shelter_samples.append(shelter_count)
        self._struct_storage_samples.append(storage_count)

        # --- Phase 4: stored food total ---
        stored_sum = float(state.stored_food.sum())
        self._stored_food_samples.append(stored_sum)

        # --- Phase 4: mean thermal over living agents ---
        if live_idx.size > 0:
            mean_th = float(store.thermal[live_idx].mean())
        else:
            mean_th = 0.0
        self._thermal_samples.append(mean_th)

        # --- Phase 5: births (from step info), genome drift, lineage diversity ---
        if info is not None:
            self.births += int(info.get("births", 0))
        if live_idx.size > 0:
            self._genome_samples.append(store.genome[live_idx].mean(axis=0))
            self._lineage_count_samples.append(int(np.unique(store.lineage_id[live_idx]).size))

        # --- Phase 6: predator count, walls, weapons ---
        # n_predators: read from info if available (avoids recomputing), else count directly.
        if info is not None and "n_predators" in info:
            n_pred = int(info["n_predators"])
        else:
            n_pred = int(np.sum(store.alive & store.is_predator))
        self._pred_count_samples.append(n_pred)

        # walls_built: count WALL tiles (structure_type == 3) on the map.
        wall_count = int(np.sum(state.structure_type == 3))
        self._wall_count_samples.append(wall_count)

        # weapons_held: count living (non-predator) agents with weapon==True.
        if live_idx.size > 0:
            weapons_now = int(np.sum(store.weapon[live_idx]))
        else:
            weapons_now = 0
        self._weapons_held_samples.append(weapons_now)

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
        - ``water_occupancy`` — fraction of living-agent-steps on tiles with
          water_proximity >= 0.6 (water-anchoring signal, Phase 2). In [0, 1].
        - ``pct_calories_farmed`` — fraction of total food from harvested crops
          this year (calorie-source split, Phase 3). In [0, 1].
        - ``fertile_occupancy`` — fraction of living-agent-steps on tiles with
          soil_fertility >= 0.6 (settlement-on-fertile signal, Phase 3). In [0, 1].

        The dict is intentionally open: later phases add keys
        (``winter_survival_rate``, ``settlement_clustering``, ...) without
        reshaping existing structure.

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

        if self._water_total_steps > 0:
            water_occupancy = float(self._water_on_steps / self._water_total_steps)
        else:
            water_occupancy = 0.0

        # --- Phase 3 metrics ---
        # pct_calories_farmed: fraction of food calories from harvested crops.
        total_food = self._harvested_total + self._foraged_total
        if total_food > 0.0:
            pct_calories_farmed = float(self._harvested_total / total_food)
        else:
            pct_calories_farmed = 0.0

        # fertile_occupancy: fraction of living-agent-steps on soil_fertility >= 0.6 tiles.
        if self._fertile_total_steps > 0:
            fertile_occupancy = float(self._fertile_on_steps / self._fertile_total_steps)
        else:
            fertile_occupancy = 0.0

        # --- Phase 4 metrics ---
        # structures_built: track end-of-year count (last sample) and peak (max sample).
        if self._struct_total_samples:
            end_total    = self._struct_total_samples[-1]
            max_total    = int(max(self._struct_total_samples))
            end_shelters = self._struct_shelter_samples[-1]
            end_storage  = self._struct_storage_samples[-1]
        else:
            end_total = max_total = end_shelters = end_storage = 0

        structures_built = {
            "total":      end_total,
            "max_total":  max_total,
            "shelters":   end_shelters,
            "storage":    end_storage,
        }

        # stored_food_total: mean and max of per-step sums over the year.
        if self._stored_food_samples:
            sf_arr = np.asarray(self._stored_food_samples, dtype=np.float64)
            stored_food_total = {
                "mean": float(sf_arr.mean()),
                "max":  float(sf_arr.max()),
            }
        else:
            stored_food_total = {"mean": 0.0, "max": 0.0}

        # mean_thermal: mean of per-step mean-thermals over the year.
        if self._thermal_samples:
            mean_thermal = float(np.mean(self._thermal_samples))
        else:
            mean_thermal = 0.0

        # --- Phase 5 metrics ---
        # mean_genome: per-gene mean over living agents, averaged across steps.
        # genome_drift: mean |gene - 0.5| (how far the population has evolved from
        # the neutral starting genome). lineage_count: mean distinct lineages alive.
        if self._genome_samples:
            mean_genome = np.mean(np.stack(self._genome_samples), axis=0)
            mean_genome_list = [float(g) for g in mean_genome]
            genome_drift = float(np.abs(mean_genome - 0.5).mean())
        else:
            mean_genome_list = []
            genome_drift = 0.0
        if self._lineage_count_samples:
            lineage_count = float(np.mean(self._lineage_count_samples))
        else:
            lineage_count = 0.0

        # --- Phase 6 metrics ---
        # n_predators: mean predator count per step.
        if self._pred_count_samples:
            n_predators = float(np.mean(self._pred_count_samples))
        else:
            n_predators = 0.0

        # walls_built: end-of-year count and peak over the year.
        if self._wall_count_samples:
            walls_end = int(self._wall_count_samples[-1])
            walls_max = int(max(self._wall_count_samples))
        else:
            walls_end = 0
            walls_max = 0
        walls_built = {"end_of_year": walls_end, "max": walls_max}

        # weapons_held: mean living agents holding a weapon per step.
        if self._weapons_held_samples:
            weapons_held = float(np.mean(self._weapons_held_samples))
        else:
            weapons_held = 0.0

        # predation_deaths / conflict_deaths: convenience aliases.
        predation_deaths = int(self.deaths.get("predator", 0))
        conflict_deaths  = int(self.deaths.get("conflict", 0))

        summary: Dict[str, Any] = {
            "population_mean":  pop_mean,
            "population_min":   pop_min,
            "population_max":   pop_max,
            "births":           self.births,
            "deaths_by_cause":  dict(self.deaths),
            "mean_displacement": mean_displacement,
            "wild_food_mean":    wild_food_mean,
            "water_occupancy":   water_occupancy,
            "pct_calories_farmed": pct_calories_farmed,
            "fertile_occupancy":   fertile_occupancy,
            # Phase 4 keys
            "structures_built":    structures_built,
            "stored_food_total":   stored_food_total,
            "mean_thermal":        mean_thermal,
            # Phase 5 keys
            "mean_genome":         mean_genome_list,
            "genome_drift":        genome_drift,
            "lineage_count":       lineage_count,
            # Phase 6 keys
            "n_predators":         n_predators,
            "walls_built":         walls_built,
            "weapons_held":        weapons_held,
            "predation_deaths":    predation_deaths,
            "conflict_deaths":     conflict_deaths,
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
        # Phase 2 accumulators
        self._water_on_steps: float = 0.0    # agent-steps spent on water_proximity >= 0.6
        self._water_total_steps: float = 0.0  # total living-agent-steps this year
        # Phase 3 accumulators
        self._foraged_total: float = 0.0      # total food units foraged from wild this year
        self._harvested_total: float = 0.0    # total food units harvested from crops this year
        self._fertile_on_steps: float = 0.0   # agent-steps on soil_fertility >= 0.6 tiles
        self._fertile_total_steps: float = 0.0  # total living-agent-steps this year
        # Phase 4 accumulators
        self._struct_total_samples: list[int] = []    # per-step total structure count
        self._struct_shelter_samples: list[int] = []  # per-step shelter count
        self._struct_storage_samples: list[int] = []  # per-step storage count
        self._stored_food_samples: list[float] = []   # per-step sum(state.stored_food)
        self._thermal_samples: list[float] = []       # per-step mean thermal of living agents
        # Phase 5 accumulators
        self._genome_samples: list = []               # per-step mean genome (G,) over living agents
        self._lineage_count_samples: list[int] = []   # per-step distinct lineage count
        # Phase 6 accumulators
        self._pred_count_samples: list[int] = []      # per-step living predator count
        self._wall_count_samples: list[int] = []      # per-step WALL tile count
        self._weapons_held_samples: list[int] = []    # per-step count of armed living agents
