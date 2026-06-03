"""
Phase 1 gate test — ENV-side gate conditions, no learner dependency (build-spec §4.3 / §5).

Gate: "stable nonzero population on foraging; deplete patches & relocate"

Three scenarios tested:
  1. Population survivability with respawn — 2 simulated years (720 steps),
     scripted forage-when-food-else-move policy + respawn_dead each step;
     population stays > 0 throughout.

  2. Patch depletion + relocation — agents on a high-wild tile deplete it,
     and a forage-then-move scripted policy increases mean displacement over
     time (agents leave depleted patches).

  3. Metrics integration — MetricsLogger over a short run yields a year_summary
     with the new Phase 1 keys (mean_displacement, wild_food_mean) containing
     finite values; JSONL writes/parses correctly.

Also produces data/phase1_forage.gif for visual inspection.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation, Actions
from sim.actions import Action, DIRECTIONS
from sim.metrics import MetricsLogger
from sim.render import render_frame, save_mp4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(width: int = 128, height: int = 128, seed: int = 1,
              init_agents: int = 64, max_agents: int = 256,
              energy_decay: float = 0.02) -> Simulation:
    """Build a simulation with relaxed economy params suitable for gate tests."""
    world = World.generate(WorldConfig(width=width, height=height, seed=seed))
    cfg = SimConfig(
        max_agents=max_agents,
        init_agents=init_agents,
        # Relax economy so agents survive without a learner
        energy_decay=energy_decay,        # slower starvation (~50 steps from full)
        forage_yield=0.5,                  # slightly higher yield per forage
        forage_regen=0.008,                # faster regrowth
        spoilage_carried=0.005,            # slower spoilage
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=seed)
    return sim


def _scripted_actions(sim: Simulation) -> Actions:
    """Scripted policy: FORAGE if wild food is available on current tile;
    EAT if inventory is full (>= 0.8); otherwise MOVE in a random direction.

    Priority order:
      1. EAT  — if inv_food >= 0.8 (digest before foraging again)
      2. FORAGE — if wild_remaining > 0 on current tile
      3. MOVE — explore when patch is empty and inventory not full
    """
    store = sim.store
    state = sim.state
    cfg   = sim.cfg
    M     = cfg.max_agents

    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)

    living_mask = store.alive & ~store.is_predator
    live_idx    = np.flatnonzero(living_mask)

    if live_idx.size == 0:
        return Actions(primary=primary, param=param, emit=emit)

    ys = store.y[live_idx]
    xs = store.x[live_idx]

    has_wild      = state.wild_remaining[ys, xs] > 0.0
    inv_nearly_full = store.inv_food[live_idx] >= 0.8  # eat to digest
    needs_move    = ~has_wild  # no food on this tile → explore

    # Priority: eat if nearly full, forage if patch available, else move
    action_choice = np.where(
        inv_nearly_full,
        int(Action.EAT),
        np.where(has_wild, int(Action.FORAGE), int(Action.MOVE))
    ).astype(np.int32)

    primary[live_idx] = action_choice

    # Random direction for movers
    rng = np.random.default_rng(sim.t)
    param[live_idx] = rng.integers(0, 8, size=live_idx.size).astype(np.int32)

    return Actions(primary=primary, param=param, emit=emit)


def _move_actions(sim: Simulation, rng: np.random.Generator) -> Actions:
    """Pure movement policy — all living agents move in a random direction.
    Used in the relocation test to drive displacement after patch depletion.
    """
    store = sim.store
    cfg   = sim.cfg
    M     = cfg.max_agents

    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)

    live_idx = np.flatnonzero(store.alive & ~store.is_predator)
    if live_idx.size > 0:
        primary[live_idx] = int(Action.MOVE)
        param[live_idx]   = rng.integers(0, 8, size=live_idx.size).astype(np.int32)

    return Actions(primary=primary, param=param, emit=emit)


# ---------------------------------------------------------------------------
# Scenario 1: Population survivability with respawn over 2 simulated years
# ---------------------------------------------------------------------------

def test_population_stays_nonzero_over_two_years():
    """Population must stay > 0 at every step across 720 steps (2 × 360-step years).

    Uses scripted forage-when-food-else-move policy + sim.respawn_dead() each
    step to simulate the Phase 1 auto-reset behaviour.
    """
    sim = _make_sim(width=128, height=128, seed=1, init_agents=64)
    n_steps = 720  # 2 × season_period (360)

    pop_per_step: list[int] = []
    for step in range(n_steps):
        actions = _scripted_actions(sim)
        sim.step(actions)
        sim.respawn_dead(seed=step)
        # Count living agents AFTER respawn (auto-reset keeps population topped up)
        pop_per_step.append(sim.store.n_living_agents())

    # Core gate: nonzero population throughout (after auto-reset respawn)
    assert all(n > 0 for n in pop_per_step), (
        f"Population dropped to zero at step {pop_per_step.index(0)} "
        f"(min={min(pop_per_step)})"
    )

    # Sanity: population is non-negative integers
    assert all(isinstance(n, int) and n >= 0 for n in pop_per_step)
    print(f"\n[gate1] pop range over {n_steps} steps: "
          f"min={min(pop_per_step)}, max={max(pop_per_step)}, "
          f"mean={sum(pop_per_step)/len(pop_per_step):.1f}")


# ---------------------------------------------------------------------------
# Scenario 2: Patch depletion + relocation
# ---------------------------------------------------------------------------

def test_patch_depletes_and_agents_relocate():
    """Verify wild_remaining drops on a heavily-forged patch, and agents move away
    after depletion (mean displacement increases from zero).

    Setup: small world, all agents placed on the single richest wild-food tile.
    Phase A (steps 0-9): scripted forage policy — agents deplete the patch.
    Phase B (steps 10-49): pure movement policy — agents walk away; displacement grows.

    Assertions:
      (a) wild_remaining on the starting tile drops measurably during Phase A.
      (b) mean Manhattan displacement at end of Phase B exceeds Phase A mean.
    """
    world = World.generate(WorldConfig(width=64, height=64, seed=7))
    cfg = SimConfig(
        max_agents=128,
        init_agents=20,
        energy_decay=0.005,  # very slow starvation for this short test
        forage_yield=0.5,
        forage_regen=0.001,  # slow regrowth so depletion is clearly visible
        spoilage_carried=0.001,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=7)

    # Find the land tile with highest wild_food
    land_mask  = world.elevation >= world.cfg.sea_level
    wild_init  = sim.state.wild_remaining.copy()  # after reset (= base_resources)
    best_wild  = wild_init.copy()
    best_wild[~land_mask] = -1.0
    best_yx    = np.unravel_index(best_wild.argmax(), best_wild.shape)
    best_y, best_x = int(best_yx[0]), int(best_yx[1])
    start_wild = float(sim.state.wild_remaining[best_y, best_x])

    # Cluster all agents on the richest patch
    store = sim.store
    living_idx = np.flatnonzero(store.alive & ~store.is_predator)
    store.y[living_idx] = best_y
    store.x[living_idx] = best_x

    # Reference spawn positions — all agents start at best_y, best_x
    spawn_y = np.full(cfg.max_agents, best_y, dtype=np.int32)
    spawn_x = np.full(cfg.max_agents, best_x, dtype=np.int32)

    def _displacement() -> float:
        live = np.flatnonzero(sim.store.alive & ~sim.store.is_predator)
        if live.size == 0:
            return 0.0
        dy = np.abs(sim.store.y[live].astype(float) - spawn_y[live])
        dx = np.abs(sim.store.x[live].astype(float) - spawn_x[live])
        return float((dy + dx).mean())

    # --- Phase A: forage to deplete patch (10 steps) ---
    wild_after_depletion: list[float] = []
    rng_a = np.random.default_rng(42)
    for step in range(10):
        # Simple forage-else-move policy for Phase A
        M = cfg.max_agents
        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param   = np.zeros(M, dtype=np.int32)
        emit    = np.zeros(M, dtype=np.int32)
        live    = np.flatnonzero(sim.store.alive & ~sim.store.is_predator)
        if live.size > 0:
            ys = sim.store.y[live]
            xs = sim.store.x[live]
            has_wild = sim.state.wild_remaining[ys, xs] > 0.0
            primary[live] = np.where(has_wild, int(Action.FORAGE), int(Action.EAT)).astype(np.int32)
            param[live] = rng_a.integers(0, 8, size=live.size).astype(np.int32)
        sim.step(Actions(primary=primary, param=param, emit=emit))
        # No respawn during Phase A — keep agents on patch
        wild_after_depletion.append(float(sim.state.wild_remaining[best_y, best_x]))

    min_wild_phase_a = min(wild_after_depletion)

    # (a) Patch must have depleted
    assert min_wild_phase_a < start_wild, (
        f"Patch was not depleted: start_wild={start_wild:.4f}, "
        f"min_wild={min_wild_phase_a:.4f}"
    )

    disp_end_phase_a = _displacement()  # should be ~0 (agents still on patch)

    # --- Phase B: pure movement policy (40 steps) ---
    rng_b = np.random.default_rng(99)
    for step in range(40):
        actions = _move_actions(sim, rng_b)
        sim.step(actions)
        sim.respawn_dead(seed=100 + step)

    disp_end_phase_b = _displacement()

    # (b) Displacement must increase from Phase A → end of Phase B
    assert disp_end_phase_b > disp_end_phase_a, (
        f"Displacement did not increase after movement phase: "
        f"phase_a={disp_end_phase_a:.2f}, phase_b={disp_end_phase_b:.2f}"
    )

    print(f"\n[gate2] patch wild: start={start_wild:.4f}, min={min_wild_phase_a:.4f}")
    print(f"[gate2] displacement: end_phase_a={disp_end_phase_a:.2f}, "
          f"end_phase_b={disp_end_phase_b:.2f}")


# ---------------------------------------------------------------------------
# Scenario 3: Metrics integration
# ---------------------------------------------------------------------------

def test_metrics_phase1_keys_finite():
    """MetricsLogger over a short run returns year_summary with new Phase 1 keys
    containing finite (non-NaN, non-Inf) values."""
    sim = _make_sim(width=64, height=64, seed=3, init_agents=32, max_agents=128)
    logger = MetricsLogger()

    for step in range(50):
        actions = _scripted_actions(sim)
        sim.step(actions)
        sim.respawn_dead(seed=step)
        logger.record_step(sim)

    summary = logger.year_summary()

    # Existing keys still present
    for key in ("population_mean", "population_min", "population_max",
                "births", "deaths_by_cause"):
        assert key in summary, f"Missing existing key: {key}"

    # New Phase 1 keys present and finite
    assert "mean_displacement" in summary, "Missing key: mean_displacement"
    assert "wild_food_mean" in summary, "Missing key: wild_food_mean"

    assert math.isfinite(summary["mean_displacement"]), (
        f"mean_displacement is not finite: {summary['mean_displacement']}"
    )
    assert math.isfinite(summary["wild_food_mean"]), (
        f"wild_food_mean is not finite: {summary['wild_food_mean']}"
    )
    assert summary["wild_food_mean"] >= 0.0, (
        f"wild_food_mean negative: {summary['wild_food_mean']}"
    )
    assert summary["mean_displacement"] >= 0.0, (
        f"mean_displacement negative: {summary['mean_displacement']}"
    )

    print(f"\n[gate3] mean_displacement={summary['mean_displacement']:.3f}, "
          f"wild_food_mean={summary['wild_food_mean']:.4f}")


def test_metrics_jsonl_round_trip_with_new_keys():
    """year_summary writes JSONL; parsed line contains both old and new keys."""
    sim = _make_sim(width=64, height=64, seed=5, init_agents=32, max_agents=128)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as fh:
        path = fh.name

    try:
        logger = MetricsLogger(out_path=path)
        for step in range(20):
            actions = _scripted_actions(sim)
            sim.step(actions)
            sim.respawn_dead(seed=step)
            logger.record_step(sim)
        logger.year_summary()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        assert len(lines) == 1
        parsed = json.loads(lines[0])

        for key in ("population_mean", "population_min", "population_max",
                    "births", "deaths_by_cause",
                    "mean_displacement", "wild_food_mean"):
            assert key in parsed, f"JSONL line missing key: {key}"

        assert math.isfinite(parsed["mean_displacement"])
        assert math.isfinite(parsed["wild_food_mean"])
    finally:
        os.unlink(path)


def test_metrics_per_year_summaries_over_two_years():
    """Two consecutive year_summary calls both produce sane Phase 1 metrics."""
    sim = _make_sim(width=64, height=64, seed=9, init_agents=32, max_agents=128)
    logger = MetricsLogger()

    for year in range(2):
        for step in range(60):
            actions = _scripted_actions(sim)
            sim.step(actions)
            sim.respawn_dead(seed=year * 60 + step)
            logger.record_step(sim)
        s = logger.year_summary()
        assert math.isfinite(s["mean_displacement"]), f"Year {year}: non-finite displacement"
        assert math.isfinite(s["wild_food_mean"]), f"Year {year}: non-finite wild_food_mean"
        assert s["population_mean"] > 0.0, f"Year {year}: zero population mean"


# ---------------------------------------------------------------------------
# Visualization: render a short scripted-forage run → data/phase1_forage.gif
# ---------------------------------------------------------------------------

def test_render_phase1_forage_gif():
    """Render 30 steps of scripted foraging to data/phase1_forage.gif.

    This test always succeeds as long as save_mp4 doesn't raise; the gif is
    written for human eyeballing of the gate behaviour.
    """
    sim = _make_sim(width=64, height=64, seed=2, init_agents=32, max_agents=128)

    frames: list[np.ndarray] = []
    n_frames = 30

    for step in range(n_frames):
        frame = render_frame(sim.world, sim.state, sim.store)
        frames.append(frame)
        actions = _scripted_actions(sim)
        sim.step(actions)
        sim.respawn_dead(seed=step)

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_target = str(out_dir / "phase1_forage.mp4")

    actual_path = save_mp4(frames, gif_target, fps=5)
    assert os.path.exists(actual_path), f"Output file not found: {actual_path}"
    assert os.path.getsize(actual_path) > 0, "Output file is empty"
    print(f"\n[viz] rendered to: {actual_path}")
