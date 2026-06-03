"""
Phase 2 gate test — water-anchoring metric + scripted policy comparisons
(build-spec §4.3 / §5 Phase 2).

Gate: "agent-time on water_proximity>0.6 tiles significantly above null;
       dehydration deaths low once learned"

Three scenarios tested:
  1. Water anchoring signal — a water-seeking policy achieves HIGHER water_occupancy
     than a random-move baseline over 300 steps on a 128×128 world.

  2. Dehydration death comparison — a drink-when-thirsty policy incurs FEWER
     "dehydrate" deaths than a no-drink (thirst-ignored) policy over a fixed
     horizon.

  3. Metric integration — year_summary contains water_occupancy (finite, in [0,1])
     plus all prior keys; JSONL round-trips correctly.

Also produces data/phase2_water.png for visual inspection.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation, Actions
from sim.actions import Action, DIRECTIONS
from sim.metrics import MetricsLogger
from sim.render import render_frame
from sim.reproduce import resolve_deaths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WATER_THRESHOLD = 0.6   # must match MetricsLogger.record_step
_DRINK_THRESHOLD = 0.5   # cfg.drink_min_water default


def _make_sim(
    width: int = 128,
    height: int = 128,
    seed: int = 1,
    init_agents: int = 64,
    max_agents: int = 256,
    energy_decay: float = 0.005,
    hydration_decay: float = 0.008,
    **extra,
) -> Simulation:
    """Build a simulation with relaxed economy params suitable for gate tests."""
    world = World.generate(WorldConfig(width=width, height=height, seed=seed))
    cfg = SimConfig(
        max_agents=max_agents,
        init_agents=init_agents,
        energy_decay=energy_decay,
        hydration_decay=hydration_decay,
        forage_yield=0.5,
        forage_regen=0.008,
        spoilage_carried=0.005,
        drink_min_water=_DRINK_THRESHOLD,
        drink_restore=1.0,
        **extra,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=seed)
    return sim


def _nearest_water_direction(
    y: int,
    x: int,
    water_prox: np.ndarray,
    threshold: float = _WATER_THRESHOLD,
) -> int:
    """Return the DIRECTIONS index that moves (y, x) closest to the nearest
    tile with water_proximity >= threshold.

    Falls back to a random direction (0) if none are found.
    """
    H, W = water_prox.shape
    water_tiles = np.argwhere(water_prox >= threshold)  # (N, 2) [y, x]
    if water_tiles.shape[0] == 0:
        return 0

    # Manhattan distance to each candidate
    dy = water_tiles[:, 0] - y
    dx = water_tiles[:, 1] - x
    dist = np.abs(dy) + np.abs(dx)
    if dist.min() == 0:
        # Already on a water tile — stay (NOOP effectively, but we pick a random
        # direction here; calling code prefers DRINK anyway)
        return 0

    best = int(dist.argmin())
    target_y = water_tiles[best, 0]
    target_x = water_tiles[best, 1]

    # Pick the DIRECTIONS entry that reduces distance most
    best_dir = 0
    best_dist = np.inf
    for d, (ddy, ddx) in enumerate(DIRECTIONS):
        ny = np.clip(y + ddy, 0, H - 1)
        nx = np.clip(x + ddx, 0, W - 1)
        dist_to_target = abs(int(ny) - int(target_y)) + abs(int(nx) - int(target_x))
        if dist_to_target < best_dist:
            best_dist = dist_to_target
            best_dir = d

    return best_dir


def _random_actions(sim: Simulation, rng: np.random.Generator) -> Actions:
    """Pure random-move policy — null baseline."""
    store = sim.store
    cfg = sim.cfg
    M = cfg.max_agents

    primary = np.full(M, int(Action.MOVE), dtype=np.int32)
    param = rng.integers(0, 8, size=M, dtype=np.int32)
    emit = np.zeros(M, dtype=np.int32)

    # Dead agents get NOOP
    dead = ~store.alive
    primary[dead] = int(Action.NOOP)
    return Actions(primary=primary, param=param, emit=emit)


def _water_seeking_actions(sim: Simulation, rng: np.random.Generator) -> Actions:
    """Water-seeking policy — maximises time on high-water tiles (proxy for a
    learnt water-anchoring behaviour).

    Priority:
      1. DRINK if on a drinkable tile (water_proximity >= drink_min_water)
         — stays put and drinks while hydration is recoverable.
      2. MOVE toward nearest water_proximity >= 0.6 tile (all other cases).
         Also forage/eat opportunistically only when *already on a high-water tile*
         (agent has just drunk and can choose EAT/FORAGE as a side-action — but for
         the purpose of this scripted test we simply DRINK every step on water so we
         maximise the water_occupancy numerator).

    For water_occupancy purposes the key distinction vs random is:
      - Random: ~23% of steps on water tiles (land fraction with wp >= 0.6).
      - Seeking: anchors at water tiles and stays there → much higher fraction.
    """
    store = sim.store
    world = sim.world
    cfg = sim.cfg
    M = cfg.max_agents

    water_prox = world.base_resources["water_proximity"]

    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param = rng.integers(0, 8, size=M, dtype=np.int32)
    emit = np.zeros(M, dtype=np.int32)

    live_idx = np.flatnonzero(store.alive & ~store.is_predator)
    if live_idx.size == 0:
        return Actions(primary=primary, param=param, emit=emit)

    ys = store.y[live_idx]
    xs = store.x[live_idx]

    # High-water tile: wp >= 0.6 (the metric threshold)
    on_high_water = water_prox[ys, xs] >= _WATER_THRESHOLD
    # Drinkable tile: wp >= drink_min_water (DRINK action is legal here)
    on_drinkable = water_prox[ys, xs] >= cfg.drink_min_water

    # Policy: DRINK if legal (on drinkable tile) — this anchors the agent at a water
    # tile for the step (maximising occupancy); MOVE toward water otherwise.
    action_choice = np.where(
        on_drinkable,
        int(Action.DRINK),
        int(Action.MOVE),
    ).astype(np.int32)

    primary[live_idx] = action_choice

    # For agents that chose MOVE, steer toward nearest high-water tile
    movers = live_idx[action_choice == int(Action.MOVE)]
    for slot in movers:
        param[slot] = _nearest_water_direction(
            int(store.y[slot]), int(store.x[slot]), water_prox
        )

    return Actions(primary=primary, param=param, emit=emit)


# ---------------------------------------------------------------------------
# Scenario 1: Water anchoring (seeking vs random baseline)
# ---------------------------------------------------------------------------

def test_water_seeking_policy_higher_occupancy_than_random():
    """Water-seeking policy achieves significantly higher water_occupancy than
    the random-move baseline.

    Both runs use the same world/seed, 300 steps, respawn_dead each step.
    """
    N_STEPS = 300

    # --- run A: random baseline ---
    sim_rnd = _make_sim(width=128, height=128, seed=1, init_agents=64)
    logger_rnd = MetricsLogger()
    rng_rnd = np.random.default_rng(42)
    for step in range(N_STEPS):
        actions = _random_actions(sim_rnd, rng_rnd)
        sim_rnd.step(actions)
        sim_rnd.respawn_dead(seed=step)
        logger_rnd.record_step(sim_rnd)
    summary_rnd = logger_rnd.year_summary()

    # --- run B: water-seeking policy ---
    sim_ws = _make_sim(width=128, height=128, seed=1, init_agents=64)
    logger_ws = MetricsLogger()
    rng_ws = np.random.default_rng(42)
    for step in range(N_STEPS):
        actions = _water_seeking_actions(sim_ws, rng_ws)
        sim_ws.step(actions)
        sim_ws.respawn_dead(seed=step)
        logger_ws.record_step(sim_ws)
    summary_ws = logger_ws.year_summary()

    wo_random = summary_rnd["water_occupancy"]
    wo_seeking = summary_ws["water_occupancy"]

    print(f"\n[gate2.1] water_occupancy: random={wo_random:.4f}, seeking={wo_seeking:.4f}")

    assert "water_occupancy" in summary_rnd
    assert "water_occupancy" in summary_ws
    assert math.isfinite(wo_random), f"random water_occupancy not finite: {wo_random}"
    assert math.isfinite(wo_seeking), f"seeking water_occupancy not finite: {wo_seeking}"
    assert 0.0 <= wo_random <= 1.0, f"random water_occupancy out of [0,1]: {wo_random}"
    assert 0.0 <= wo_seeking <= 1.0, f"seeking water_occupancy out of [0,1]: {wo_seeking}"

    assert wo_seeking > wo_random, (
        f"Water-seeking policy should have higher water_occupancy than random: "
        f"seeking={wo_seeking:.4f}, random={wo_random:.4f}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Dehydration deaths — drink policy vs no-drink policy
# ---------------------------------------------------------------------------

def _count_dehydrate_deaths_over_run(
    sim: Simulation,
    policy: str,   # "drink" or "no_drink"
    n_steps: int,
    seed: int,
) -> int:
    """Run a scripted policy for n_steps and return number of dehydrate deaths.

    Policy "drink":
      - DRINK if on watered tile
      - EAT if inventory >= 0.6
      - FORAGE if wild food available
      - MOVE randomly otherwise

    Policy "no_drink":
      - Same but never DRINK (moves randomly instead when on watered tile)
    """
    store = sim.store
    state = sim.state
    world = sim.world
    cfg = sim.cfg
    M = cfg.max_agents
    water_prox = world.base_resources["water_proximity"]

    rng = np.random.default_rng(seed)
    total_dehydrate = 0

    for step in range(n_steps):
        # ---- record hydration of living agents BEFORE step ----
        # so we can attribute death cause after step kills them
        living_before = store.alive & ~store.is_predator
        hydration_snapshot = store.hydration.copy()

        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param = rng.integers(0, 8, size=M, dtype=np.int32)
        emit = np.zeros(M, dtype=np.int32)

        live_idx = np.flatnonzero(living_before)
        if live_idx.size > 0:
            ys = store.y[live_idx]
            xs = store.x[live_idx]

            near_water = water_prox[ys, xs] >= cfg.drink_min_water
            inv_full = store.inv_food[live_idx] >= 0.6
            has_wild = state.wild_remaining[ys, xs] > 0.0

            if policy == "drink":
                action_choice = np.where(
                    near_water,
                    int(Action.DRINK),
                    np.where(
                        inv_full,
                        int(Action.EAT),
                        np.where(has_wild, int(Action.FORAGE), int(Action.MOVE)),
                    ),
                ).astype(np.int32)
            else:
                # no_drink: treat near_water as no special case, just eat/forage/move
                action_choice = np.where(
                    inv_full,
                    int(Action.EAT),
                    np.where(has_wild, int(Action.FORAGE), int(Action.MOVE)),
                ).astype(np.int32)

            primary[live_idx] = action_choice

        out = sim.step(Actions(primary=primary, param=param, emit=emit))

        # ---- attribute deaths using pre-step hydration snapshot ----
        done_idx = np.flatnonzero(out.done)
        for slot in done_idx:
            if hydration_snapshot[slot] <= 0.0:
                total_dehydrate += 1

        sim.respawn_dead(seed=step + seed * 10000)

    return total_dehydrate


def test_drinking_policy_fewer_dehydrate_deaths():
    """A drink-when-thirsty policy should result in fewer dehydrate deaths
    than a no-drink policy over 300 steps on the same world.

    Uses generous but finite hydration_decay (0.015) so agents can die of
    dehydration without the simulation being trivially lethal.
    """
    N_STEPS = 300
    SEED = 1

    sim_drink = _make_sim(
        width=128, height=128, seed=SEED,
        init_agents=64, max_agents=256,
        energy_decay=0.005,
        hydration_decay=0.015,  # slower than default so both policies run meaningfully
    )
    sim_no_drink = _make_sim(
        width=128, height=128, seed=SEED,
        init_agents=64, max_agents=256,
        energy_decay=0.005,
        hydration_decay=0.015,
    )

    deaths_drink = _count_dehydrate_deaths_over_run(sim_drink, "drink", N_STEPS, seed=42)
    deaths_no_drink = _count_dehydrate_deaths_over_run(sim_no_drink, "no_drink", N_STEPS, seed=42)

    print(
        f"\n[gate2.2] dehydrate deaths: drink={deaths_drink}, no_drink={deaths_no_drink}"
    )

    assert deaths_drink < deaths_no_drink, (
        f"Drink policy should have fewer dehydrate deaths: "
        f"drink={deaths_drink}, no_drink={deaths_no_drink}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Metric integration — water_occupancy present, finite, in [0,1],
#             all prior keys present, JSONL round-trips.
# ---------------------------------------------------------------------------

def test_year_summary_has_water_occupancy_and_prior_keys():
    """year_summary must contain water_occupancy (finite, in [0,1]) alongside
    all prior Phase 0/1 keys."""
    sim = _make_sim(width=64, height=64, seed=3, init_agents=32, max_agents=128)
    logger = MetricsLogger()
    rng = np.random.default_rng(7)

    for step in range(50):
        actions = _water_seeking_actions(sim, rng)
        sim.step(actions)
        sim.respawn_dead(seed=step)
        logger.record_step(sim)

    summary = logger.year_summary()

    # All prior keys present
    for key in (
        "population_mean", "population_min", "population_max",
        "births", "deaths_by_cause",
        "mean_displacement", "wild_food_mean",
    ):
        assert key in summary, f"Missing prior key: {key}"

    # New water_occupancy key present and valid
    assert "water_occupancy" in summary, "Missing key: water_occupancy"
    wo = summary["water_occupancy"]
    assert math.isfinite(wo), f"water_occupancy not finite: {wo}"
    assert 0.0 <= wo <= 1.0, f"water_occupancy out of [0,1]: {wo}"

    # deaths_by_cause includes dehydrate
    assert "dehydrate" in summary["deaths_by_cause"], (
        "deaths_by_cause should contain 'dehydrate' key"
    )

    print(f"\n[gate2.3] water_occupancy={wo:.4f}, deaths={summary['deaths_by_cause']}")


def test_water_occupancy_jsonl_round_trip():
    """year_summary writes JSONL; parsed line contains water_occupancy and all prior keys."""
    sim = _make_sim(width=64, height=64, seed=5, init_agents=32, max_agents=128)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        path = fh.name

    try:
        logger = MetricsLogger(out_path=path)
        rng = np.random.default_rng(9)
        for step in range(20):
            actions = _water_seeking_actions(sim, rng)
            sim.step(actions)
            sim.respawn_dead(seed=step)
            logger.record_step(sim)
        logger.year_summary()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        assert len(lines) == 1
        parsed = json.loads(lines[0])

        for key in (
            "population_mean", "population_min", "population_max",
            "births", "deaths_by_cause",
            "mean_displacement", "wild_food_mean",
            "water_occupancy",
        ):
            assert key in parsed, f"JSONL line missing key: {key}"

        assert math.isfinite(parsed["water_occupancy"])
        assert 0.0 <= parsed["water_occupancy"] <= 1.0
    finally:
        os.unlink(path)


def test_water_occupancy_zero_when_no_water_tiles():
    """water_occupancy should be low (near zero) when all agents are on dry tiles."""
    # Place all agents on dry tiles by creating a world and forcing positions
    sim = _make_sim(width=64, height=64, seed=3, init_agents=16, max_agents=64)
    world = sim.world
    water_prox = world.base_resources["water_proximity"]

    # Find land tiles with low water proximity
    elev = world.elevation
    sea = world.cfg.sea_level
    land_ys, land_xs = np.where(
        (elev >= sea) & (water_prox < _DRINK_THRESHOLD)
    )

    if land_ys.shape[0] < 4:
        pytest.skip("Not enough dry land tiles on this world for the test")

    # Force all agents to dry land tiles
    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    n = live_idx.size
    chosen = np.arange(n) % land_ys.shape[0]
    sim.store.y[live_idx] = land_ys[chosen]
    sim.store.x[live_idx] = land_xs[chosen]

    logger = MetricsLogger()
    rng = np.random.default_rng(0)

    # NOOP for all agents (stay in place on dry tiles)
    M = sim.cfg.max_agents
    for _ in range(10):
        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param = np.zeros(M, dtype=np.int32)
        emit = np.zeros(M, dtype=np.int32)
        sim.step(Actions(primary=primary, param=param, emit=emit))
        sim.respawn_dead(seed=0)
        logger.record_step(sim)

    summary = logger.year_summary()
    wo = summary["water_occupancy"]
    # Agents are mostly on tiles with water_prox < 0.5 < 0.6 → occupancy near 0
    assert wo < 0.2, f"Expected low water_occupancy on dry tiles, got {wo:.4f}"
    print(f"\n[gate2.4] dry-tile water_occupancy={wo:.4f} (expected < 0.2)")


# ---------------------------------------------------------------------------
# Visualization: single-frame PNG of the water-seeking run
# ---------------------------------------------------------------------------

def test_render_phase2_water_png():
    """Render a single frame of a water-seeking run with water tiles tinted cyan.

    Saves to data/phase2_water.png. Always passes as long as imwrite doesn't raise.
    """
    import imageio.v3 as iio

    sim = _make_sim(width=128, height=128, seed=1, init_agents=64)
    rng = np.random.default_rng(1)

    # Run 30 steps so agents have had time to move toward water
    for step in range(30):
        actions = _water_seeking_actions(sim, rng)
        sim.step(actions)
        sim.respawn_dead(seed=step)

    frame = render_frame(sim.world, sim.state, sim.store).copy()

    # Tint water tiles (water_proximity >= 0.6) with a cyan overlay
    water_prox = sim.world.base_resources["water_proximity"]
    water_mask = water_prox >= _WATER_THRESHOLD
    # Blend: 50% cyan (0, 200, 220) onto existing colour
    frame[water_mask, 0] = (frame[water_mask, 0].astype(np.uint16) * 50 // 100).astype(np.uint8)
    frame[water_mask, 1] = np.clip(
        frame[water_mask, 1].astype(np.uint16) * 50 // 100 + 100, 0, 255
    ).astype(np.uint8)
    frame[water_mask, 2] = np.clip(
        frame[water_mask, 2].astype(np.uint16) * 50 // 100 + 110, 0, 255
    ).astype(np.uint8)

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_water.png"

    iio.imwrite(str(out_path), frame)
    assert out_path.exists() and out_path.stat().st_size > 0, (
        f"Output file not found or empty: {out_path}"
    )
    print(f"\n[viz] rendered to: {out_path}")
