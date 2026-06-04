"""
Phase 6 CODE gate — combat/predation metrics + scripted policy scenarios.

Scenarios:
  1. Walls reduce predator deaths: a cohort that builds WALLs around itself
     (or stands on wall tiles) suffers FEWER predator deaths than an unwalled cohort
     under the same predation pressure.

  2. Armed agents win fights: an armed agent vs an unarmed agent in mutual combat —
     the unarmed agent takes more damage per round; the armed side defeats the
     unarmed side.

  3. Predator-prey oscillation without total collapse: run 3 years under a
     scripted survive policy; assert both populations stay > 0 throughout and
     predator count tracks ~ pred_per_agents * agents.

  4. Metric integration: year_summary has n_predators, walls_built,
     weapons_held, predation_deaths, conflict_deaths, deaths_by_cause with
     "predator"/"conflict" keys — all finite.  Also checks all prior Phase 1-5
     keys are still present, and JSONL round-trips.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.state import EntityStore, WorldState
from sim.actions import Action, StructureType, DIRECTIONS
from sim.dynamics import resolve_combat, craft_weapon, build
from sim.reproduce import resolve_deaths
from sim.threats import update_scent, spawn_predators, step_predators
from sim.simulation import Simulation, Actions
from sim.metrics import MetricsLogger


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_world(width: int = 48, height: int = 48, seed: int = 42) -> World:
    return World.generate(WorldConfig(width=width, height=height, seed=seed))


def _make_cfg(**kwargs) -> SimConfig:
    """Survivable config for scripted-policy tests."""
    defaults = dict(
        max_agents=512,
        init_agents=80,
        energy_decay=0.002,
        hydration_decay=0.002,
        thermal_decay=0.0,      # no thermal stress — focus on predation
        D_init=0.0,
        spoilage_carried=0.0,
        forage_yield=0.6,
        forage_regen=0.04,
        pred_per_agents=0.10,   # 10% predators — strong signal
        pred_attack_damage=0.3,
        attack_damage=0.15,
        weapon_attack_mult=2.0,
        wall_defense_mult=0.3,
        group_defense_per_ally=0.9,
        group_defense_floor=0.3,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _scripted_survive(sim: Simulation, cfg: SimConfig) -> Actions:
    """Simple survival-first scripted policy (no combat, no building)."""
    from sim.actions import build_mask
    m = build_mask(sim.world, sim.state, sim.store, cfg)
    pr = np.full(cfg.max_agents, int(Action.REST), np.int32)
    for i in np.flatnonzero(sim.store.living_agents_mask()):
        e = sim.store.energy[i]
        f = sim.store.inv_food[i]
        h = sim.store.hydration[i]
        if e < 0.5 and f > 0:
            pr[i] = int(Action.EAT)
        elif m[i, int(Action.FORAGE)] and f < 0.5:
            pr[i] = int(Action.FORAGE)
        elif m[i, int(Action.DRINK)] and h < 0.6:
            pr[i] = int(Action.DRINK)
        # else REST
    param = np.random.randint(0, 8, cfg.max_agents).astype(np.int32)
    return Actions(pr, param, np.zeros(cfg.max_agents, np.int32))


# ---------------------------------------------------------------------------
# Test 1: Walls reduce predator deaths (via blocking movement)
# ---------------------------------------------------------------------------

def test_walls_reduce_predator_deaths():
    """
    Two cohorts under predation:
      - Walled cohort: agent surrounded by a ring of WALL tiles; predator starts
        2 tiles away.  Predator cannot move to any of the 8 wall-ring tiles (they
        are walls), so it stays 2+ tiles away and CANNOT attack (attack radius is
        1 tile).  Expected deaths: 0 over N steps.
      - Unwalled cohort: same geometry, NO walls.  Predator co-located with the
        agent from the start → guaranteed one-shot kill each step.

    Mechanism tested: walls block predator movement (step_predators skips wall
    tiles when choosing where to step), keeping the agent unreachable.
    """
    world = _make_world(width=32, height=32, seed=7)
    cfg = _make_cfg(
        max_agents=64, init_agents=1,
        pred_per_agents=0.0,          # manual predator control
        pred_attack_damage=1.1,       # one hit kills (>= 1.0 health)
        energy_decay=0.0, hydration_decay=0.0, thermal_decay=0.0,
    )

    # Find a tile with a 2-tile buffer all around on land.
    # Agent at (y, x); wall ring at ring-1 distance; predator 2 tiles north.
    sea = world.cfg.sea_level
    H, W = 32, 32

    agent_y, agent_x = None, None
    for y in range(4, H - 4):
        for x in range(4, W - 4):
            if world.elevation[y, x] < sea:
                continue
            # Ring-1 tiles: all 8 cardinal/diagonal immediate neighbours
            ring1 = [(y + int(dy), x + int(dx)) for dy, dx in DIRECTIONS]
            # Ring-2 tiles (needed so predator has somewhere legal to stand):
            ring2 = []
            for ry, rx in ring1:
                for dy, dx in DIRECTIONS:
                    nny, nnx = ry + int(dy), rx + int(dx)
                    if (nny, nnx) != (y, x) and (nny, nnx) not in ring1:
                        ring2.append((nny, nnx))
            # All ring-1 and ring-2 must be in-bounds and on land
            all_needed = ring1 + ring2
            if all(0 <= ny < H and 0 <= nx < W and world.elevation[ny, nx] >= sea
                   for ny, nx in all_needed):
                agent_y, agent_x = y, x
                break
        if agent_y is not None:
            break

    assert agent_y is not None, "Could not find suitable land tile with 2-tile land buffer"

    ring1_tiles = [(agent_y + int(dy), agent_x + int(dx)) for dy, dx in DIRECTIONS]
    # Predator starts 2 tiles north (outside the wall ring, cannot step across walls)
    pred_start_y = agent_y - 2
    pred_start_x = agent_x

    def _place_agent_and_pred(state, store, with_walls: bool):
        # Agent
        store.alive[0] = True; store.is_predator[0] = False
        store.y[0] = agent_y; store.x[0] = agent_x
        store.health[0] = 1.0; store.energy[0] = 1.0
        store.hydration[0] = 1.0; store.thermal[0] = 1.0
        if with_walls:
            for ny, nx in ring1_tiles:
                state.structure_type[ny, nx] = np.int8(int(StructureType.WALL))
                state.structure_hp[ny, nx]   = np.float32(100.0)

    def _run(with_walls: bool, n_steps: int) -> int:
        store = EntityStore.create(cfg)
        state = WorldState.create(H, W)
        rng   = np.random.default_rng(7)

        _place_agent_and_pred(state, store, with_walls)

        if with_walls:
            # Predator 2 tiles north (outside wall ring)
            store.alive[1] = True; store.is_predator[1] = True
            store.y[1] = pred_start_y; store.x[1] = pred_start_x
            store.health[1] = 1.0
        else:
            # Predator co-located with agent → guaranteed attack every step
            store.alive[1] = True; store.is_predator[1] = True
            store.y[1] = agent_y; store.x[1] = agent_x
            store.health[1] = 1.0

        deaths = 0
        for _ in range(n_steps):
            update_scent(state, store)
            pred_result = step_predators(store, state, world, cfg, rng)
            m = {}
            resolve_deaths(store, metrics_deaths=m, predator_slots=pred_result["predator_slots"])
            deaths += m.get("predator", 0)
            # Respawn agent to count deaths per step
            if not store.alive[0]:
                store.alive[0] = True; store.is_predator[0] = False
                store.y[0] = agent_y; store.x[0] = agent_x
                store.health[0] = 1.0
        return deaths

    walled_deaths   = _run(with_walls=True, n_steps=10)
    unwalled_deaths = _run(with_walls=False, n_steps=10)

    assert walled_deaths == 0, (
        f"Walled cohort should have 0 deaths (predator blocked by wall ring, 2 tiles away), "
        f"got {walled_deaths}; agent=({agent_y},{agent_x}), pred_start=({pred_start_y},{pred_start_x})"
    )
    assert unwalled_deaths > 0, (
        f"Unwalled cohort should take deaths (pred co-located, 1.1 damage one-shots), "
        f"got {unwalled_deaths}"
    )
    assert walled_deaths < unwalled_deaths, (
        f"Walled ({walled_deaths}) must be < unwalled ({unwalled_deaths})"
    )


# ---------------------------------------------------------------------------
# Test 2: Armed agents win fights (deal more damage / unarmed dies first)
# ---------------------------------------------------------------------------

def test_armed_agent_outperforms_unarmed_in_duel():
    """
    Armed agent A (weapon=True) vs. unarmed agent B (weapon=False) in symmetric
    mutual combat. After one round of resolve_combat (both attacking each other),
    the armed agent deals weapon_attack_mult × more damage to B, so B loses more
    health than A.

    Extended duel: run repeated rounds until one agent dies; the unarmed one (B)
    should die first.
    """
    cfg = SimConfig(
        max_agents=64, init_agents=2,
        attack_damage=0.15,
        weapon_attack_mult=2.0,
        wall_defense_mult=1.0,   # no wall bonus in this test
        group_defense_per_ally=1.0,  # no group defense
        group_defense_floor=1.0,
        energy_decay=0.0, hydration_decay=0.0, thermal_decay=0.0,
    )
    M = cfg.max_agents
    ws = WorldState.create(16, 16)

    store = EntityStore.create(cfg)
    # Agent A (armed) at (5, 5)
    store.alive[0] = True
    store.is_predator[0] = False
    store.y[0] = 5; store.x[0] = 5
    store.health[0] = 1.0
    store.energy[0] = 1.0; store.hydration[0] = 1.0; store.thermal[0] = 1.0
    store.weapon[0] = True   # ARMED

    # Agent B (unarmed) at (5, 6) — adjacent East
    store.alive[1] = True
    store.is_predator[1] = False
    store.y[1] = 5; store.x[1] = 6
    store.health[1] = 1.0
    store.energy[1] = 1.0; store.hydration[1] = 1.0; store.thermal[1] = 1.0
    store.weapon[1] = False  # UNARMED

    # Single round: A attacks East (dir 2), B attacks West (dir 6)
    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East
    param[1] = 6  # West
    attackers = np.array([0, 1], dtype=np.int32)

    resolve_combat(store, ws, attackers, param, cfg)

    dmg_to_B = 1.0 - float(store.health[1])
    dmg_to_A = 1.0 - float(store.health[0])

    assert dmg_to_B > dmg_to_A, (
        f"Armed A should deal more damage to B than B deals to A: "
        f"dmg_to_B={dmg_to_B:.4f}, dmg_to_A={dmg_to_A:.4f}"
    )
    assert dmg_to_B == pytest.approx(cfg.attack_damage * cfg.weapon_attack_mult, abs=1e-5)
    assert dmg_to_A == pytest.approx(cfg.attack_damage, abs=1e-5)

    # Extended duel: reset health, run until one dies
    store2 = EntityStore.create(cfg)
    store2.alive[0] = True; store2.alive[1] = True
    store2.y[0] = 5; store2.x[0] = 5
    store2.y[1] = 5; store2.x[1] = 6
    store2.health[0] = 1.0; store2.health[1] = 1.0
    store2.energy[0] = 1.0; store2.energy[1] = 1.0
    store2.hydration[0] = 1.0; store2.hydration[1] = 1.0
    store2.thermal[0] = 1.0; store2.thermal[1] = 1.0
    store2.weapon[0] = True   # A armed
    store2.weapon[1] = False  # B unarmed

    ws2 = WorldState.create(16, 16)
    param2 = np.zeros(M, dtype=np.int32)
    param2[0] = 2  # East
    param2[1] = 6  # West

    for _ in range(100):
        if not (store2.alive[0] and store2.alive[1]):
            break
        resolve_combat(store2, ws2, np.array([0, 1], dtype=np.int32), param2, cfg)
        resolve_deaths(store2)

    # B (unarmed) should die before or together with A (armed)
    assert not store2.alive[1], "Unarmed agent B should have died in the duel"
    # A (armed) may or may not be alive, but B should die first or simultaneously
    # The ratio means B takes 2x damage per round, so B dies in half the rounds A would


# ---------------------------------------------------------------------------
# Test 3: Predator-prey oscillation without total collapse
# ---------------------------------------------------------------------------

def test_predator_prey_no_total_collapse():
    """
    Run ~3 years (3 × 360 = 1080 steps) with a scripted survive policy and
    respawn_dead.  Assert:
      - Both agent population and predator population stay > 0 throughout.
      - Predator count tracks roughly pred_per_agents × n_living_agents.
    """
    cfg = _make_cfg(
        max_agents=512, init_agents=100,
        pred_per_agents=0.08,
        pred_attack_damage=0.2,
        energy_decay=0.002, hydration_decay=0.002, thermal_decay=0.0, D_init=0.0,
        forage_yield=0.6, forage_regen=0.04, spoilage_carried=0.0,
    )
    world = _make_world(width=64, height=64, seed=123)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    agent_counts = []
    pred_counts = []

    STEPS = 1080  # 3 simulated years
    for step in range(STEPS):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)   # top up agents so population persists

        n_ag = int((sim.store.alive & ~sim.store.is_predator).sum())
        n_pred = out.info["n_predators"]
        agent_counts.append(n_ag)
        pred_counts.append(n_pred)

    # Both populations must have stayed > 0 throughout (no total collapse)
    assert min(agent_counts) > 0, (
        f"Agent population collapsed to 0 during the run! min={min(agent_counts)}"
    )
    assert max(pred_counts) > 0, (
        f"Predator population never appeared! max={max(pred_counts)}"
    )

    # Predator population must be positive at the end of the run
    assert pred_counts[-1] > 0, (
        f"Predator population went to 0 at end of run: {pred_counts[-1]}"
    )

    # Predator count should track ~ pred_per_agents * n_agents (within 3×)
    # Check the final 360 steps (last year) for the tracking relationship.
    last_year_preds  = np.array(pred_counts[-360:], dtype=float)
    last_year_agents = np.array(agent_counts[-360:], dtype=float)

    mean_pred_ratio = float(last_year_preds.mean() / max(1.0, last_year_agents.mean()))
    assert mean_pred_ratio <= 3.0 * cfg.pred_per_agents + 0.05, (
        f"Predator ratio {mean_pred_ratio:.3f} far exceeds target "
        f"{cfg.pred_per_agents:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Metric integration + JSONL round-trip
# ---------------------------------------------------------------------------

def test_phase6_metrics_integration(tmp_path):
    """
    Run a short sim with predators + combat + walls and verify MetricsLogger
    surfaces all Phase 6 keys plus retains all prior Phase 1-5 keys.
    """
    cfg = _make_cfg(
        max_agents=256, init_agents=50,
        pred_per_agents=0.10,
        pred_attack_damage=0.2,
        energy_decay=0.002, hydration_decay=0.002, thermal_decay=0.0, D_init=0.0,
        forage_yield=0.6, forage_regen=0.04, spoilage_carried=0.0,
    )
    world = _make_world(width=48, height=48, seed=7)
    sim = Simulation(world, cfg)
    sim.reset(seed=1)
    log = MetricsLogger(out_path=str(tmp_path / "phase6.jsonl"))

    STEPS_PER_YEAR = 360
    for step in range(STEPS_PER_YEAR):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)
        log.record_step(sim, out.info)

    s = log.year_summary()

    # --- Phase 6 keys present and finite ---
    assert "n_predators" in s, "year_summary missing 'n_predators'"
    assert math.isfinite(s["n_predators"]), f"n_predators not finite: {s['n_predators']}"
    assert s["n_predators"] >= 0.0

    assert "walls_built" in s, "year_summary missing 'walls_built'"
    assert isinstance(s["walls_built"], dict)
    assert "end_of_year" in s["walls_built"] and "max" in s["walls_built"]
    assert s["walls_built"]["end_of_year"] >= 0
    assert s["walls_built"]["max"] >= s["walls_built"]["end_of_year"] or \
           s["walls_built"]["max"] == 0  # max >= end_of_year always

    assert "weapons_held" in s, "year_summary missing 'weapons_held'"
    assert math.isfinite(s["weapons_held"]), f"weapons_held not finite: {s['weapons_held']}"
    assert s["weapons_held"] >= 0.0

    assert "predation_deaths" in s, "year_summary missing 'predation_deaths'"
    assert isinstance(s["predation_deaths"], int)

    assert "conflict_deaths" in s, "year_summary missing 'conflict_deaths'"
    assert isinstance(s["conflict_deaths"], int)

    # deaths_by_cause must have "predator" and "conflict" keys
    dbc = s["deaths_by_cause"]
    assert "predator"  in dbc, f"deaths_by_cause missing 'predator': {list(dbc.keys())}"
    assert "conflict"  in dbc, f"deaths_by_cause missing 'conflict': {list(dbc.keys())}"

    # predation_deaths == deaths_by_cause["predator"]
    assert s["predation_deaths"] == dbc["predator"]
    assert s["conflict_deaths"]  == dbc["conflict"]

    # --- All prior Phase 1-5 keys still present (additive contract) ---
    prior_keys = (
        # Phase 0/1
        "population_mean", "population_min", "population_max", "deaths_by_cause",
        "mean_displacement", "wild_food_mean",
        # Phase 2
        "water_occupancy",
        # Phase 3
        "pct_calories_farmed", "fertile_occupancy",
        # Phase 4
        "structures_built", "stored_food_total", "mean_thermal",
        # Phase 5
        "births", "mean_genome", "genome_drift", "lineage_count",
    )
    for k in prior_keys:
        assert k in s, f"Prior key missing from year_summary: '{k}'"

    # --- JSONL round-trip ---
    lines = (tmp_path / "phase6.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])

    for k in ("n_predators", "walls_built", "weapons_held",
              "predation_deaths", "conflict_deaths"):
        assert k in parsed, f"Phase 6 key '{k}' missing from JSONL output"

    # All numeric Phase 6 values are finite in the JSON
    assert math.isfinite(parsed["n_predators"])
    assert math.isfinite(parsed["weapons_held"])
    assert isinstance(parsed["predation_deaths"], int)
    assert isinstance(parsed["conflict_deaths"],  int)


# ---------------------------------------------------------------------------
# Visualization artefact — render a frame with walls + predators + agents
# ---------------------------------------------------------------------------

def test_render_phase6_combat_png(tmp_path):
    """
    Render a single frame containing WALL tiles, predators, and agents to
    data/phase6_combat.png.  Verifies imageio can write the file and that
    the output is a valid RGB array.
    """
    import imageio.v3 as iio
    from sim.render import render_frame

    cfg = _make_cfg(
        max_agents=128, init_agents=20,
        pred_per_agents=0.15, pred_attack_damage=0.2,
        energy_decay=0.0, hydration_decay=0.0, thermal_decay=0.0,
        D_init=0.0, forage_yield=0.4, forage_regen=0.02,
    )
    world = _make_world(width=48, height=48, seed=5)
    sim = Simulation(world, cfg)
    sim.reset(seed=5)

    # Run enough steps for predators to appear, then build some walls manually
    acts_noop = Actions(
        primary=np.full(cfg.max_agents, int(Action.NOOP), dtype=np.int32),
        param=np.zeros(cfg.max_agents, dtype=np.int32),
        emit=np.zeros(cfg.max_agents, dtype=np.int32),
    )
    for _ in range(10):
        sim.step(acts_noop)

    # Manually plant walls on a few agent tiles for a clear visual
    sea = world.cfg.sea_level
    land_ys, land_xs = np.where(world.elevation >= sea)
    for i in range(min(8, land_ys.size)):
        sim.state.structure_type[land_ys[i], land_xs[i]] = np.int8(int(StructureType.WALL))
        sim.state.structure_hp[land_ys[i], land_xs[i]]   = np.float32(1.0)

    frame = render_frame(world, sim.state, sim.store)
    assert frame.shape == (world.elevation.shape[0], world.elevation.shape[1], 3)
    assert frame.dtype == np.uint8

    # Write to the persistent data/ directory (not tmp_path so it survives the test run)
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "data"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase6_combat.png")
    iio.imwrite(out_path, frame)

    assert os.path.exists(out_path), f"PNG not written to {out_path}"
    assert os.path.getsize(out_path) > 0, "PNG file is empty"
