"""
Phase 6 combat tests — deterministic HP fighting, weapon crafting, and walls
(build-spec §2.6, §5 Phase 6).

Tests:
  1. ATTACK reduces a target agent's health by attack_damage (unarmed, no wall,
     no allies).
  2. A WEAPONED attacker deals weapon_attack_mult × damage.
  3. A target on a WALL tile takes wall_defense_mult × reduced damage.
  4. Group defense: a target with co-located allies takes less damage than a lone
     target.
  5. Mutual attack: two adjacent agents attacking each other BOTH take damage
     this step (pre-step HP snapshot).
  6. CRAFT consumes wood/stone/minerals and sets weapon=True; insufficient
     materials → no craft (masked/skipped).
  7. FORAGE on a mineral-rich (mountain) tile raises inv_minerals.
  8. WALL builds (structure_type=3) costing wall_cost.
  9. An agent killed by ATTACK is attributed death cause "conflict".
"""
from __future__ import annotations

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.state import EntityStore, WorldState
from sim.actions import Action, StructureType, DIRECTIONS, build_mask
from sim.dynamics import resolve_combat, craft_weapon, forage, build
from sim.reproduce import resolve_deaths
from sim.simulation import Simulation, Actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world(width: int = 32, height: int = 32, seed: int = 7) -> World:
    return World.generate(WorldConfig(width=width, height=height, seed=seed))


def _make_cfg(**kwargs) -> SimConfig:
    defaults = dict(
        max_agents=64,
        init_agents=4,
        attack_damage=0.15,
        weapon_attack_mult=2.0,
        wall_defense_mult=0.3,
        group_defense_per_ally=0.9,
        group_defense_floor=0.3,
        energy_decay=0.001,
        hydration_decay=0.001,
        thermal_decay=0.001,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _store(cfg: SimConfig, n: int = 1, **fields) -> EntityStore:
    """Create an EntityStore with `n` living agents at (0,0) with full stats."""
    s = EntityStore.create(cfg)
    s.alive[:n]      = True
    s.energy[:n]     = 1.0
    s.hydration[:n]  = 1.0
    s.thermal[:n]    = 1.0
    s.health[:n]     = 1.0
    for k, v in fields.items():
        arr = getattr(s, k)
        arr[:n] = v
    return s


def _actions(M: int, primary: int = int(Action.NOOP)) -> Actions:
    return Actions(
        primary=np.full(M, primary, dtype=np.int32),
        param=np.zeros(M, dtype=np.int32),
        emit=np.zeros(M, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Test 1: Unarmed attack reduces target health by attack_damage
# ---------------------------------------------------------------------------

def test_attack_reduces_health_unarmed_no_wall_no_allies():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    # Agent 0 at (3, 3), Agent 1 at (3, 4) — one tile apart East.
    store = _store(cfg, n=2)
    store.y[0] = 3; store.x[0] = 3
    store.y[1] = 3; store.x[1] = 4

    # Agent 0 attacks East (direction index 2 → (0, +1))
    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East

    attacker_idx = np.array([0], dtype=np.int32)
    resolve_combat(store, ws, attacker_idx, param, cfg)

    expected_health = 1.0 - cfg.attack_damage  # no weapon, no wall, no allies
    assert store.health[1] == pytest.approx(expected_health, abs=1e-5)
    assert store.health[0] == pytest.approx(1.0, abs=1e-5)  # attacker untouched


# ---------------------------------------------------------------------------
# Test 2: Armed attacker deals weapon_attack_mult × damage
# ---------------------------------------------------------------------------

def test_weaponed_attacker_does_multiplied_damage():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    store = _store(cfg, n=2)
    store.y[0] = 3; store.x[0] = 3
    store.y[1] = 3; store.x[1] = 4
    store.weapon[0] = True   # attacker is armed

    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East

    attacker_idx = np.array([0], dtype=np.int32)
    resolve_combat(store, ws, attacker_idx, param, cfg)

    expected_health = 1.0 - cfg.attack_damage * cfg.weapon_attack_mult
    assert store.health[1] == pytest.approx(expected_health, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Target on a WALL tile takes wall_defense_mult × reduced damage
# ---------------------------------------------------------------------------

def test_wall_reduces_damage_to_defender():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    # Place a wall at the target tile (3, 4)
    ws.structure_type[3, 4] = StructureType.WALL
    ws.structure_hp[3, 4]   = 1.0

    store = _store(cfg, n=2)
    store.y[0] = 3; store.x[0] = 3
    store.y[1] = 3; store.x[1] = 4

    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East

    attacker_idx = np.array([0], dtype=np.int32)
    resolve_combat(store, ws, attacker_idx, param, cfg)

    expected_health = 1.0 - cfg.attack_damage * cfg.wall_defense_mult
    assert store.health[1] == pytest.approx(expected_health, abs=1e-5)
    # Damage must be strictly less than without wall
    assert store.health[1] > 1.0 - cfg.attack_damage


# ---------------------------------------------------------------------------
# Test 4: Group defense — a target with allies takes less damage than a lone target
# ---------------------------------------------------------------------------

def test_group_defense_reduces_damage_with_allies():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    # 3 agents: slot 0 attacks East; slot 1 is the lone target; slot 2 has an ally.
    # Lone target test: agents 0 (attacker at 3,3), 1 (lone target at 3,4)
    store_lone = _store(cfg, n=2)
    store_lone.y[0] = 3; store_lone.x[0] = 3
    store_lone.y[1] = 3; store_lone.x[1] = 4

    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East
    attacker_idx = np.array([0], dtype=np.int32)
    resolve_combat(store_lone, ws, attacker_idx, param, cfg)
    lone_health = float(store_lone.health[1])

    # Group target test: agents 0 (attacker at 3,3), 1+2 both on target tile (3,4)
    store_group = _store(cfg, n=3)
    store_group.y[0] = 3; store_group.x[0] = 3
    store_group.y[1] = 3; store_group.x[1] = 4
    store_group.y[2] = 3; store_group.x[2] = 4  # ally co-located with target

    param2 = np.zeros(M, dtype=np.int32)
    param2[0] = 2  # East
    attacker_idx2 = np.array([0], dtype=np.int32)
    resolve_combat(store_group, ws, attacker_idx2, param2, cfg)
    # Both slot 1 and slot 2 should have taken damage, but less than lone
    group_health_1 = float(store_group.health[1])
    group_health_2 = float(store_group.health[2])

    # Group targets took less damage than the lone target
    assert group_health_1 > lone_health
    assert group_health_2 > lone_health


# ---------------------------------------------------------------------------
# Test 5: Mutual attack — both agents take damage (pre-step HP snapshot)
# ---------------------------------------------------------------------------

def test_mutual_attack_both_take_damage():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    # Agent 0 at (3,3), Agent 1 at (3,4)
    # Agent 0 attacks East (→ hits tile 3,4)
    # Agent 1 attacks West (dir 6 → (0,-1) → hits tile 3,3)
    store = _store(cfg, n=2)
    store.y[0] = 3; store.x[0] = 3
    store.y[1] = 3; store.x[1] = 4

    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East
    param[1] = 6  # West

    # Both agents attack simultaneously
    attacker_idx = np.array([0, 1], dtype=np.int32)
    resolve_combat(store, ws, attacker_idx, param, cfg)

    # BOTH agents should have taken damage this step
    assert store.health[0] < 1.0, "Agent 0 should have taken damage from agent 1"
    assert store.health[1] < 1.0, "Agent 1 should have taken damage from agent 0"

    # Damage values should match unarmed attack damage
    assert store.health[0] == pytest.approx(1.0 - cfg.attack_damage, abs=1e-5)
    assert store.health[1] == pytest.approx(1.0 - cfg.attack_damage, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 6: CRAFT consumes materials and sets weapon=True; insufficient materials = no craft
# ---------------------------------------------------------------------------

def test_craft_weapon_consumes_materials_and_arms_agent():
    cfg   = _make_cfg()
    M     = cfg.max_agents
    store = _store(cfg, n=1)

    # Give agent exactly the weapon cost
    wc = cfg.weapon_cost
    store.inv_wood[0]     = np.float32(wc["wood"])
    store.inv_stone[0]    = np.float32(wc["stone"])
    store.inv_minerals[0] = np.float32(wc["minerals"])
    store.weapon[0]       = False

    crafted = craft_weapon(store, np.array([0], dtype=np.int32), cfg)

    assert crafted == 1
    assert bool(store.weapon[0]) is True
    assert float(store.inv_wood[0])     == pytest.approx(0.0, abs=1e-5)
    assert float(store.inv_stone[0])    == pytest.approx(0.0, abs=1e-5)
    assert float(store.inv_minerals[0]) == pytest.approx(0.0, abs=1e-5)


def test_craft_weapon_fails_with_insufficient_materials():
    cfg   = _make_cfg()
    M     = cfg.max_agents
    store = _store(cfg, n=1)

    # Give the agent zero materials
    store.inv_wood[0]     = np.float32(0.0)
    store.inv_stone[0]    = np.float32(0.0)
    store.inv_minerals[0] = np.float32(0.0)
    store.weapon[0]       = False

    crafted = craft_weapon(store, np.array([0], dtype=np.int32), cfg)

    assert crafted == 0
    assert bool(store.weapon[0]) is False  # no weapon crafted


# ---------------------------------------------------------------------------
# Test 7: FORAGE on a mineral-rich tile raises inv_minerals
# ---------------------------------------------------------------------------

def test_forage_on_mountain_tile_raises_inv_minerals():
    """Use the real world so we get a tile with minerals > 0."""
    world = _make_world()
    cfg   = _make_cfg()

    # Find a tile with minerals > 0
    min_layer = world.base_resources["minerals"]
    rich_ys, rich_xs = np.where(min_layer > 0.1)
    assert rich_ys.size > 0, "Test world has no mineral-rich tiles"

    ty, tx = int(rich_ys[0]), int(rich_xs[0])

    ws    = WorldState.create(*world.elevation.shape)
    ws.wild_remaining[:] = world.base_resources["wild_food"]
    store = _store(cfg, n=1)
    store.y[0] = ty
    store.x[0] = tx

    before = float(store.inv_minerals[0])
    forage(store, ws, np.array([0], dtype=np.int32), cfg, world=world)
    after = float(store.inv_minerals[0])

    assert after > before, f"inv_minerals did not increase on mineral-rich tile ({ty},{tx})"


# ---------------------------------------------------------------------------
# Test 8: BUILD with WALL param sets structure_type=3 costing wall_cost
# ---------------------------------------------------------------------------

def test_build_wall_creates_wall_structure():
    cfg   = _make_cfg()
    M     = cfg.max_agents
    ws    = WorldState.create(8, 8)

    # Give agent the wall cost
    wc = cfg.wall_cost
    store = _store(cfg, n=1)
    store.inv_wood[0]  = np.float32(wc["wood"] + 1.0)   # extra to be sure
    store.inv_stone[0] = np.float32(wc["stone"] + 1.0)

    param = np.zeros(M, dtype=np.int32)
    param[0] = int(StructureType.WALL)   # param = 3

    built = build(store, ws, np.array([0], dtype=np.int32), param, cfg)

    assert built == 1
    assert int(ws.structure_type[0, 0]) == int(StructureType.WALL)
    assert float(ws.structure_hp[0, 0]) == pytest.approx(cfg.structure_init_hp, abs=1e-5)
    # Cost deducted
    assert float(store.inv_wood[0])  == pytest.approx(1.0, abs=1e-5)
    assert float(store.inv_stone[0]) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Agent killed by ATTACK is attributed "conflict" death cause
# ---------------------------------------------------------------------------

def test_death_from_attack_attributed_conflict():
    cfg = _make_cfg()
    M   = cfg.max_agents
    ws  = WorldState.create(8, 8)

    # Start target with tiny health so one armed hit kills it
    store = _store(cfg, n=2)
    store.y[0] = 3; store.x[0] = 3
    store.y[1] = 3; store.x[1] = 4
    store.health[1] = np.float32(0.05)  # very low health
    store.weapon[0] = True              # attacker armed → big damage

    param = np.zeros(M, dtype=np.int32)
    param[0] = 2  # East

    attacker_idx = np.array([0], dtype=np.int32)
    combat_result = resolve_combat(store, ws, attacker_idx, param, cfg)

    # Confirm target health ≤ 0
    assert store.health[1] <= 0.0, "Target should have been killed"

    metrics = {}
    resolve_deaths(
        store,
        metrics_deaths=metrics,
        conflict_slots=combat_result["conflict_slots"],
    )

    assert metrics.get("conflict", 0) == 1, (
        f"Expected 1 conflict death, got metrics={metrics}"
    )
    assert not store.alive[1], "Dead agent slot should be marked not alive"


# ---------------------------------------------------------------------------
# Test 10: build_mask Phase 6 — ATTACK enabled when adjacent entity exists
# ---------------------------------------------------------------------------

def test_build_mask_attack_enabled_when_adjacent_entity():
    world = _make_world()
    cfg   = _make_cfg()
    ws    = WorldState.create(*world.elevation.shape)

    store = EntityStore.create(cfg)
    # Agent 0 at (5, 5), Agent 1 at (5, 6) — adjacent East
    store.alive[:2]     = True
    store.y[0] = 5; store.x[0] = 5
    store.y[1] = 5; store.x[1] = 6
    store.energy[:2]    = 1.0
    store.hydration[:2] = 1.0
    store.thermal[:2]   = 1.0
    store.health[:2]    = 1.0

    mask = build_mask(world, ws, store, cfg)

    # Both agents should have ATTACK unmasked (each is adjacent to the other)
    assert mask[0, Action.ATTACK], "Agent 0 should be able to ATTACK agent 1"
    assert mask[1, Action.ATTACK], "Agent 1 should be able to ATTACK agent 0"


def test_build_mask_attack_disabled_when_no_adjacent_entity():
    world = _make_world()
    cfg   = _make_cfg()
    ws    = WorldState.create(*world.elevation.shape)

    store = EntityStore.create(cfg)
    # Single isolated agent — no one adjacent
    store.alive[0]    = True
    store.y[0] = 5; store.x[0] = 5
    store.energy[0]   = 1.0
    store.hydration[0]= 1.0
    store.thermal[0]  = 1.0
    store.health[0]   = 1.0

    mask = build_mask(world, ws, store, cfg)
    assert not mask[0, Action.ATTACK], "Isolated agent should not have ATTACK unmasked"


# ---------------------------------------------------------------------------
# Test 11: build_mask Phase 6 — CRAFT enabled when unarmed + can afford
# ---------------------------------------------------------------------------

def test_build_mask_craft_enabled_when_can_afford():
    world = _make_world()
    cfg   = _make_cfg()
    ws    = WorldState.create(*world.elevation.shape)

    store = EntityStore.create(cfg)
    store.alive[0]    = True
    store.y[0] = 5; store.x[0] = 5
    store.energy[0]   = 1.0
    store.hydration[0]= 1.0
    store.thermal[0]  = 1.0
    store.health[0]   = 1.0
    wc = cfg.weapon_cost
    store.inv_wood[0]     = np.float32(wc["wood"])
    store.inv_stone[0]    = np.float32(wc["stone"])
    store.inv_minerals[0] = np.float32(wc["minerals"])
    store.weapon[0]       = False

    mask = build_mask(world, ws, store, cfg)
    assert mask[0, Action.CRAFT], "Agent with enough materials should have CRAFT enabled"


def test_build_mask_craft_disabled_when_already_armed():
    world = _make_world()
    cfg   = _make_cfg()
    ws    = WorldState.create(*world.elevation.shape)

    store = EntityStore.create(cfg)
    store.alive[0]    = True
    store.y[0] = 5; store.x[0] = 5
    store.energy[0]   = 1.0
    store.hydration[0]= 1.0
    store.thermal[0]  = 1.0
    store.health[0]   = 1.0
    wc = cfg.weapon_cost
    store.inv_wood[0]     = np.float32(wc["wood"])
    store.inv_stone[0]    = np.float32(wc["stone"])
    store.inv_minerals[0] = np.float32(wc["minerals"])
    store.weapon[0]       = True   # already armed

    mask = build_mask(world, ws, store, cfg)
    assert not mask[0, Action.CRAFT], "Already armed agent should NOT have CRAFT enabled"


# ---------------------------------------------------------------------------
# Test 12: Full simulation step with ATTACK → info contains weapons_crafted / attacks
# ---------------------------------------------------------------------------

def test_simulation_step_attack_info_keys():
    world = _make_world()
    cfg   = _make_cfg()
    sim   = Simulation(world, cfg)
    sim.reset(seed=42)

    M = cfg.max_agents
    acts = _actions(M, int(Action.NOOP))
    out  = sim.step(acts)

    assert "weapons_crafted" in out.info
    assert "attacks"         in out.info
    assert out.info["weapons_crafted"] == 0
    assert out.info["attacks"]         == 0
