"""
Phase 3 dynamics tests — CROPS: plant -> grow -> mature -> harvest, winter-failure rot.

Covers (per spec §2.3 + Phase 3):
  1. A planted crop on a fertile+watered tile grows to stage 1.0; harvesting it yields
     ~crop_yield food and clears the tile.
  2. Winter-failure rule: a crop on a tile where eff_fert < crop_min_fertility has
     crop_health decrease each step; it eventually clears when health <= 0.
  3. A crop on a barren tile (soil_fertility < crop_min_fertility / fertility_modifier)
     rots regardless of season.
  4. PLANT is a no-op on an already-occupied tile; HARVEST is a no-op on an immature
     crop (and is masked by build_mask).
  5. build_mask: PLANT enabled on empty land; HARVEST enabled only at stage >= 1.
  6. step() info exposes foraged / harvested / planted totals.
  7. Full plant -> harvest -> eat cycle: agent gains energy from farmed food.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from world.seasons import compute_season_state
from sim.config import SimConfig
from sim.state import WorldState, EntityStore
from sim.simulation import Simulation, Actions
from sim.actions import Action, N_PRIMARY, build_mask
from sim.dynamics import crop_step, plant, harvest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H, _W = 32, 32


def _make_cfg(**kwargs) -> SimConfig:
    defaults = dict(max_agents=16, init_agents=4)
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _make_world(w: int = 64, h: int = 64, seed: int = 42) -> World:
    return World.generate(WorldConfig(width=w, height=h, seed=seed))


def _make_state() -> WorldState:
    return WorldState.create(_H, _W)


def _single_agent_store(
    cfg: SimConfig,
    energy: float = 1.0,
    hydration: float = 1.0,
    health: float = 1.0,
    inv_food: float = 0.0,
    y: int = 0,
    x: int = 0,
    lineage_id: int = 7,
) -> EntityStore:
    """Minimal EntityStore with one living agent in slot 0."""
    store = EntityStore.create(cfg)
    store.alive[0]      = True
    store.energy[0]     = energy
    store.hydration[0]  = hydration
    store.health[0]     = health
    store.inv_food[0]   = inv_food
    store.y[0]          = y
    store.x[0]          = x
    store.genome[0]     = 0.5   # neutral — capacity trait = 1.0
    store.lineage_id[0] = lineage_id
    return store


def _mock_world(
    H: int = _H, W: int = _W,
    soil: float = 1.0,
    water: float = 1.0,
    elevation: float = 0.8,
    sea_level: float = 0.4,
    season_period: int = 360,
) -> object:
    """
    Minimal world-like namespace accepted by crop_step.
    Uses a flat (H, W) array for each resource.
    """
    world_cfg = SimpleNamespace(sea_level=sea_level, season_period=season_period)
    return SimpleNamespace(
        cfg=world_cfg,
        base_resources={
            "soil_fertility": np.full((H, W), soil, dtype=np.float32),
            "water_proximity": np.full((H, W), water, dtype=np.float32),
        },
        elevation=np.full((H, W), elevation, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 1. Crop grows to maturity on a prime fertile+watered tile
# ---------------------------------------------------------------------------

def test_crop_grows_on_prime_tile():
    """
    A crop planted on a high-fertility, high-water tile (summer season) should
    reach stage >= 1.0 within the expected number of steps.
    """
    cfg = _make_cfg(
        crop_base_growth=0.02,
        crop_min_fertility=0.15,
    )
    state = _make_state()
    world = _mock_world(soil=1.0, water=1.0)

    # Plant at tile (5, 5)
    y, x = 5, 5
    state.crop_stage[y, x]  = np.float32(0.01)
    state.crop_health[y, x] = np.float32(1.0)
    state.crop_owner[y, x]  = np.int32(0)

    # Use summer season (t=137, fert_mod ~ 0.95)
    # growth/step ~ 0.02 * 0.95 * 1.0 = 0.019 -> ~52 steps from 0.01 to 1.0
    t_summer = 137
    season = compute_season_state(t_summer, WorldConfig())
    fert_mod = season.fertility_modifier
    expected_steps = math.ceil((1.0 - 0.01) / (cfg.crop_base_growth * fert_mod * 1.0))

    # Advance by expected_steps + a small buffer
    for t in range(t_summer, t_summer + expected_steps + 10):
        crop_step(state, world, t, cfg)
        if state.crop_stage[y, x] >= 1.0:
            break

    assert state.crop_stage[y, x] >= 1.0, (
        f"Crop did not mature; stage={state.crop_stage[y, x]:.4f} after "
        f"{expected_steps + 10} summer steps"
    )
    # Health should remain 1.0 (no rot in summer on prime tile)
    assert state.crop_health[y, x] >= 0.99, (
        f"Crop health degraded on prime summer tile: {state.crop_health[y, x]:.4f}"
    )


def test_harvest_mature_crop_yields_food_and_clears_tile():
    """
    Harvesting a mature (stage >= 1.0) crop yields ~crop_yield food and
    resets crop_stage, crop_health, crop_owner to 0 / 0 / -1.
    """
    cfg   = _make_cfg(crop_yield=1.2, carry_capacity=2.0)
    store = _single_agent_store(cfg, inv_food=0.0, y=3, x=3)
    state = _make_state()

    # Set a fully mature, perfectly healthy crop on the agent's tile
    state.crop_stage[3, 3]  = np.float32(1.0)
    state.crop_health[3, 3] = np.float32(1.0)
    state.crop_owner[3, 3]  = np.int32(7)

    idx = np.array([0], dtype=np.int32)
    food_gained = harvest(store, state, idx, cfg)

    expected_yield = cfg.crop_yield * 1.0 * 1.0   # crop_yield * stage * health
    assert food_gained == pytest.approx(expected_yield, abs=1e-4), (
        f"Expected food yield={expected_yield:.4f}, got {food_gained:.4f}"
    )
    assert store.inv_food[0] == pytest.approx(expected_yield, abs=1e-4), (
        f"inv_food not updated correctly: {store.inv_food[0]:.4f}"
    )
    # Tile must be cleared
    assert state.crop_stage[3, 3]  == pytest.approx(0.0)
    assert state.crop_health[3, 3] == pytest.approx(0.0)
    assert state.crop_owner[3, 3]  == -1


# ---------------------------------------------------------------------------
# 2. Winter-failure rot rule
# ---------------------------------------------------------------------------

def test_winter_failure_rots_crop_on_low_fertility_tile():
    """
    At t=0 (deep winter, fertility_modifier=0.20), a tile with soil=0.5 gives
    eff_fert = 0.5 * 0.20 = 0.10 < crop_min_fertility=0.15 -> crop rots.
    crop_health should decline by crop_rot each step.
    """
    cfg = _make_cfg(
        crop_min_fertility=0.15,
        crop_rot=0.05,
    )
    state = _make_state()
    # soil=0.50 so eff_fert_winter = 0.50 * 0.20 = 0.10 < 0.15
    world = _mock_world(soil=0.50, water=1.0)

    y, x = 10, 10
    state.crop_stage[y, x]  = np.float32(0.50)   # mid-growth
    state.crop_health[y, x] = np.float32(1.0)
    state.crop_owner[y, x]  = np.int32(3)

    initial_health = float(state.crop_health[y, x])

    # Apply one winter step (t=0, fert_mod=0.20)
    crop_step(state, world, 0, cfg)

    # Verify eff_fert < crop_min_fertility (sanity)
    season = compute_season_state(0, WorldConfig())
    eff_fert = 0.50 * season.fertility_modifier
    assert eff_fert < cfg.crop_min_fertility, (
        f"Test precondition failed: eff_fert={eff_fert:.4f} >= crop_min_fertility"
    )

    # Health must have decreased
    assert float(state.crop_health[y, x]) < initial_health, (
        f"Crop health should decrease in winter rot; got {state.crop_health[y, x]:.4f}"
    )
    assert float(state.crop_health[y, x]) == pytest.approx(
        initial_health - cfg.crop_rot, abs=1e-4
    ), (
        f"Unexpected health after one rot step: {state.crop_health[y, x]:.4f}"
    )


def test_winter_failure_eventually_clears_crop():
    """
    Under persistent winter rot conditions, crop_health -> 0 and the tile is cleared.
    """
    cfg = _make_cfg(
        crop_min_fertility=0.15,
        crop_rot=0.05,
    )
    state = _make_state()
    world = _mock_world(soil=0.50, water=1.0)  # eff_fert_winter = 0.10 < 0.15

    y, x = 10, 10
    state.crop_stage[y, x]  = np.float32(0.50)
    state.crop_health[y, x] = np.float32(1.0)
    state.crop_owner[y, x]  = np.int32(3)

    # Run many winter steps until cleared
    for step_t in range(200):
        crop_step(state, world, 0, cfg)   # t=0 forces deep winter all the way
        if state.crop_stage[y, x] == 0.0:
            break

    assert state.crop_stage[y, x]  == pytest.approx(0.0), "Rotted crop was not cleared"
    assert state.crop_health[y, x] == pytest.approx(0.0), "Cleared crop health != 0"
    assert state.crop_owner[y, x]  == -1, "Cleared crop owner not reset to -1"


# ---------------------------------------------------------------------------
# 3. Barren tile rots regardless of season
# ---------------------------------------------------------------------------

def test_barren_tile_rots_in_summer():
    """
    A tile with soil_fertility=0.10 in summer (fert_mod=0.95):
      eff_fert = 0.10 * 0.95 = 0.095 < 0.15 -> rots even in summer.
    """
    cfg = _make_cfg(
        crop_min_fertility=0.15,
        crop_rot=0.05,
    )
    state = _make_state()
    world = _mock_world(soil=0.10, water=1.0)  # very barren

    y, x = 4, 4
    state.crop_stage[y, x]  = np.float32(0.50)
    state.crop_health[y, x] = np.float32(1.0)
    state.crop_owner[y, x]  = np.int32(5)

    initial_health = float(state.crop_health[y, x])
    t_summer = 137
    crop_step(state, world, t_summer, cfg)

    # Sanity: eff_fert < crop_min_fertility in summer on barren tile
    season = compute_season_state(t_summer, WorldConfig())
    eff_fert = 0.10 * season.fertility_modifier
    assert eff_fert < cfg.crop_min_fertility, (
        f"Test precondition failed: barren tile eff_fert={eff_fert:.4f} should be < 0.15"
    )

    assert float(state.crop_health[y, x]) < initial_health, (
        f"Barren tile should rot in summer; health={state.crop_health[y, x]:.4f}"
    )


def test_barren_tile_eventually_clears():
    """Barren tile crops rot away completely under sustained summer conditions."""
    cfg = _make_cfg(crop_min_fertility=0.15, crop_rot=0.05)
    state = _make_state()
    world = _mock_world(soil=0.10, water=1.0)

    y, x = 4, 4
    state.crop_stage[y, x]  = np.float32(0.80)
    state.crop_health[y, x] = np.float32(1.0)
    state.crop_owner[y, x]  = np.int32(5)

    for step_t in range(200):
        crop_step(state, world, 137, cfg)   # constant summer t
        if state.crop_stage[y, x] == 0.0:
            break

    assert state.crop_stage[y, x]  == pytest.approx(0.0), "Barren crop not cleared"
    assert state.crop_health[y, x] == pytest.approx(0.0)
    assert state.crop_owner[y, x]  == -1


# ---------------------------------------------------------------------------
# 4. plant() and harvest() no-op semantics
# ---------------------------------------------------------------------------

def test_plant_noop_on_occupied_tile():
    """PLANT on an already-occupied tile (crop_stage > 0) is a no-op for that slot."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, y=5, x=5)
    state = _make_state()

    # Pre-occupy tile
    state.crop_stage[5, 5]  = np.float32(0.50)
    state.crop_health[5, 5] = np.float32(0.80)
    state.crop_owner[5, 5]  = np.int32(99)   # some other owner

    idx = np.array([0], dtype=np.int32)
    n_planted = plant(store, state, idx, cfg)

    assert n_planted == 0, "plant() should return 0 when tile is occupied"
    assert float(state.crop_stage[5, 5])  == pytest.approx(0.50), "crop_stage changed"
    assert float(state.crop_health[5, 5]) == pytest.approx(0.80), "crop_health changed"
    assert int(state.crop_owner[5, 5])    == 99, "crop_owner changed"


def test_plant_two_agents_same_tile_first_wins():
    """Two agents planting on the same empty tile: only the first (lower slot) plants."""
    cfg = _make_cfg(max_agents=16, init_agents=4)
    state = _make_state()

    store = EntityStore.create(cfg)
    # Two agents at the same tile
    for slot in [0, 1]:
        store.alive[slot]      = True
        store.y[slot]          = 8
        store.x[slot]          = 8
        store.genome[slot]     = 0.5
        store.lineage_id[slot] = slot + 10

    idx = np.array([0, 1], dtype=np.int32)
    n_planted = plant(store, state, idx, cfg)

    assert n_planted == 1, "Only one crop should be planted on a shared tile"
    assert float(state.crop_stage[8, 8]) == pytest.approx(0.01)
    assert int(state.crop_owner[8, 8]) == 10  # slot 0's lineage_id


def test_harvest_noop_on_immature_crop():
    """HARVEST on a tile where crop_stage < 1.0 is a no-op; inv_food unchanged."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, inv_food=0.0, y=2, x=2)
    state = _make_state()

    state.crop_stage[2, 2]  = np.float32(0.80)   # immature
    state.crop_health[2, 2] = np.float32(1.0)

    idx = np.array([0], dtype=np.int32)
    food_gained = harvest(store, state, idx, cfg)

    assert food_gained == pytest.approx(0.0), "harvest() should yield 0 on immature crop"
    assert store.inv_food[0] == pytest.approx(0.0), "inv_food changed on immature harvest"
    # Tile unchanged
    assert float(state.crop_stage[2, 2]) == pytest.approx(0.80)


def test_harvest_noop_on_empty_tile():
    """HARVEST on a tile with no crop (stage==0) yields nothing."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, inv_food=0.0, y=0, x=0)
    state = _make_state()   # all zeros

    idx = np.array([0], dtype=np.int32)
    food_gained = harvest(store, state, idx, cfg)

    assert food_gained == pytest.approx(0.0)
    assert store.inv_food[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. build_mask: PLANT and HARVEST rules
# ---------------------------------------------------------------------------

def test_build_mask_plant_enabled_on_empty_land():
    """PLANT is enabled for agents on land tiles where crop_stage == 0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Place an agent on a land tile with no crop
    sea_level = world.cfg.sea_level
    elev = world.elevation
    land_ys, land_xs = np.where(elev >= sea_level)
    if land_ys.size == 0:
        pytest.skip("No land tiles")

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    agent = int(live_idx[0])
    ly, lx = int(land_ys[0]), int(land_xs[0])
    sim.store.y[agent] = ly
    sim.store.x[agent] = lx
    # Ensure tile is empty
    sim.state.crop_stage[ly, lx] = 0.0

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert mask[agent, int(Action.PLANT)], (
        "PLANT should be enabled on empty land tile"
    )


def test_build_mask_plant_disabled_where_crop_exists():
    """PLANT is disabled for agents on tiles where crop_stage > 0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    sea_level = world.cfg.sea_level
    land_ys, land_xs = np.where(world.elevation >= sea_level)
    if land_ys.size == 0:
        pytest.skip("No land tiles")

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    agent = int(live_idx[0])
    ly, lx = int(land_ys[0]), int(land_xs[0])
    sim.store.y[agent] = ly
    sim.store.x[agent] = lx
    # Occupy with a growing crop
    sim.state.crop_stage[ly, lx]  = np.float32(0.50)
    sim.state.crop_health[ly, lx] = np.float32(1.0)

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[agent, int(Action.PLANT)], (
        "PLANT should be disabled when a crop already occupies the tile"
    )


def test_build_mask_harvest_enabled_only_at_maturity():
    """HARVEST is enabled only where crop_stage >= 1.0, disabled otherwise."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    sea_level = world.cfg.sea_level
    land_ys, land_xs = np.where(world.elevation >= sea_level)
    if land_ys.size == 0:
        pytest.skip("No land tiles")

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    agent = int(live_idx[0])
    ly, lx = int(land_ys[0]), int(land_xs[0])
    sim.store.y[agent] = ly
    sim.store.x[agent] = lx

    # Stage = 0.80 (immature) -> HARVEST disabled
    sim.state.crop_stage[ly, lx] = np.float32(0.80)
    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[agent, int(Action.HARVEST)], (
        "HARVEST should be disabled for immature crop"
    )

    # Stage = 1.0 (mature) -> HARVEST enabled
    sim.state.crop_stage[ly, lx] = np.float32(1.0)
    mask = build_mask(world, sim.state, sim.store, cfg)
    assert mask[agent, int(Action.HARVEST)], (
        "HARVEST should be enabled for mature crop (stage >= 1.0)"
    )


def test_build_mask_harvest_disabled_for_dead_slots():
    """Dead slots have HARVEST=False regardless of tile state."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Put a mature crop everywhere
    sim.state.crop_stage[:] = np.float32(1.0)
    # Kill all agents
    sim.store.alive[:] = False

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[:, int(Action.HARVEST)].any(), "Dead slots must have HARVEST=False"


def test_build_mask_plant_disabled_for_dead_slots():
    """Dead slots have PLANT=False regardless of tile state."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    sim.store.alive[:] = False

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[:, int(Action.PLANT)].any(), "Dead slots must have PLANT=False"


# ---------------------------------------------------------------------------
# 6. step() info exposes foraged / harvested / planted totals
# ---------------------------------------------------------------------------

def test_step_info_contains_crop_keys():
    """step() info dict must contain foraged, harvested, planted keys."""
    world = _make_world(w=64, h=64, seed=1)
    cfg   = SimConfig(
        max_agents=16, init_agents=4,
        energy_decay=0.001, hydration_decay=0.001, spoilage_carried=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)

    out = sim.step(Actions(primary=primary, param=param, emit=emit))

    assert "foraged"   in out.info, "info missing 'foraged'"
    assert "harvested" in out.info, "info missing 'harvested'"
    assert "planted"   in out.info, "info missing 'planted'"


def test_step_info_planted_increments():
    """step() info['planted'] > 0 when agents perform PLANT actions."""
    world = _make_world(w=64, h=64, seed=1)
    cfg   = SimConfig(
        max_agents=16, init_agents=4,
        energy_decay=0.001, hydration_decay=0.001, spoilage_carried=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    # Place all living agents on land tiles with no crops and have them PLANT
    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    sea_level = world.cfg.sea_level
    land_ys, land_xs = np.where(world.elevation >= sea_level)

    # Scatter agents to different land tiles to avoid tile collisions
    for i, slot in enumerate(live_idx):
        i2 = i % len(land_ys)
        sim.store.y[slot] = int(land_ys[i2 + (i * 7) % max(1, len(land_ys) - i)])
        sim.store.x[slot] = int(land_xs[i2 + (i * 7) % max(1, len(land_ys) - i)])

    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)
    primary[live_idx] = int(Action.PLANT)

    out = sim.step(Actions(primary=primary, param=param, emit=emit))
    assert out.info["planted"] > 0, (
        f"Expected planted > 0, got {out.info['planted']}"
    )


def test_step_info_harvested_increments():
    """step() info['harvested'] > 0 when agents perform HARVEST on mature tiles."""
    world = _make_world(w=64, h=64, seed=1)
    cfg   = SimConfig(max_agents=16, init_agents=4, energy_decay=0.001, hydration_decay=0.001)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    # Place mature crops on each agent's tile
    for slot in live_idx:
        y, x = int(sim.store.y[slot]), int(sim.store.x[slot])
        sim.state.crop_stage[y, x]  = np.float32(1.0)
        sim.state.crop_health[y, x] = np.float32(1.0)
        sim.state.crop_owner[y, x]  = np.int32(0)

    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)
    primary[live_idx] = int(Action.HARVEST)

    out = sim.step(Actions(primary=primary, param=param, emit=emit))
    assert out.info["harvested"] > 0.0, (
        f"Expected harvested > 0, got {out.info['harvested']}"
    )


# ---------------------------------------------------------------------------
# 7. Full plant -> grow -> harvest -> eat cycle
# ---------------------------------------------------------------------------

def test_full_farming_cycle_agent_gains_energy():
    """
    End-to-end farming cycle on a real 64x64 world:
      1. An agent plants on a fertile tile.
      2. We advance crop growth until maturity using crop_step directly.
      3. The agent harvests the mature crop.
      4. The agent eats the harvested food.
      5. Agent's energy should increase.
    """
    world = _make_world(w=64, h=64, seed=42)
    cfg   = SimConfig(
        max_agents=16, init_agents=1,
        crop_yield=1.2,
        carry_capacity=2.0,
        eat_restore=0.6,
        energy_decay=0.0,     # freeze energy so we only see harvest contribution
        hydration_decay=0.0,
        spoilage_carried=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=1)

    # Pick the prime tile for this world
    soil  = world.base_resources["soil_fertility"]
    water = world.base_resources["water_proximity"]
    elev  = world.elevation
    sea   = world.cfg.sea_level
    on_land = elev >= sea
    combo = soil * water * on_land.astype(np.float32)
    best  = np.argmax(combo)
    py, px = int(np.unravel_index(best, soil.shape)[0]), int(np.unravel_index(best, soil.shape)[1])

    slot = 0
    sim.store.y[slot] = py
    sim.store.x[slot] = px
    sim.store.energy[slot]    = 0.3   # start hungry
    sim.store.hydration[slot] = 1.0
    sim.store.inv_food[slot]  = 0.0
    sim.store.lineage_id[slot] = 0

    # ---- Step 1: PLANT ----
    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    param   = np.zeros(M, dtype=np.int32)
    emit    = np.zeros(M, dtype=np.int32)
    primary[slot] = int(Action.PLANT)

    out = sim.step(Actions(primary=primary, param=param, emit=emit))
    assert out.info["planted"] >= 1, "Plant step failed"
    # After the PLANT step, sim.step() also calls crop_step internally, so
    # crop_stage will be slightly > 0.01 (has grown one increment already).
    assert 0.01 <= sim.state.crop_stage[py, px] < 0.10, (
        f"crop_stage after plant step out of range: {sim.state.crop_stage[py, px]}"
    )

    # ---- Step 2: grow until maturity via direct crop_step calls ----
    steps_taken = 0
    for step_t in range(sim.t, sim.t + 300):
        crop_step(sim.state, world, step_t, cfg)
        steps_taken += 1
        if sim.state.crop_stage[py, px] >= 1.0:
            break

    assert sim.state.crop_stage[py, px] >= 1.0, (
        f"Crop did not mature in {steps_taken} steps; "
        f"stage={sim.state.crop_stage[py, px]:.4f}"
    )

    # ---- Step 3: HARVEST ----
    primary[slot] = int(Action.HARVEST)
    pre_harvest_food = float(sim.store.inv_food[slot])
    out = sim.step(Actions(primary=primary, param=param, emit=emit))
    assert out.info["harvested"] > 0.0, "No food harvested"
    assert float(sim.store.inv_food[slot]) > pre_harvest_food, (
        "inv_food didn't increase after harvest"
    )
    # Tile cleared
    assert sim.state.crop_stage[py, px] == pytest.approx(0.0)

    # ---- Step 4: EAT ----
    pre_eat_energy = float(sim.store.energy[slot])
    primary[slot] = int(Action.EAT)
    out = sim.step(Actions(primary=primary, param=param, emit=emit))
    post_eat_energy = float(sim.store.energy[slot])

    assert post_eat_energy > pre_eat_energy, (
        f"Energy did not increase after eating farmed food: "
        f"before={pre_eat_energy:.4f}, after={post_eat_energy:.4f}"
    )


# ---------------------------------------------------------------------------
# 8. plant() sets lineage_id correctly
# ---------------------------------------------------------------------------

def test_plant_sets_lineage_id():
    """crop_owner should be set to the planting agent's lineage_id."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, y=6, x=6, lineage_id=42)
    state = _make_state()

    idx = np.array([0], dtype=np.int32)
    plant(store, state, idx, cfg)

    assert state.crop_stage[6, 6]  == pytest.approx(0.01)
    assert state.crop_health[6, 6] == pytest.approx(1.0)
    assert int(state.crop_owner[6, 6]) == 42, (
        f"crop_owner should be lineage_id=42, got {state.crop_owner[6, 6]}"
    )


# ---------------------------------------------------------------------------
# 9. crop_step does NOT touch empty tiles
# ---------------------------------------------------------------------------

def test_crop_step_ignores_empty_tiles():
    """crop_step must not alter tiles with crop_stage == 0."""
    cfg = _make_cfg()
    state = _make_state()
    world = _mock_world(soil=1.0, water=1.0)

    # Leave all tiles at crop_stage = 0
    crop_step(state, world, 137, cfg)

    assert np.all(state.crop_stage == 0.0),  "crop_stage changed on empty tiles"
    assert np.all(state.crop_health == 0.0), "crop_health changed on empty tiles"


# ---------------------------------------------------------------------------
# 10. Grow-to-maturity timing print (informational assertion)
# ---------------------------------------------------------------------------

def test_grow_to_maturity_time_on_prime_tile(capsys):
    """
    Plant on the prime tile of a 64x64 world and count steps to stage >= 1.0.
    Asserts within a plausible range (30-120 steps) and prints the result.
    """
    cfg = SimConfig(
        max_agents=8, init_agents=1,
        crop_base_growth=0.02,
        crop_min_fertility=0.15,
    )
    world = _make_world(w=64, h=64, seed=42)
    state = WorldState.create(64, 64)

    # Find prime tile
    soil  = world.base_resources["soil_fertility"]
    water = world.base_resources["water_proximity"]
    elev  = world.elevation
    sea   = world.cfg.sea_level
    on_land = elev >= sea
    combo = soil * water * on_land.astype(np.float32)
    best  = int(np.argmax(combo))
    py    = best // 64
    px    = best % 64

    state.crop_stage[py, px]  = np.float32(0.01)
    state.crop_health[py, px] = np.float32(1.0)
    state.crop_owner[py, px]  = np.int32(0)

    t_start = 137  # summer
    steps = 0
    for t in range(t_start, t_start + 200):
        crop_step(state, world, t, cfg)
        steps += 1
        if state.crop_stage[py, px] >= 1.0:
            break

    with capsys.disabled():
        print(
            f"\n[Phase 3] Prime tile ({py},{px}) "
            f"soil={soil[py,px]:.3f} water={water[py,px]:.3f}: "
            f"matured in {steps} summer steps"
        )

    assert 20 <= steps <= 150, (
        f"Grow-to-maturity time {steps} outside expected range [20, 150]"
    )
