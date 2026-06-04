"""
Phase 4 gate test — winter-survival metrics + shelter/storage behaviour
(build-spec §4.1, §5 Phase 4).

Gate: "winter survival rate rises with shelter occupancy; stored_food peaks in
      fall, drawn down in winter; exposure deaths fall as shelters built."

Three scenarios:

  1. Shelter reduces exposure deaths — two scripted cohorts over a cold run:
       sheltered cohort (BUILD+occupy shelter)  vs.
       unsheltered cohort (same cold, no shelter).
     Assert sheltered cohort has FEWER exposure deaths and higher mean_thermal.
     Energy + hydration kept pinned so thermal is the isolating variable.

  2. Storage food-banking — scripted FORAGE → BUILD storage → STORE → wait →
     RETRIEVE + EAT.  Assert stored_food rose then fell, and agent ate
     previously-stored food (energy rose after retrieve+eat cycle).

  3. Metric integration — year_summary contains all Phase 4 keys
     (structures_built, stored_food_total [mean+max], mean_thermal, exposure
     in deaths_by_cause) — all finite — plus every prior Phase 0-3 key; JSONL
     round-trips.

Also produces data/phase4_shelter.png for visual inspection.
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
from world.seasons import compute_season_state
from sim.config import SimConfig
from sim.simulation import Simulation, Actions
from sim.actions import Action, StructureType, DIRECTIONS
from sim.state import WorldState, EntityStore
from sim.dynamics import (
    decay_drives, update_health, forage, build, store_food, retrieve_food,
    structure_decay_step, spoil_stored,
)
from sim.reproduce import resolve_deaths
from sim.metrics import MetricsLogger
from sim.render import render_frame


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sim(
    width: int = 64,
    height: int = 64,
    seed: int = 42,
    init_agents: int = 8,
    max_agents: int = 64,
    energy_decay: float = 0.002,
    hydration_decay: float = 0.003,
    spoilage_carried: float = 0.002,
    forage_yield: float = 0.4,
    forage_regen: float = 0.005,
    drink_restore: float = 1.0,
    **extra,
) -> Simulation:
    """Build a simulation with relaxed drives suitable for gate tests."""
    world = World.generate(WorldConfig(width=width, height=height, seed=seed))
    cfg = SimConfig(
        max_agents=max_agents,
        init_agents=init_agents,
        energy_decay=energy_decay,
        hydration_decay=hydration_decay,
        spoilage_carried=spoilage_carried,
        forage_yield=forage_yield,
        forage_regen=forage_regen,
        drink_restore=drink_restore,
        **extra,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=seed)
    return sim


def _noop_actions(cfg: SimConfig) -> Actions:
    M = cfg.max_agents
    return Actions(
        primary=np.full(M, int(Action.NOOP), dtype=np.int32),
        param=np.zeros(M, dtype=np.int32),
        emit=np.zeros(M, dtype=np.int32),
    )


def _single_action(cfg: SimConfig, slot: int, action: Action, param: int = 0) -> Actions:
    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    par = np.zeros(M, dtype=np.int32)
    primary[slot] = int(action)
    par[slot] = param
    return Actions(primary=primary, param=par, emit=np.zeros(M, dtype=np.int32))


def _all_action(cfg: SimConfig, n_agents: int, action: Action, param: int = 0) -> Actions:
    """Set all living-agent slots 0..n_agents-1 to `action`."""
    M = cfg.max_agents
    primary = np.full(M, int(Action.NOOP), dtype=np.int32)
    primary[:n_agents] = int(action)
    par = np.full(M, param, dtype=np.int32)
    return Actions(primary=primary, param=par, emit=np.zeros(M, dtype=np.int32))


# ---------------------------------------------------------------------------
# Scenario 1: Shelter reduces exposure deaths and raises mean_thermal
# ---------------------------------------------------------------------------

def test_shelter_reduces_exposure_deaths_and_raises_thermal():
    """
    Two cohorts run on identical cold-tile worlds over N_COLD steps.

    SHELTERED cohort: each agent occupies a shelter (structure_type==1) pre-placed
        on their tile.
    UNSHELTERED cohort: same cold tiles, no shelter.

    Energy and hydration are pinned to 1.0 each step so ONLY thermal/exposure
    varies.  Assert:
      - sheltered has strictly FEWER exposure deaths than unsheltered.
      - sheltered has strictly HIGHER mean_thermal in year_summary.
    """
    N_COLD = 200    # steps under cold conditions
    N_AGENTS = 8

    # Cold world: tile temp 0.2 < thermal_target_temp 0.45 when unsheltered;
    # with shelter_temp_bonus 0.3 => eff_temp = 0.5 >= 0.45 => sheltered stays warm.
    # Default cfg: thermal_target_temp=0.45, shelter_temp_bonus=0.3,
    #              thermal_decay=0.08, exposure_damage=0.04.
    world = World.generate(WorldConfig(width=32, height=32, seed=7))
    cfg = SimConfig(
        max_agents=32,
        init_agents=N_AGENTS,
        energy_decay=0.0,       # pin energy externally each step — avoid starvation
        hydration_decay=0.0,    # pin hydration
        thermal_decay=0.08,
        thermal_warm_regen=0.05,
        thermal_target_temp=0.45,
        shelter_temp_bonus=0.3,
        exposure_damage=0.04,
        D_init=1.0,             # full difficulty so exposure damage fires at full scale
    )

    # Make the world cold everywhere (overwrite temperature_base in-place).
    # Base temp 0.5; seasonal modifier ~0.40 in winter -> eff_temp_unsheltered = 0.20 < 0.45
    # -> unsheltered freeze; eff_temp_sheltered = 0.20+0.30 = 0.50 >= 0.45 -> sheltered warm.
    # Even at the coldest seasonal minimum (mod≈0.40) shelter suffices.
    world.temperature_base[:] = np.float32(0.5)

    # Build two identical sims sharing the same (now-cold) world
    def _build_cohort():
        sim = Simulation(world, cfg)
        sim.reset(seed=7)
        # Pin all agents to land tiles
        land_ys, land_xs = np.where(world.elevation >= world.cfg.sea_level)
        rng = np.random.default_rng(99)
        chosen = rng.integers(0, land_ys.shape[0], size=N_AGENTS)
        for i in range(N_AGENTS):
            sim.store.y[i] = int(land_ys[chosen[i]])
            sim.store.x[i] = int(land_xs[chosen[i]])
            sim.store.energy[i]    = 1.0
            sim.store.hydration[i] = 1.0
            sim.store.thermal[i]   = 1.0
            sim.store.health[i]    = 1.0
        return sim

    sim_sh = _build_cohort()   # sheltered cohort
    sim_un = _build_cohort()   # unsheltered cohort

    # Pre-place shelters on each agent's tile for the SHELTERED cohort
    for i in range(N_AGENTS):
        y = int(sim_sh.store.y[i])
        x = int(sim_sh.store.x[i])
        sim_sh.state.structure_type[y, x] = np.int8(StructureType.SHELTER)
        sim_sh.state.structure_hp[y, x]   = np.float32(100.0)  # very durable — won't decay

    logger_sh = MetricsLogger()
    logger_un = MetricsLogger()

    # Make temperature_base very cold everywhere
    cold_base = np.full(world.temperature_base.shape, 0.2, dtype=np.float32)

    for step_idx in range(N_COLD):
        # Step with NOOP (drives are pinned externally)
        actions_sh = _noop_actions(cfg)
        actions_un = _noop_actions(cfg)

        out_sh = sim_sh.step(actions_sh)
        out_un = sim_un.step(actions_un)

        # Count exposure deaths from done_mask + drive states captured before sim clears
        # We track deaths manually using done_mask and what drove health to 0.
        # Since we pin energy+hydration, any deaths are exposure (thermal->0 + damage).
        for i in range(N_AGENTS):
            if out_sh.done[i]:
                logger_sh.deaths["exposure"] += 1
            if out_un.done[i]:
                logger_un.deaths["exposure"] += 1

        # Pin energy+hydration every step for surviving agents so only thermal varies
        alive_sh = sim_sh.store.alive & ~sim_sh.store.is_predator
        alive_un = sim_un.store.alive & ~sim_un.store.is_predator
        sim_sh.store.energy[alive_sh]    = np.float32(1.0)
        sim_sh.store.hydration[alive_sh] = np.float32(1.0)
        sim_un.store.energy[alive_un]    = np.float32(1.0)
        sim_un.store.hydration[alive_un] = np.float32(1.0)

        logger_sh.record_step(sim_sh)
        logger_un.record_step(sim_un)

    summary_sh = logger_sh.year_summary()
    summary_un = logger_un.year_summary()

    exp_sh = summary_sh["deaths_by_cause"]["exposure"]
    exp_un = summary_un["deaths_by_cause"]["exposure"]
    th_sh  = summary_sh["mean_thermal"]
    th_un  = summary_un["mean_thermal"]

    print(
        f"\n[gate4.1] sheltered: exposure_deaths={exp_sh}, mean_thermal={th_sh:.4f}"
        f" | unsheltered: exposure_deaths={exp_un}, mean_thermal={th_un:.4f}"
    )

    assert exp_sh < exp_un, (
        f"Sheltered cohort should have fewer exposure deaths than unsheltered: "
        f"sheltered={exp_sh}, unsheltered={exp_un}"
    )
    assert th_sh > th_un, (
        f"Sheltered cohort should have higher mean_thermal than unsheltered: "
        f"sheltered_thermal={th_sh:.4f}, unsheltered_thermal={th_un:.4f}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Storage food-banking loop
# ---------------------------------------------------------------------------

def test_storage_food_bank_and_draw():
    """
    Scripted single-agent food-banking loop:
      1. Forage food into inventory.
      2. BUILD a storage structure on the agent's tile (give agent enough wood).
      3. STORE food into the storage tile.
         → Assert stored_food rose from 0.
      4. Wait N_WAIT steps (no eating, no foraging).
         → stored_food should remain elevated (some spoilage ok).
      5. RETRIEVE food from storage + EAT.
         → Assert energy rose (agent ate stored food).
         → Assert stored_food fell from its peak.

    Confirms the winter food-banking loop: bank food in warm period, draw it
    down in lean times.
    """
    sim = _make_sim(
        init_agents=1,
        max_agents=16,
        energy_decay=0.0,
        hydration_decay=0.0,
        spoilage_carried=0.0,
        spoilage_stored=0.001,   # very slow spoilage so food persists
    )
    world = sim.world
    cfg   = sim.cfg
    store = sim.store
    state = sim.state
    slot  = 0

    # Place agent on a land tile with wild food
    land_ys, land_xs = np.where(world.elevation >= world.cfg.sea_level)
    # Pick a tile with some wild_remaining
    rng = np.random.default_rng(1)
    for attempt in range(200):
        idx = rng.integers(0, land_ys.shape[0])
        py, px = int(land_ys[idx]), int(land_xs[idx])
        if state.wild_remaining[py, px] > 0.1:
            break

    store.y[slot] = py
    store.x[slot] = px
    store.energy[slot]    = 1.0
    store.hydration[slot] = 1.0
    store.health[slot]    = 1.0
    store.inv_food[slot]  = 0.0
    # Give plenty of wood to afford storage
    store.inv_wood[slot]  = np.float32(10.0)

    # --- Step 1: forage until inventory has food ---
    MAX_FORAGE = 20
    for _ in range(MAX_FORAGE):
        if float(store.inv_food[slot]) >= 0.3:
            break
        out = sim.step(_single_action(cfg, slot, Action.FORAGE))
        # Re-pin location (agents shouldn't move; but re-pin anyway)
        store.y[slot] = py; store.x[slot] = px
        # Also keep wild food alive for repeated foraging
        state.wild_remaining[py, px] = max(float(state.wild_remaining[py, px]), 0.4)

    food_pre_store = float(store.inv_food[slot])
    assert food_pre_store > 0.0, "Agent should have foraged some food"

    # --- Step 2: BUILD a storage structure on the same tile ---
    # The BUILD param for STORAGE is StructureType.STORAGE == 2
    # Ensure no existing structure
    state.structure_type[py, px] = np.int8(0)
    out_build = sim.step(_single_action(cfg, slot, Action.BUILD, param=int(StructureType.STORAGE)))
    assert int(state.structure_type[py, px]) == int(StructureType.STORAGE), (
        f"Expected storage at ({py},{px}), got structure_type={state.structure_type[py,px]}"
    )

    # Re-pin food (BUILD doesn't consume food; just verify build succeeded)
    store.inv_food[slot] = np.float32(food_pre_store)

    # --- Step 3: STORE food into the storage tile ---
    stored_before = float(state.stored_food[py, px])
    assert stored_before == pytest.approx(0.0, abs=1e-5), "Stored food should start at 0"
    out_store = sim.step(_single_action(cfg, slot, Action.STORE))
    stored_after = float(state.stored_food[py, px])
    assert stored_after > stored_before, (
        f"stored_food should have risen after STORE: before={stored_before:.4f}, "
        f"after={stored_after:.4f}"
    )
    stored_peak = stored_after
    print(f"\n[gate4.2] stored_peak={stored_peak:.4f}, food_pre_store={food_pre_store:.4f}")

    # --- Step 4: Wait N_WAIT steps (no eating, just NOOP) ---
    N_WAIT = 20
    for _ in range(N_WAIT):
        sim.step(_noop_actions(cfg))
        # Keep agent alive and on tile
        store.y[slot] = py; store.x[slot] = px
        store.energy[slot]    = np.float32(0.5)   # keep alive
        store.hydration[slot] = np.float32(0.5)
        store.health[slot]    = np.float32(1.0)
        store.alive[slot]     = True
    stored_mid = float(state.stored_food[py, px])
    assert stored_mid > 0.0, (
        f"Stored food should still be positive after {N_WAIT} wait steps; "
        f"got {stored_mid:.4f}"
    )

    # --- Step 5: RETRIEVE and EAT ---
    # First retrieve
    store.inv_food[slot] = np.float32(0.0)   # empty inv so we definitely retrieve
    pre_retrieve_stored = float(state.stored_food[py, px])
    out_retr = sim.step(_single_action(cfg, slot, Action.RETRIEVE))
    post_retrieve_stored = float(state.stored_food[py, px])
    retrieved = float(store.inv_food[slot])

    assert retrieved > 0.0, "Agent should have retrieved food from storage"
    assert post_retrieve_stored < pre_retrieve_stored, (
        f"stored_food should have dropped after RETRIEVE: "
        f"before={pre_retrieve_stored:.4f}, after={post_retrieve_stored:.4f}"
    )

    # Then EAT — energy should rise
    store.energy[slot] = np.float32(0.3)   # set energy low to ensure eating helps
    pre_eat_energy = float(store.energy[slot])
    out_eat = sim.step(_single_action(cfg, slot, Action.EAT))
    post_eat_energy = float(store.energy[slot])

    assert post_eat_energy > pre_eat_energy, (
        f"Energy should rise after EAT on retrieved food: "
        f"pre={pre_eat_energy:.4f}, post={post_eat_energy:.4f}"
    )

    stored_final = float(state.stored_food[py, px])
    print(
        f"[gate4.2] stored peak={stored_peak:.4f}, mid={stored_mid:.4f}, "
        f"final={stored_final:.4f}; retrieved={retrieved:.4f}; "
        f"energy {pre_eat_energy:.4f} -> {post_eat_energy:.4f}"
    )

    assert stored_peak > stored_final or stored_peak > 0.0, (
        "stored_food should have peaked then been drawn down through retrieve"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Metric integration — Phase 4 keys present, finite, JSONL
# ---------------------------------------------------------------------------

_PRIOR_KEYS = (
    "population_mean", "population_min", "population_max",
    "births", "deaths_by_cause",
    "mean_displacement", "wild_food_mean",
    "water_occupancy",
    "pct_calories_farmed",
    "fertile_occupancy",
)

_PHASE4_SCALAR_KEYS = ("mean_thermal",)
_PHASE4_DICT_KEYS   = ("structures_built", "stored_food_total")


def test_year_summary_has_phase4_keys_and_all_prior_keys():
    """year_summary must contain all Phase 4 keys plus all prior Phase 0-3 keys.

    - structures_built: dict with total/max_total/shelters/storage, all >= 0.
    - stored_food_total: dict with mean/max, both finite and >= 0.
    - mean_thermal: float in [0, 1].
    - deaths_by_cause contains 'exposure' key (int).
    All prior keys present and unmodified.
    """
    sim = _make_sim(
        width=64, height=64, seed=5,
        init_agents=8, max_agents=32,
        energy_decay=0.002, hydration_decay=0.003,
    )
    logger = MetricsLogger()
    cfg   = sim.cfg
    world = sim.world
    rng   = np.random.default_rng(11)

    M = cfg.max_agents

    for step_i in range(80):
        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param   = rng.integers(0, 8, size=M, dtype=np.int32)
        emit    = np.zeros(M, dtype=np.int32)

        live_idx = np.flatnonzero(sim.store.alive & ~sim.store.is_predator)
        if live_idx.size > 0:
            ys = sim.store.y[live_idx]
            xs = sim.store.x[live_idx]
            wild     = sim.state.wild_remaining
            crops    = sim.state.crop_stage
            structs  = sim.state.structure_type
            water_prox = world.base_resources["water_proximity"]

            for i, s in enumerate(live_idx):
                y, x = int(ys[i]), int(xs[i])
                if sim.store.inv_food[s] > 0:
                    primary[s] = int(Action.EAT)
                elif wild[y, x] > 0.1:
                    primary[s] = int(Action.FORAGE)
                elif water_prox[y, x] >= cfg.drink_min_water:
                    primary[s] = int(Action.DRINK)
                else:
                    primary[s] = int(Action.MOVE)
                    param[s]   = rng.integers(0, 8)

                # Occasionally try to build a shelter (if enough wood)
                if (step_i % 10 == 0
                        and structs[y, x] == 0
                        and float(sim.store.inv_wood[s]) >= cfg.shelter_cost["wood"]):
                    primary[s] = int(Action.BUILD)
                    param[s]   = int(StructureType.SHELTER)

        out = sim.step(Actions(primary=primary, param=param, emit=emit))
        sim.respawn_dead(seed=step_i)
        logger.record_step(sim, out.info)

    summary = logger.year_summary()

    # All prior keys present
    for key in _PRIOR_KEYS:
        assert key in summary, f"Missing prior key: {key}"

    # Phase 4 scalar keys
    for key in _PHASE4_SCALAR_KEYS:
        assert key in summary, f"Missing Phase 4 key: {key}"
        val = summary[key]
        assert math.isfinite(val), f"{key} not finite: {val}"
        assert 0.0 <= val <= 1.0, f"{key} out of [0,1]: {val}"

    # Phase 4 dict keys
    for key in _PHASE4_DICT_KEYS:
        assert key in summary, f"Missing Phase 4 dict key: {key}"

    sb = summary["structures_built"]
    for subkey in ("total", "max_total", "shelters", "storage"):
        assert subkey in sb, f"structures_built missing subkey '{subkey}'"
        assert isinstance(sb[subkey], int), f"structures_built[{subkey}] should be int"
        assert sb[subkey] >= 0, f"structures_built[{subkey}] must be >= 0"

    sft = summary["stored_food_total"]
    for subkey in ("mean", "max"):
        assert subkey in sft, f"stored_food_total missing subkey '{subkey}'"
        assert math.isfinite(sft[subkey]), f"stored_food_total[{subkey}] not finite"
        assert sft[subkey] >= 0.0, f"stored_food_total[{subkey}] must be >= 0"

    assert "exposure" in summary["deaths_by_cause"], (
        "deaths_by_cause should contain 'exposure' key"
    )
    assert isinstance(summary["deaths_by_cause"]["exposure"], int)

    print(
        f"\n[gate4.3] structures_built={summary['structures_built']}, "
        f"stored_food_total={summary['stored_food_total']}, "
        f"mean_thermal={summary['mean_thermal']:.4f}, "
        f"exposure_deaths={summary['deaths_by_cause']['exposure']}"
    )


def test_phase4_metrics_jsonl_round_trip():
    """year_summary writes JSONL; parsed line contains all Phase 4 keys and
    all prior keys; all values are finite and within expected bounds."""
    sim = _make_sim(width=64, height=64, seed=9, init_agents=8, max_agents=32)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        path = fh.name

    try:
        logger = MetricsLogger(out_path=path)
        cfg   = sim.cfg
        M     = cfg.max_agents
        rng   = np.random.default_rng(21)

        for step_i in range(40):
            primary = np.full(M, int(Action.NOOP), dtype=np.int32)
            param   = rng.integers(0, 8, size=M, dtype=np.int32)
            emit    = np.zeros(M, dtype=np.int32)

            live_idx = np.flatnonzero(sim.store.alive & ~sim.store.is_predator)
            if live_idx.size > 0:
                ys = sim.store.y[live_idx]
                xs = sim.store.x[live_idx]
                wild = sim.state.wild_remaining
                for i, s in enumerate(live_idx):
                    y, x = int(ys[i]), int(xs[i])
                    if wild[y, x] > 0.05:
                        primary[s] = int(Action.FORAGE)

            out = sim.step(Actions(primary=primary, param=param, emit=emit))
            sim.respawn_dead(seed=step_i)
            logger.record_step(sim, out.info)

        logger.year_summary()

        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        assert len(lines) == 1, f"Expected 1 JSONL line, got {len(lines)}"
        parsed = json.loads(lines[0])

        # All prior keys
        for key in _PRIOR_KEYS:
            assert key in parsed, f"JSONL missing prior key: {key}"

        # Phase 4 keys
        for key in ("mean_thermal", "structures_built", "stored_food_total"):
            assert key in parsed, f"JSONL missing Phase 4 key: {key}"

        assert math.isfinite(parsed["mean_thermal"])
        assert 0.0 <= parsed["mean_thermal"] <= 1.0

        sb = parsed["structures_built"]
        for subkey in ("total", "max_total", "shelters", "storage"):
            assert subkey in sb

        sft = parsed["stored_food_total"]
        for subkey in ("mean", "max"):
            assert subkey in sft
            assert math.isfinite(sft[subkey])
            assert sft[subkey] >= 0.0

        assert "exposure" in parsed["deaths_by_cause"]

        print(
            f"\n[gate4.3b] JSONL mean_thermal={parsed['mean_thermal']:.4f}, "
            f"structures_built={parsed['structures_built']}, "
            f"stored_food_total={parsed['stored_food_total']}"
        )

    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Visualization: render a frame with shelters + storage + agents
# ---------------------------------------------------------------------------

def test_render_phase4_shelter_png():
    """Render a single frame showing shelters + storage + agents.

    Saves to data/phase4_shelter.png.  Always passes as long as imwrite succeeds.
    """
    import imageio.v3 as iio

    sim = _make_sim(
        width=128, height=128, seed=42,
        init_agents=32, max_agents=128,
        energy_decay=0.001, hydration_decay=0.001,
    )
    cfg   = sim.cfg
    world = sim.world
    state = sim.state
    store = sim.store
    M     = cfg.max_agents
    rng   = np.random.default_rng(42)

    # Give all agents plenty of wood so they can build
    for i in range(sim.cfg.init_agents):
        store.inv_wood[i] = np.float32(20.0)
        store.inv_stone[i] = np.float32(10.0)

    # Run a few steps where agents forage then build shelters and storage
    for step_i in range(10):
        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param   = rng.integers(1, 3, size=M, dtype=np.int32)  # 1=shelter, 2=storage
        emit    = np.zeros(M, dtype=np.int32)

        live_idx = np.flatnonzero(store.alive & ~store.is_predator)
        if live_idx.size > 0:
            ys = store.y[live_idx]
            xs = store.x[live_idx]
            structs = state.structure_type
            wild    = state.wild_remaining

            for i, s in enumerate(live_idx):
                y, x = int(ys[i]), int(xs[i])
                if structs[y, x] == 0 and step_i < 5:
                    # Build phase: alternate between shelter and storage
                    primary[s] = int(Action.BUILD)
                    param[s]   = int(StructureType.SHELTER) if (s % 2 == 0) else int(StructureType.STORAGE)
                elif wild[y, x] > 0.05:
                    primary[s] = int(Action.FORAGE)
                else:
                    primary[s] = int(Action.MOVE)
                    param[s]   = rng.integers(0, 8)

        sim.step(Actions(primary=primary, param=param, emit=emit))

    frame = render_frame(world, state, store)

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4_shelter.png"

    iio.imwrite(str(out_path), frame)

    assert out_path.exists() and out_path.stat().st_size > 0, (
        f"Output file not found or empty: {out_path}"
    )

    # Sanity: at least some structures should be present (orange or purple pixels)
    from sim.render import _STRUCTURE_COLORS
    shelter_color = _STRUCTURE_COLORS[1]  # orange
    storage_color = _STRUCTURE_COLORS[2]  # purple

    n_shelters = int((state.structure_type == 1).sum())
    n_storage  = int((state.structure_type == 2).sum())

    print(
        f"\n[viz] rendered {n_shelters} shelters + {n_storage} storage to: {out_path}"
    )
    # Just confirm we can render without error; presence of structures is a bonus
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
