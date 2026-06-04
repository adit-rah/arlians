"""
Phase 7 tests — episodic catastrophes + signaling activation (build-spec §2.7,
§1.4-1.5, §5 Phase 7). Authored at integration time (the Phase-7 env agent's
session dropped before writing tests; orchestrator validates here).
"""
import numpy as np
import pytest

from world import World, WorldConfig
from sim.config import SimConfig
from sim.state import EntityStore, WorldState
from sim.actions import Action, StructureType, build_mask
from sim.simulation import Simulation, Actions
from sim.threats import roll_catastrophe, apply_catastrophes
from sim.observe import SPATIAL_CHANNELS


def _world(w=48, h=48, seed=5):
    return World.generate(WorldConfig(width=w, height=h, seed=seed))


def _event(world, cfg, etype, t=0, cy=None, cx=None):
    land = np.argwhere(world.elevation >= world.cfg.sea_level)
    if cy is None:
        cy, cx = int(land[0][0]), int(land[0][1])
    return {"type": etype, "cy": cy, "cx": cx, "radius": cfg.catastrophe_radius,
            "magnitude": cfg.catastrophe_magnitude, "expiry": t + cfg.catastrophe_duration}


# ---- roll / event lifecycle ----

def test_roll_creates_event_with_high_prob():
    w = _world(); cfg = SimConfig(catastrophe_prob=1.0)  # force
    rng = np.random.default_rng(0)
    events = roll_catastrophe([], w, t=0, cfg=cfg, rng=rng)
    assert len(events) == 1
    e = events[0]
    assert set(e) >= {"type", "cy", "cx", "radius", "magnitude", "expiry"}
    assert e["type"] in ("cold_snap", "drought", "flood", "storm")


def test_event_mask_inside_low_outside_high_then_expires():
    w = _world(); cfg = SimConfig()
    state = WorldState.create(*w.elevation.shape)
    store = EntityStore.create(cfg)
    ev = _event(w, cfg, "drought", t=0, cy=24, cx=24)
    surviving = apply_catastrophes(state, store, w, [ev], t=0, cfg=cfg)
    assert len(surviving) == 1
    assert state.event_mask[24, 24] == pytest.approx(cfg.catastrophe_magnitude)
    assert state.event_mask[0, 0] == pytest.approx(1.0)  # far outside region
    # after expiry, event dropped and mask restored
    surviving = apply_catastrophes(state, store, w, surviving, t=cfg.catastrophe_duration, cfg=cfg)
    assert surviving == []
    assert (state.event_mask == 1.0).all()


# ---- per-type effects ----

def test_cold_snap_damages_agents_in_region():
    w = _world(); cfg = SimConfig(max_agents=8, init_agents=2)
    state = WorldState.create(*w.elevation.shape)
    store = EntityStore.create(cfg)
    store.alive[:2] = True; store.health[:2] = 1.0
    store.y[0], store.x[0] = 24, 24     # inside
    store.y[1], store.x[1] = 0, 0       # outside (region around 24,24 radius 20 -> 4..44)
    apply_catastrophes(state, store, w, [_event(w, cfg, "cold_snap", cy=24, cx=24)], t=0, cfg=cfg)
    assert store.health[0] < 1.0        # damaged inside
    assert store.health[1] == pytest.approx(1.0)  # safe outside


def test_storm_damages_structures_in_region():
    w = _world(); cfg = SimConfig()
    state = WorldState.create(*w.elevation.shape)
    store = EntityStore.create(cfg)
    state.structure_type[24, 24] = StructureType.SHELTER
    state.structure_hp[24, 24] = 1.0
    apply_catastrophes(state, store, w, [_event(w, cfg, "storm", cy=24, cx=24)], t=0, cfg=cfg)
    assert state.structure_hp[24, 24] < 1.0


def test_drought_suppresses_wild_food_in_region():
    w = _world(); cfg = SimConfig()
    state = WorldState.create(*w.elevation.shape)
    store = EntityStore.create(cfg)
    state.wild_remaining[24, 24] = 1.0
    apply_catastrophes(state, store, w, [_event(w, cfg, "drought", cy=24, cx=24)], t=0, cfg=cfg)
    assert state.wild_remaining[24, 24] < 1.0


# ---- signaling activation ----

def test_emit_writes_last_signal_and_shows_in_obs():
    w = _world(); cfg = SimConfig(max_agents=16, init_agents=4, catastrophe_prob=0.0,
                                  pred_per_agents=0.0, n_symbols=8)
    sim = Simulation(w, cfg); sim.reset(seed=0)
    live = np.flatnonzero(sim.store.living_agents_mask())
    sym = 5
    emit = np.zeros(cfg.max_agents, np.int32)
    emit[live] = sym
    out = sim.step(Actions(primary=np.zeros(cfg.max_agents, np.int32),
                           param=np.zeros(cfg.max_agents, np.int32), emit=emit))
    # last_signal recorded
    assert int(sim.store.last_signal[live[0]]) == sym
    # nearby-signal obs channel (index 23) is nonzero in the emitter's own window
    ch = SPATIAL_CHANNELS.index("nearby_signal")
    win = out.obs.spatial[live[0], ch]
    assert win.max() > 0.0


# ---- integrated ----

def test_integrated_catastrophes_run():
    w = _world(); cfg = SimConfig(max_agents=256, init_agents=128, catastrophe_prob=0.3,
                                  energy_decay=0.002, hydration_decay=0.002, thermal_decay=0.0)
    sim = Simulation(w, cfg); sim.reset(seed=1)
    seen = 0
    for _ in range(100):
        m = build_mask(w, sim.state, sim.store, cfg)
        pr = np.zeros(cfg.max_agents, np.int32)
        for i in np.flatnonzero(sim.store.living_agents_mask()):
            va = np.flatnonzero(m[i]); va = va[va != int(Action.REPRODUCE)]
            pr[i] = np.random.choice(va) if va.size else 0
        out = sim.step(Actions(pr, np.random.randint(0, 8, cfg.max_agents).astype(np.int32),
                               np.random.randint(0, cfg.n_symbols, cfg.max_agents).astype(np.int32)))
        sim.respawn_dead(seed=1)
        seen = max(seen, out.info.get("n_catastrophes", 0))
    assert seen >= 1  # at least one catastrophe occurred
