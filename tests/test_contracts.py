"""
Contract tests — pin the FROZEN §1 interfaces so the fleet cannot silently drift them.

These test the schema/shape/layout contracts only (no phased logic). Unimplemented
phased bodies are asserted to raise NotImplementedError, documenting what is not yet
built. If any of these change, it means a contract changed -> requires a re-lock.
"""
import numpy as np
import pytest

from sim.config import SimConfig
from sim.state import WorldState, EntityStore
from sim import actions, observe
from sim.actions import Action, StructureType, DIRECTIONS, N_PRIMARY, N_PARAM
from sim.curriculum import Curriculum
from sim.simulation import Simulation, Obs, Actions, StepOut


# ---------------- config ----------------

def test_simconfig_defaults_and_invariants():
    c = SimConfig()
    assert c.max_agents == 4096
    assert c.window_radius == 7
    assert c.window_size == 15
    assert c.genome_dim == 6
    assert c.n_symbols == 8
    # evolution-only reward: only comfort + alive weights exist
    assert c.w_h == 1.0 and c.w_a == 0.05
    # cost maps are dicts keyed by resource name
    assert c.wall_cost == {"stone": 3, "wood": 1}
    assert c.weapon_cost["minerals"] == 1


def test_simconfig_rejects_bad_values():
    with pytest.raises(AssertionError):
        SimConfig(init_agents=999999)  # > max_agents


# ---------------- world state ----------------

def test_worldstate_shapes_and_dtypes():
    ws = WorldState.create(64, 48)
    assert ws.shape == (64, 48)
    assert ws.wild_remaining.dtype == np.float32
    assert ws.crop_owner.dtype == np.int32 and (ws.crop_owner == -1).all()
    assert ws.structure_type.dtype == np.int8
    assert (ws.event_mask == 1.0).all()  # neutral multiplier


# ---------------- entity store ----------------

def test_entitystore_shapes_and_neutrals():
    c = SimConfig(max_agents=32, init_agents=8, genome_dim=6)
    es = EntityStore.create(c)
    assert es.capacity == 32
    assert es.alive.dtype == bool and not es.alive.any()
    assert es.genome.shape == (32, 6)
    assert np.allclose(es.genome, 0.5)          # neutral body
    assert (es.lineage_id == -1).all()
    assert es.n_living_agents() == 0
    assert len(es.free_slots()) == 32


def test_entitystore_living_excludes_predators():
    c = SimConfig(max_agents=8, init_agents=4)
    es = EntityStore.create(c)
    es.alive[:4] = True
    es.is_predator[2:4] = True
    assert es.n_living_agents() == 2          # slots 0,1
    assert es.living_agents_mask().sum() == 2


# ---------------- action space ----------------

def test_action_space_layout():
    assert N_PRIMARY == 14
    assert Action.REPRODUCE == 13
    assert N_PARAM == 8
    assert DIRECTIONS.shape == (8, 2)
    assert StructureType.WALL == 3


def test_build_mask_phase1_implemented():
    """Phase 1: build_mask is now implemented; calling with None world/state/store
    will raise an AttributeError (not NotImplementedError) since the body
    accesses store.alive etc.  Verify the stub is gone."""
    c = SimConfig()
    # build_mask no longer raises NotImplementedError — it is a live implementation.
    # Calling with all-None args will AttributeError at store.alive; that is fine.
    try:
        actions.build_mask(None, None, None, c)
    except NotImplementedError:
        raise AssertionError("build_mask must be implemented for Phase 1 — NotImplementedError should not be raised")
    except Exception:
        pass  # AttributeError or similar is expected with None inputs


# ---------------- observation layout ----------------

def test_observation_layout():
    assert observe.N_SPATIAL == 24
    assert observe.WORLD_OBS_CHANNELS == 14
    c = SimConfig(genome_dim=6)
    assert observe.vector_len(c) == 16        # 10 base + 6 genome
    # first 14 spatial channels mirror world.get_obs
    assert observe.SPATIAL_CHANNELS[13] == "w_season_phase"
    assert observe.SPATIAL_CHANNELS[23] == "nearby_signal"


# ---------------- curriculum (implemented) ----------------

def test_curriculum_ramps_when_stable():
    c = SimConfig(D_init=0.3, D_step=0.1, stable_years_to_ramp=3)
    cur = Curriculum(c)
    assert cur.D == 0.3
    for _ in range(3):
        cur.on_year_end(population=100, collapsed=False)
    assert cur.D == pytest.approx(0.4)        # ramped once after 3 stable years


def test_curriculum_backs_off_on_collapse():
    c = SimConfig(D_init=0.5, D_step=0.1)
    cur = Curriculum(c)
    cur.on_year_end(population=0, collapsed=True)
    assert cur.D == pytest.approx(0.4)


# ---------------- simulation facade ----------------

def test_simulation_constructs_on_small_world():
    from world import World, WorldConfig
    # tiny world for a fast contract test
    w = World.generate(WorldConfig(width=64, height=64, seed=1))
    c = SimConfig(max_agents=16, init_agents=8)
    sim = Simulation(w, c)
    assert sim.H == 64 and sim.W == 64
    assert sim.store.capacity == 16
    assert sim.state.shape == (64, 64)
    assert isinstance(sim.alive_mask, np.ndarray) and sim.alive_mask.shape == (16,)


def test_simulation_phased_methods_are_stubs():
    from world import World, WorldConfig
    w = World.generate(WorldConfig(width=64, height=64, seed=1))
    sim = Simulation(w, SimConfig(max_agents=16, init_agents=8))
    # Phase 0 is implemented: reset() and observe() (via build_observation) work.
    obs = sim.reset()
    assert isinstance(obs, Obs)
    obs2 = sim.observe()
    assert isinstance(obs2, Obs)
    # Phase 1: build_mask is now implemented — it should NOT raise NotImplementedError.
    from sim import actions
    mask = actions.build_mask(w, sim.state, sim.store, sim.cfg)
    M = sim.cfg.max_agents
    from sim.actions import N_PRIMARY
    assert mask.shape == (M, N_PRIMARY)


def test_dataclass_contracts_exist():
    # Obs / Actions / StepOut are the frozen learner-facing shapes
    assert {f for f in Obs.__dataclass_fields__} == {"alive_mask", "spatial", "vector"}
    assert {f for f in Actions.__dataclass_fields__} == {"primary", "param", "emit"}
    assert {f for f in StepOut.__dataclass_fields__} == {"obs", "reward", "done", "info"}
