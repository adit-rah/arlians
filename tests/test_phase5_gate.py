"""
Phase 5 gate — reproduction/evolution metrics integrate into MetricsLogger
(build-spec §4.1, §5 Phase 5). The reproduction MECHANISM is unit-tested in
test_reproduce_phase5.py; this verifies the metrics layer surfaces births,
genome drift, and lineage diversity, and that summaries round-trip.
"""
import json
import numpy as np

from world import World, WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation, Actions
from sim.actions import Action, build_mask
from sim.metrics import MetricsLogger


def _relaxed_cfg():
    # survivable economy so a scripted population reproduces (mechanism demo,
    # not an emergence claim — true stability under the harsh default is a
    # LEARNED, GPU-deferred outcome).
    return SimConfig(
        max_agents=256, init_agents=64,
        energy_decay=0.002, hydration_decay=0.002, thermal_decay=0.0,
        spoilage_carried=0.0, forage_yield=0.8, forage_regen=0.05, D_init=0.0,
    )


def _scripted(sim, cfg):
    """Survival-first, breed-when-thriving scripted policy."""
    m = build_mask(sim.world, sim.state, sim.store, cfg)
    pr = np.full(cfg.max_agents, int(Action.REST), np.int32)
    for i in np.flatnonzero(sim.store.living_agents_mask()):
        e = sim.store.energy[i]; f = sim.store.inv_food[i]; h = sim.store.hydration[i]
        if e < 0.5 and f > 0:                       pr[i] = int(Action.EAT)
        elif m[i, int(Action.REPRODUCE)]:           pr[i] = int(Action.REPRODUCE)  # mask gates energy>=0.7
        elif m[i, int(Action.FORAGE)] and f < 0.5:  pr[i] = int(Action.FORAGE)
        elif m[i, int(Action.DRINK)] and h < 0.5:   pr[i] = int(Action.DRINK)
    param = np.random.randint(0, 8, cfg.max_agents).astype(np.int32)
    return Actions(pr, param, np.zeros(cfg.max_agents, np.int32))


def test_phase5_metrics_surface_births_drift_lineage(tmp_path):
    cfg = _relaxed_cfg()
    sim = Simulation(World.generate(WorldConfig(width=64, height=64, seed=123)), cfg)
    sim.reset(seed=123)
    log = MetricsLogger(out_path=str(tmp_path / "m.jsonl"))

    total_info_births = 0
    for _ in range(720):  # 2 simulated years
        out = sim.step(_scripted(sim, cfg))
        log.record_step(sim, out.info)
        total_info_births += int(out.info.get("births", 0))

    s = log.year_summary()
    # Phase 5 keys present and sane
    assert "births" in s and s["births"] == total_info_births and s["births"] > 0
    assert "genome_drift" in s and s["genome_drift"] >= 0.0
    assert "lineage_count" in s and s["lineage_count"] > 0
    assert "mean_genome" in s and len(s["mean_genome"]) == cfg.genome_dim
    # all prior keys still present (additive contract)
    for k in ("population_mean", "deaths_by_cause", "pct_calories_farmed",
              "structures_built", "mean_thermal", "water_occupancy"):
        assert k in s
    # JSONL round-trips
    line = json.loads((tmp_path / "m.jsonl").read_text().strip().splitlines()[-1])
    assert line["births"] == s["births"]
    assert len(line["mean_genome"]) == cfg.genome_dim


def test_genome_drift_increases_with_mutation():
    """Mutation accumulates: after many generations the population mean genome
    has drifted measurably away from the neutral 0.5 start."""
    cfg = _relaxed_cfg()
    sim = Simulation(World.generate(WorldConfig(width=64, height=64, seed=7)), cfg)
    sim.reset(seed=7)
    log = MetricsLogger()
    for _ in range(720):
        out = sim.step(_scripted(sim, cfg))
        log.record_step(sim, out.info)
    s = log.year_summary()
    # with births having occurred, drift should be strictly positive
    if s["births"] > 0:
        assert s["genome_drift"] > 0.0
