"""
Reproduction, death, and genome evolution (build-spec §2.8, §2.5 trait map).

Filled by the env-dynamics fleet role:
  Phase 1: death resolution (health <= 0 -> free the slot, mark done)
  Phase 5: REPRODUCE action (energy-gated, costed, cooldown), child placement in a
           nearby free slot, genome inheritance + gaussian mutation, lineage_id.
  traits_from_genome(): map genome[M,G] -> body traits (metabolism, water_need,
           cold_tol, capacity). Used by dynamics; selection is implicit via survival.
"""
from __future__ import annotations

from typing import Optional, NamedTuple
import numpy as np

from .config import SimConfig
from .state import EntityStore


class BodyTraits(NamedTuple):
    """Per-agent body traits derived from genome. All arrays shape (M,) float32."""
    metab: np.ndarray       # metabolism multiplier on energy_decay; neutral=1.0
    water_need: np.ndarray  # hydration decay multiplier; neutral=1.0
    cold_tol: np.ndarray    # thermal offset vs. thermal_target_temp; neutral=0.0
    capacity: np.ndarray    # carry-capacity multiplier on cfg.carry_capacity; neutral=1.0


def traits_from_genome(genome: np.ndarray) -> BodyTraits:
    """
    Map genome (M, G) in [0,1] -> body traits for all M slots.

    Trait encoding (build-spec §2.5):
      g0 -> metab      : lerp(0.5, 1.5, g0)   neutral g0=0.5 -> 1.0
      g1 -> water_need : lerp(0.5, 1.5, g1)   neutral g1=0.5 -> 1.0
      g2 -> cold_tol   : lerp(-0.2, 0.2, g2)  neutral g2=0.5 -> 0.0
      g3 -> capacity   : lerp(0.7, 1.3, g3)   neutral g3=0.5 -> 1.0

    Genes beyond index 3 are reserved for future phases.
    Returns BodyTraits of (M,) float32 arrays.
    """
    G = genome.shape[1]

    def _lerp(lo: float, hi: float, g: np.ndarray) -> np.ndarray:
        return (lo + (hi - lo) * g).astype(np.float32)

    g0 = genome[:, 0] if G > 0 else np.full(genome.shape[0], 0.5, dtype=np.float32)
    g1 = genome[:, 1] if G > 1 else np.full(genome.shape[0], 0.5, dtype=np.float32)
    g2 = genome[:, 2] if G > 2 else np.full(genome.shape[0], 0.5, dtype=np.float32)
    g3 = genome[:, 3] if G > 3 else np.full(genome.shape[0], 0.5, dtype=np.float32)

    return BodyTraits(
        metab=_lerp(0.5, 1.5, g0),
        water_need=_lerp(0.5, 1.5, g1),
        cold_tol=_lerp(-0.2, 0.2, g2),
        capacity=_lerp(0.7, 1.3, g3),
    )


def resolve_deaths(
    store: EntityStore,
    metrics_deaths: Optional[dict] = None,
) -> np.ndarray:
    """
    Kill any living agent whose health <= 0 and mark their slot as dead.

    Steps:
      1. Find slots where alive=True and health <= 0.
      2. Attribute cause of death (BEFORE zeroing drives):
           hydration <= 0 -> "dehydrate"
           energy <= 0 (and hydration > 0) -> "starve"
           thermal <= 0 (and energy > 0 and hydration > 0) -> "exposure"
           else -> "starve"  (health drained by other means)
      3. Zero / reset all per-slot fields so the freed slot is clean for reuse.
      4. Set alive=False (the key flag).

    Returns a boolean (M,) mask: True for slots that died THIS step.
    The mask is also used as the `done` signal for the step output.

    If `metrics_deaths` is a dict it receives increments to "starve", "dehydrate",
    and/or "exposure" matching each dead agent's attributed cause.
    """
    died_mask: np.ndarray = store.alive & (store.health <= 0.0)

    if not died_mask.any():
        return died_mask

    idx = np.flatnonzero(died_mask)

    # --- attribute cause of death BEFORE zeroing drives ---
    if metrics_deaths is not None:
        # Priority: hydration <= 0 -> dehydrate
        #           else energy <= 0 -> starve
        #           else thermal <= 0 -> exposure
        #           else -> starve (generic health drain)
        dehydrate_mask = store.hydration[idx] <= 0.0
        starve_mask    = (~dehydrate_mask) & (store.energy[idx] <= 0.0)
        exposure_mask  = (~dehydrate_mask) & (~starve_mask) & (store.thermal[idx] <= 0.0)
        other_mask     = (~dehydrate_mask) & (~starve_mask) & (~exposure_mask)

        n_dehydrate = int(dehydrate_mask.sum())
        n_starve    = int(starve_mask.sum()) + int(other_mask.sum())
        n_exposure  = int(exposure_mask.sum())

        metrics_deaths["dehydrate"] = metrics_deaths.get("dehydrate", 0) + n_dehydrate
        metrics_deaths["starve"]    = metrics_deaths.get("starve",    0) + n_starve
        metrics_deaths["exposure"]  = metrics_deaths.get("exposure",  0) + n_exposure

    # --- clear all per-slot fields to neutral/zero so the slot is ready for reuse ---
    store.alive[idx]        = False
    store.is_predator[idx]  = False
    store.y[idx]            = 0
    store.x[idx]            = 0
    store.energy[idx]       = 0.0
    store.hydration[idx]    = 0.0
    store.thermal[idx]      = 0.0
    store.health[idx]       = 0.0
    store.age[idx]          = 0
    store.inv_food[idx]     = 0.0
    store.inv_wood[idx]     = 0.0
    store.inv_stone[idx]    = 0.0
    store.inv_minerals[idx] = 0.0
    store.weapon[idx]       = False
    store.lineage_id[idx]   = -1
    store.repro_cd[idx]     = 0
    store.genome[idx]       = 0.5   # reset to neutral genome
    store.last_signal[idx]  = 0

    return died_mask


def reproduce(
    store: EntityStore,
    idx: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
    H: int,
    W: int,
) -> int:
    """
    Process REPRODUCE actions for living agents in `idx` (build-spec §2.8, Phase 5).

    For each parent in ascending slot order:
      - Requires at least one free slot (global soft cap).
      - Child slot = next available free slot (allocated greedily so two parents
        don't grab the same slot).
      - Child position = parent (y, x) + uniform random offset in [-2, 2] each axis,
        clamped to [0, H-1] / [0, W-1] for kin clustering (§G).
      - child.genome = clip(parent.genome + rng.normal(0, mutation_sigma, G), 0, 1).
      - child.lineage_id = parent.lineage_id  (kin share lineage).
      - Child initialised alive, energy=0.5, hydration=1.0, thermal=1.0, health=1.0,
        age=0, repro_cd=cfg.repro_cooldown, is_predator=False, inventory zeroed.
      - Parent: energy -= cfg.repro_energy_cost; repro_cd = cfg.repro_cooldown.

    Returns the number of successful births.
    If no free slot exists when a parent is processed, that parent is skipped (soft cap).

    Parameters
    ----------
    store : EntityStore
        Mutable entity store modified in-place.
    idx : np.ndarray
        Indices of living agents that chose REPRODUCE and passed the energy/cd gate.
        Processed in ascending order (as passed; caller should supply ascending).
    cfg : SimConfig
        Simulation configuration (repro_energy_cost, repro_cooldown, mutation_sigma).
    rng : np.random.Generator
        Seeded NumPy RNG for deterministic mutation and offset sampling.
    H, W : int
        Grid height and width for clamping child positions.

    Returns
    -------
    int
        Number of births this call.
    """
    if idx.size == 0:
        return 0

    G = cfg.genome_dim
    births = 0

    # We track which slots have been allocated this call to avoid double-allocation.
    # Rebuild free list once; remove from it as we allocate.
    free = list(store.free_slots())

    for parent in idx:
        if not free:
            break  # no more capacity; remaining parents skip

        child = free.pop(0)

        # --- child position: parent tile + random offset in [-2, 2], clamped ---
        offset_y = int(rng.integers(-2, 3))   # integers(lo, hi) is [lo, hi)
        offset_x = int(rng.integers(-2, 3))
        cy = int(np.clip(int(store.y[parent]) + offset_y, 0, H - 1))
        cx = int(np.clip(int(store.x[parent]) + offset_x, 0, W - 1))

        # --- child genome: inherited + gaussian mutation, clipped to [0, 1] ---
        child_genome = np.clip(
            store.genome[parent] + rng.normal(0.0, cfg.mutation_sigma, G),
            0.0, 1.0,
        ).astype(np.float32)

        # --- populate child slot ---
        store.alive[child]        = True
        store.is_predator[child]  = False
        store.y[child]            = cy
        store.x[child]            = cx
        store.energy[child]       = np.float32(0.5)
        store.hydration[child]    = np.float32(1.0)
        store.thermal[child]      = np.float32(1.0)
        store.health[child]       = np.float32(1.0)
        store.age[child]          = 0
        store.genome[child]       = child_genome
        store.lineage_id[child]   = int(store.lineage_id[parent])
        store.repro_cd[child]     = cfg.repro_cooldown
        # inventory zeroed (slots are cleared by resolve_deaths, but be explicit)
        store.inv_food[child]     = np.float32(0.0)
        store.inv_wood[child]     = np.float32(0.0)
        store.inv_stone[child]    = np.float32(0.0)
        store.inv_minerals[child] = np.float32(0.0)
        store.weapon[child]       = False
        store.last_signal[child]  = np.int8(0)

        # --- update parent ---
        store.energy[parent] = np.float32(
            max(0.0, float(store.energy[parent]) - cfg.repro_energy_cost)
        )
        store.repro_cd[parent] = cfg.repro_cooldown

        births += 1

    return births
