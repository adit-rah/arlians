"""
World dynamics — pure functions advancing mutable state one step (build-spec §2.3-2.5).

Filled incrementally by the env-dynamics fleet role:
  Phase 1: forage depletion/regrowth (§2.4), drive decay + health update (§2.5)
  Phase 3: crop growth + winter-failure rot (§2.3)
  Phase 4: thermal update, spoilage, structure decay

All functions operate in-place / vectorized on WorldState + EntityStore, driven by
SimConfig + the world's seasonal scalars. No magic numbers — everything from SimConfig.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .state import EntityStore, WorldState
from .reproduce import traits_from_genome


# ---------------------------------------------------------------------------
# Phase 1 drive dynamics
# ---------------------------------------------------------------------------

def decay_drives(
    store: EntityStore,
    cfg: SimConfig,
    *,
    temp_base: np.ndarray | None = None,
    temperature_modifier: float = 1.0,
    state: "WorldState | None" = None,
    exposure_scale: float = 1.0,
) -> None:
    """
    Decay per-step energy, hydration, and thermal (Phase 4) for all LIVING agents.

    Phase 1: energy decays.
      energy -= cfg.energy_decay * metab   (clip >= 0)

    Phase 2: hydration also decays.
      hydration -= cfg.hydration_decay * water_need   (clip >= 0)

    Phase 4: thermal updated via eff_temp:
      eff_temp = temp_base[tile] * temperature_modifier + cold_tol
                 + (shelter_temp_bonus if sheltered)
      if eff_temp >= thermal_target_temp: thermal += thermal_warm_regen
      else:                               thermal -= thermal_decay * exposure_scale
      clip thermal to [0, 1].

    `temp_base` is the world.temperature_base (H,W) array; if None thermal is skipped.
    `state` is needed to check structure_type for shelter status (Phase 4).
    `exposure_scale` is curriculum.exposure_scale (D).
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return

    traits = traits_from_genome(store.genome)
    # energy -= energy_decay * metab; only for living agents
    store.energy[living] = np.clip(
        store.energy[living] - cfg.energy_decay * traits.metab[living],
        0.0, None,
    ).astype(np.float32)
    # hydration -= hydration_decay * water_need; only for living agents (Phase 2)
    store.hydration[living] = np.clip(
        store.hydration[living] - cfg.hydration_decay * traits.water_need[living],
        0.0, None,
    ).astype(np.float32)

    # Phase 4: thermal update
    if temp_base is not None:
        live_idx = np.flatnonzero(living)
        ys = store.y[live_idx]
        xs = store.x[live_idx]

        # Tile temperature (base * seasonal modifier)
        tile_temp = (temp_base[ys, xs] * temperature_modifier).astype(np.float32)

        # Shelter bonus: structure_type==1 on the tile
        if state is not None:
            sheltered = (state.structure_type[ys, xs] == 1).astype(np.float32)
        else:
            sheltered = np.zeros(live_idx.size, dtype=np.float32)

        shelter_bonus = sheltered * cfg.shelter_temp_bonus

        # Effective temperature = tile_temp + cold_tol + shelter bonus
        cold_tol = traits.cold_tol[live_idx]
        eff_temp = tile_temp + cold_tol + shelter_bonus

        warm = eff_temp >= cfg.thermal_target_temp
        delta = np.where(
            warm,
            cfg.thermal_warm_regen,
            -cfg.thermal_decay * exposure_scale,
        ).astype(np.float32)

        store.thermal[live_idx] = np.clip(
            store.thermal[live_idx] + delta, 0.0, 1.0
        ).astype(np.float32)


def update_health(
    store: EntityStore,
    cfg: SimConfig,
    *,
    temp_base: np.ndarray | None = None,
    temperature_modifier: float = 1.0,
    state: "WorldState | None" = None,
    exposure_scale: float = 1.0,
) -> None:
    """
    Regenerate or damage health based on homeostatic drive status.

    Phase 4 rules (energy, hydration, AND thermal are active drives):
      - If all three drives >= cfg.drive_safe_band -> health += cfg.health_regen
      - Each drive at 0 -> health -= cfg.starve_damage  (independent, additive)
      - Exposure damage (Phase 4): when eff_temp < thermal_target_temp:
          health -= cfg.exposure_damage * exposure_scale
                    * (shelter_exposure_mult if sheltered else 1.0)

    `temp_base` is world.temperature_base (H,W); if None exposure damage is skipped.
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return

    energy    = store.energy[living]
    hydration = store.hydration[living]
    thermal   = store.thermal[living]
    health    = store.health[living]

    # All three active drives must be in safe band to regen health
    safe = (
        (energy >= cfg.drive_safe_band)
        & (hydration >= cfg.drive_safe_band)
        & (thermal >= cfg.drive_safe_band)
    )
    health = np.where(safe, health + cfg.health_regen, health)

    # Each drive at zero deals independent starve_damage
    starving    = energy    <= 0.0
    dehydrated  = hydration <= 0.0
    freezing    = thermal   <= 0.0
    damage = (
        starving.astype(np.float32)
        + dehydrated.astype(np.float32)
        + freezing.astype(np.float32)
    ) * cfg.starve_damage
    health = health - damage

    # Phase 4 exposure damage
    if temp_base is not None:
        live_idx = np.flatnonzero(living)
        ys = store.y[live_idx]
        xs = store.x[live_idx]

        tile_temp = (temp_base[ys, xs] * temperature_modifier).astype(np.float32)

        if state is not None:
            sheltered = state.structure_type[ys, xs] == 1
        else:
            sheltered = np.zeros(live_idx.size, dtype=bool)

        traits = traits_from_genome(store.genome)
        cold_tol = traits.cold_tol[live_idx]
        shelter_bonus = sheltered.astype(np.float32) * cfg.shelter_temp_bonus
        eff_temp = tile_temp + cold_tol + shelter_bonus

        cold_exposure = eff_temp < cfg.thermal_target_temp
        mult = np.where(sheltered, cfg.shelter_exposure_mult, 1.0).astype(np.float32)
        exp_damage = (cold_exposure.astype(np.float32)
                      * cfg.exposure_damage * exposure_scale * mult)
        health = health - exp_damage

    store.health[living] = np.clip(health, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 foraging & eating
# ---------------------------------------------------------------------------

def forage(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
    world=None,
) -> None:
    """
    Execute FORAGE for the agents indicated by `idx` (integer array of slot indices).

    Phase 1 (food):
        take = min(capacity - inv_food, wild_remaining[tile], cfg.forage_yield)
        deduct from wild_remaining, add to inv_food.

    Phase 4 (materials — abiotic, NOT depleted from the tile):
        inv_wood  += min(material_cap - inv_wood,  wood_layer[tile])
        inv_stone += min(material_cap - inv_stone, stone_layer[tile])

    `idx` must be pre-filtered to contain only living agents that chose FORAGE.
    The order of idx is the conflict-resolution order (ascending slot index for
    determinism); agents processed later see the updated wild_remaining.

    `world` (optional): if provided, also gathers wood/stone from
        world.base_resources["wood"] / ["stone"].  If None, only food is gathered.

    Uses traits_from_genome to get per-agent carry capacity.
    """
    if idx.size == 0:
        return

    traits = traits_from_genome(store.genome)
    # Per-agent carry capacity = base * genome capacity trait
    cap_per_agent = (cfg.carry_capacity * traits.capacity[idx]).astype(np.float32)
    # Per-agent material capacity = base * genome capacity trait
    mat_cap_per_agent = (cfg.material_capacity * traits.capacity[idx]).astype(np.float32)

    ys = store.y[idx]
    xs = store.x[idx]

    # Material layers (None if world not provided)
    wood_layer     = world.base_resources["wood"]     if world is not None else None
    stone_layer    = world.base_resources["stone"]    if world is not None else None
    minerals_layer = world.base_resources["minerals"] if world is not None else None

    # Process tile by tile for correct conflict resolution.
    # Group foragers by tile so we apply depletion atomically per tile;
    # within a tile, process in the order they appear in idx (ascending slot).
    # We iterate unique tiles but keep the order from idx for within-tile ordering.
    for slot_pos, (slot, y, x, cap, mat_cap) in enumerate(
        zip(idx, ys, xs, cap_per_agent, mat_cap_per_agent)
    ):
        # --- food (biotic, depletes tile) ---
        available = state.wild_remaining[y, x]
        if available > 0.0:
            room = cap - store.inv_food[slot]
            if room > 0.0:
                take = float(min(room, available, cfg.forage_yield))
                if take > 0.0:
                    state.wild_remaining[y, x] = max(0.0, available - take)
                    store.inv_food[slot] = min(cap, store.inv_food[slot] + take)

        # --- wood (abiotic, NOT depleted) ---
        if wood_layer is not None:
            wood_avail = float(wood_layer[y, x])
            if wood_avail > 0.0:
                room_w = float(mat_cap) - float(store.inv_wood[slot])
                if room_w > 0.0:
                    gain_w = min(room_w, wood_avail)
                    store.inv_wood[slot] = np.float32(
                        min(float(mat_cap), float(store.inv_wood[slot]) + gain_w)
                    )

        # --- stone (abiotic, NOT depleted) ---
        if stone_layer is not None:
            stone_avail = float(stone_layer[y, x])
            if stone_avail > 0.0:
                room_s = float(mat_cap) - float(store.inv_stone[slot])
                if room_s > 0.0:
                    gain_s = min(room_s, stone_avail)
                    store.inv_stone[slot] = np.float32(
                        min(float(mat_cap), float(store.inv_stone[slot]) + gain_s)
                    )

        # --- minerals (abiotic, NOT depleted) ---
        if minerals_layer is not None:
            minerals_avail = float(minerals_layer[y, x])
            if minerals_avail > 0.0:
                room_m = float(mat_cap) - float(store.inv_minerals[slot])
                if room_m > 0.0:
                    gain_m = min(room_m, minerals_avail)
                    store.inv_minerals[slot] = np.float32(
                        min(float(mat_cap), float(store.inv_minerals[slot]) + gain_m)
                    )


def eat(store: EntityStore, idx: np.ndarray, cfg: SimConfig) -> None:
    """
    Convert carried food to energy for agents in `idx` that chose EAT.

    Mechanics:
        need     = (1 - energy) / cfg.eat_restore   [food units needed to fill up]
        consumed = min(inv_food, need)
        energy  += cfg.eat_restore * consumed        (clip <= 1)
        inv_food -= consumed                         (clip >= 0)

    `idx` must be pre-filtered to living agents that chose EAT.
    """
    if idx.size == 0:
        return

    energy   = store.energy[idx]
    inv_food = store.inv_food[idx]

    need     = (1.0 - energy) / cfg.eat_restore
    consumed = np.minimum(inv_food, need).astype(np.float32)
    consumed = np.maximum(consumed, 0.0)

    store.energy[idx]   = np.clip(energy + cfg.eat_restore * consumed, 0.0, 1.0).astype(np.float32)
    store.inv_food[idx] = np.maximum(inv_food - consumed, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 2 drinking
# ---------------------------------------------------------------------------

def drink(
    store: EntityStore,
    water_proximity: np.ndarray,
    idx: np.ndarray,
    cfg: SimConfig,
) -> None:
    """
    Execute DRINK for the agents indicated by `idx` (integer array of slot indices).

    For each agent in `idx` whose tile has water_proximity >= cfg.drink_min_water,
    hydration is restored:
        hydration = min(1.0, hydration + cfg.drink_restore)

    Agents on a dry tile (water_proximity < drink_min_water) are no-ops — the mask
    should have already prevented this, but we guard here for safety.

    `idx` must be pre-filtered to contain only living agents that chose DRINK.
    `water_proximity` is a (H, W) float32 array (base_resources or get_obs channel 8).
    """
    if idx.size == 0:
        return

    ys = store.y[idx]
    xs = store.x[idx]
    tile_water = water_proximity[ys, xs]

    # Only agents on sufficiently watered tiles benefit
    can_drink = tile_water >= cfg.drink_min_water
    drink_idx = idx[can_drink]

    if drink_idx.size == 0:
        return

    store.hydration[drink_idx] = np.clip(
        store.hydration[drink_idx] + cfg.drink_restore, 0.0, 1.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 world regeneration & spoilage
# ---------------------------------------------------------------------------

def regrow_wild(state: WorldState, world, t: int, cfg: SimConfig) -> None:
    """
    Regrow wild food toward the seasonal cap each step.

    cap = world.base_resources["wild_food"] * season.wild_food_modifier
    wild_remaining += cfg.forage_regen * (cap - wild_remaining)
    Clip to >= 0 (can't go below zero).
    """
    from world.seasons import compute_season_state
    season = compute_season_state(t, world.cfg)
    cap = (world.base_resources["wild_food"] * season.wild_food_modifier).astype(np.float32)
    state.wild_remaining += cfg.forage_regen * (cap - state.wild_remaining)
    np.clip(state.wild_remaining, 0.0, None, out=state.wild_remaining)


def spoil_carried(store: EntityStore, cfg: SimConfig) -> None:
    """
    Spoil food carried in agent inventories each step.

        inv_food -= cfg.spoilage_carried   (clip >= 0)

    Only living agents are affected.
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return
    store.inv_food[living] = np.maximum(
        store.inv_food[living] - cfg.spoilage_carried, 0.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 3 crop dynamics
# ---------------------------------------------------------------------------

def crop_step(state: WorldState, world, t: int, cfg: SimConfig) -> None:
    """
    Advance crop growth and apply winter-failure rot — vectorized over the grid.

    Per §2.3:
      eff_fert = soil_fertility * fertility_modifier(t)
      growth   = crop_base_growth * eff_fert * water_proximity
      crop_stage[growing] = min(1.0, crop_stage + growth)

      rot condition: growing tile AND eff_fert < crop_min_fertility
      crop_health[rot] -= crop_rot

      Fully-rotted tiles (crop_health <= 0): clear crop_stage, crop_health, crop_owner.
    """
    from world.seasons import compute_season_state

    season = compute_season_state(t, world.cfg)
    soil_fertility  = world.base_resources["soil_fertility"]          # (H, W) f32
    water_proximity = world.base_resources["water_proximity"]         # (H, W) f32

    eff_fert = (soil_fertility * season.fertility_modifier).astype(np.float32)

    growing = state.crop_stage > 0.0  # (H, W) bool

    # Growth on all growing tiles
    growth = (cfg.crop_base_growth * eff_fert * water_proximity).astype(np.float32)
    new_stage = np.where(growing, np.minimum(1.0, state.crop_stage + growth), state.crop_stage)
    state.crop_stage[:] = new_stage.astype(np.float32)

    # Rot: growing AND eff_fert below threshold
    rot = growing & (eff_fert < cfg.crop_min_fertility)
    state.crop_health[rot] = (state.crop_health[rot] - cfg.crop_rot).astype(np.float32)

    # Clear fully-rotted tiles
    dead = state.crop_health <= 0.0
    clear = growing & dead
    state.crop_stage[clear]  = 0.0
    state.crop_health[clear] = 0.0
    state.crop_owner[clear]  = -1


# eff_fert at which plant success probability reaches 1 (matches metrics fertile signal)
_PLANT_FERTILE_CAP = 0.6


def _plant_success_prob(eff_fert: float, cfg: SimConfig) -> float:
    """Steep ramp: 0 below crop_min_fertility, ~1 at _PLANT_FERTILE_CAP (cubic in between)."""
    if eff_fert < cfg.crop_min_fertility:
        return 0.0
    if eff_fert >= _PLANT_FERTILE_CAP:
        return 1.0
    t = (eff_fert - cfg.crop_min_fertility) / (
        _PLANT_FERTILE_CAP - cfg.crop_min_fertility
    )
    return float(t * t * t)


def plant(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
    world,
    t: int,
    rng: np.random.Generator,
) -> int:
    """
    Execute PLANT for agents in `idx` (chose PLANT action).

    For each agent (ascending slot order for determinism):
      - If the tile already has a crop: no-op (no cost).
      - On empty land: pay plant_energy_cost, then succeed with probability
        p(eff_fert) where eff_fert = soil_fertility * season.fertility_modifier.
      - On success: crop_stage = 0.01, crop_health = 1.0, crop_owner = lineage_id.

    Returns total count of successful plants this step.
    """
    from world.seasons import compute_season_state

    if idx.size == 0:
        return 0

    season = compute_season_state(t, world.cfg)
    soil_fertility = world.base_resources["soil_fertility"]
    cost = float(cfg.plant_energy_cost)

    planted = 0
    for slot in idx:
        y = int(store.y[slot])
        x = int(store.x[slot])
        if state.crop_stage[y, x] != 0.0:
            continue

        eff_fert = float(soil_fertility[y, x] * season.fertility_modifier)
        store.energy[slot] = np.float32(
            max(0.0, float(store.energy[slot]) - cost)
        )
        if rng.random() < _plant_success_prob(eff_fert, cfg):
            state.crop_stage[y, x]  = np.float32(0.01)
            state.crop_health[y, x] = np.float32(1.0)
            state.crop_owner[y, x]  = int(store.lineage_id[slot])
            planted += 1
    return planted


def harvest(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
) -> float:
    """
    Execute HARVEST for agents in `idx` (chose HARVEST action).

    For each agent (ascending slot order for determinism):
      - If the agent's tile has crop_stage >= 1.0 (mature):
          food_yield = cfg.crop_yield * crop_stage * crop_health
          inv_food[slot] += food_yield, capped at carry_capacity * genome capacity
          Clear tile: crop_stage=0, crop_health=0, crop_owner=-1
      - Otherwise: no-op (immature tile).

    Returns total food harvested this step.
    """
    if idx.size == 0:
        return 0.0

    traits = traits_from_genome(store.genome)
    total_harvested = 0.0

    for slot in idx:
        y = int(store.y[slot])
        x = int(store.x[slot])
        if state.crop_stage[y, x] >= 1.0:
            food_yield = float(cfg.crop_yield * state.crop_stage[y, x] * state.crop_health[y, x])
            cap = float(cfg.carry_capacity * traits.capacity[slot])
            room = cap - float(store.inv_food[slot])
            gain = min(food_yield, max(0.0, room))
            store.inv_food[slot] = np.float32(min(cap, float(store.inv_food[slot]) + gain))
            total_harvested += gain
            # Clear the tile
            state.crop_stage[y, x]  = np.float32(0.0)
            state.crop_health[y, x] = np.float32(0.0)
            state.crop_owner[y, x]  = np.int32(-1)
    return float(total_harvested)


# ---------------------------------------------------------------------------
# Phase 4 structures: build, store/retrieve, decay, spoil stored
# ---------------------------------------------------------------------------

def build(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    param: np.ndarray,
    cfg: SimConfig,
) -> int:
    """
    Execute BUILD for agents in `idx` (chose BUILD action).

    For each agent (ascending slot order for determinism):
      - Tile must have structure_type == 0 (no existing structure).
      - `param[slot]` selects the structure type (1=SHELTER, 2=STORAGE, 3=WALL).
      - Agent must carry enough material (wood/stone) per the cost dict.
      - On success: deduct cost, set structure_type and structure_hp=structure_init_hp.

    Returns count of structures built this step.
    """
    if idx.size == 0:
        return 0

    cost_map = {
        1: cfg.shelter_cost,   # SHELTER
        2: cfg.storage_cost,   # STORAGE
        3: cfg.wall_cost,      # WALL — enabled in Phase 6
    }

    built = 0
    for slot in idx:
        y = int(store.y[slot])
        x = int(store.x[slot])
        if state.structure_type[y, x] != 0:
            continue  # tile already occupied

        stype = int(param[slot])
        if stype not in cost_map:
            continue  # unsupported type (e.g. WALL) or invalid param

        cost = cost_map[stype]
        wood_cost  = float(cost.get("wood",  0.0))
        stone_cost = float(cost.get("stone", 0.0))

        if float(store.inv_wood[slot]) < wood_cost:
            continue
        if float(store.inv_stone[slot]) < stone_cost:
            continue

        # Deduct cost and place structure
        store.inv_wood[slot]  = np.float32(float(store.inv_wood[slot])  - wood_cost)
        store.inv_stone[slot] = np.float32(float(store.inv_stone[slot]) - stone_cost)
        state.structure_type[y, x] = np.int8(stype)
        state.structure_hp[y, x]   = np.float32(cfg.structure_init_hp)
        built += 1

    return built


def store_food(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
) -> float:
    """
    Execute STORE for agents in `idx` (chose STORE action).

    Moves inv_food -> stored_food on the agent's tile, which must be a STORAGE
    structure (structure_type==2).  Transfer is capped by storage_capacity.

    Returns total food deposited this step.
    """
    if idx.size == 0:
        return 0.0

    total = 0.0
    for slot in idx:
        y = int(store.y[slot])
        x = int(store.x[slot])
        if int(state.structure_type[y, x]) != 2:
            continue  # not a storage tile
        already = float(state.stored_food[y, x])
        room = cfg.storage_capacity - already
        if room <= 0.0:
            continue
        move = min(float(store.inv_food[slot]), room)
        if move <= 0.0:
            continue
        state.stored_food[y, x] = np.float32(already + move)
        store.inv_food[slot]    = np.float32(float(store.inv_food[slot]) - move)
        total += move

    return float(total)


def retrieve_food(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
) -> float:
    """
    Execute RETRIEVE for agents in `idx` (chose RETRIEVE action).

    Moves stored_food -> inv_food on the agent's tile (must be STORAGE structure).
    Transfer is capped by agent's carry capacity (genome-scaled).

    Returns total food retrieved this step.
    """
    if idx.size == 0:
        return 0.0

    traits = traits_from_genome(store.genome)
    total = 0.0

    for slot in idx:
        y = int(store.y[slot])
        x = int(store.x[slot])
        if int(state.structure_type[y, x]) != 2:
            continue  # not a storage tile
        available = float(state.stored_food[y, x])
        if available <= 0.0:
            continue
        cap = float(cfg.carry_capacity * traits.capacity[slot])
        room = cap - float(store.inv_food[slot])
        if room <= 0.0:
            continue
        move = min(available, room)
        if move <= 0.0:
            continue
        store.inv_food[slot]    = np.float32(float(store.inv_food[slot]) + move)
        state.stored_food[y, x] = np.float32(available - move)
        total += move

    return float(total)


def structure_decay_step(state: WorldState, cfg: SimConfig) -> int:
    """
    Decay all structure HP each step; clear structures that reach hp <= 0.

    structure_hp -= cfg.structure_decay   (for all tiles with structure_type != 0)
    Tiles with hp <= 0 after decay: structure_type=0, structure_hp=0, stored_food=0.

    Returns count of structures that collapsed this step.
    """
    has_structure = state.structure_type != 0
    if not has_structure.any():
        return 0

    state.structure_hp[has_structure] = (
        state.structure_hp[has_structure] - cfg.structure_decay
    ).astype(np.float32)

    collapsed = has_structure & (state.structure_hp <= 0.0)
    n_collapsed = int(collapsed.sum())
    if n_collapsed > 0:
        state.structure_type[collapsed] = np.int8(0)
        state.structure_hp[collapsed]   = np.float32(0.0)
        state.stored_food[collapsed]    = np.float32(0.0)

    return n_collapsed


def spoil_stored(state: WorldState, cfg: SimConfig) -> None:
    """
    Spoil food sitting in storage structures each step.

        stored_food -= cfg.spoilage_stored   (clip >= 0)

    Only tiles with stored_food > 0 are affected.
    """
    has_food = state.stored_food > 0.0
    if not has_food.any():
        return
    state.stored_food[has_food] = np.maximum(
        state.stored_food[has_food] - cfg.spoilage_stored, 0.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 6 combat: resolve_combat and craft_weapon
# ---------------------------------------------------------------------------

def resolve_combat(
    store: EntityStore,
    state: WorldState,
    attacker_idx: np.ndarray,
    param: np.ndarray,
    cfg: SimConfig,
) -> dict:
    """
    Execute ATTACK for agents in `attacker_idx` (chose ATTACK, living).

    Uses a PRE-STEP SNAPSHOT of positions and HP so that simultaneous/mutual
    attacks both land (if A and B attack each other, both take damage this step).

    For each attacker:
      - target_tile = attacker (y, x) + DIRECTIONS[param[slot]], clamped to bounds.
      - Find ALL living entities (agents or predators) on that tile.
      - Compute damage per attacker:
          dmg = attack_damage
                * (weapon_attack_mult if attacker has weapon else 1.0)
                * (wall_defense_mult if target tile has WALL else 1.0)
                * group_mult
        where group_mult = max(group_defense_floor,
                               group_defense_per_ally ** n_other_living_on_target)
        "n_other_living_on_target" = all living entities on target tile minus the
        one we're computing damage against (i.e., co-located allies protect each target).
      - Accumulate damage per target slot (multiple attackers stack).
      - Apply accumulated damage AFTER all attackers are evaluated (atomic step).

    Returns a dict: {"conflict_slots": set_of_slot_indices_that_took_combat_damage}
    This is used by resolve_deaths to attribute "conflict" cause of death.

    Design note: predator attackers reuse this function — they simply have weapon=False.
    """
    from .actions import DIRECTIONS, StructureType

    if attacker_idx.size == 0:
        return {"conflict_slots": set()}

    H, W = state.structure_type.shape

    # --- PRE-STEP SNAPSHOT ---
    # Take snapshot of positions and living status BEFORE applying any damage.
    snap_alive = store.alive.copy()
    snap_y     = store.y.copy()
    snap_x     = store.x.copy()

    # Build a fast tile -> [slot] lookup using snapshot positions
    # Only include living entities (agents + predators) in the lookup
    live_mask = snap_alive
    live_slots = np.flatnonzero(live_mask)

    # Map (y, x) -> list of living slots on that tile
    from collections import defaultdict
    tile_to_slots: dict = defaultdict(list)
    for slot in live_slots:
        tile_to_slots[(int(snap_y[slot]), int(snap_x[slot]))].append(slot)

    # Accumulate damage per target slot
    damage_accum: dict[int, float] = {}

    for atk in attacker_idx:
        if not snap_alive[atk]:
            continue  # attacker died earlier this step (shouldn't happen, but guard)

        # Compute target tile
        dir_idx = int(param[atk]) % len(DIRECTIONS)
        dy, dx  = DIRECTIONS[dir_idx]
        ty = int(np.clip(int(snap_y[atk]) + dy, 0, H - 1))
        tx = int(np.clip(int(snap_x[atk]) + dx, 0, W - 1))

        # Targets: all living entities on the target tile
        targets_on_tile = tile_to_slots.get((ty, tx), [])
        if not targets_on_tile:
            continue

        # Is the target tile a WALL?
        target_on_wall = int(state.structure_type[ty, tx]) == int(StructureType.WALL)
        wall_mult = float(cfg.wall_defense_mult) if target_on_wall else 1.0

        # Weapon multiplier for attacker
        weapon_mult = float(cfg.weapon_attack_mult) if bool(store.weapon[atk]) else 1.0

        # Group defense: n_other = number of OTHER living entities co-located with each
        # specific target. For each target individually, it's len(targets_on_tile) - 1.
        n_others = len(targets_on_tile) - 1
        group_mult = max(
            float(cfg.group_defense_floor),
            float(cfg.group_defense_per_ally) ** n_others,
        )

        base_dmg = (
            float(cfg.attack_damage)
            * weapon_mult
            * wall_mult
            * group_mult
        )

        for tgt in targets_on_tile:
            damage_accum[tgt] = damage_accum.get(tgt, 0.0) + base_dmg

    # --- Apply accumulated damage ---
    conflict_slots: set = set()
    for tgt_slot, dmg in damage_accum.items():
        if not store.alive[tgt_slot]:
            continue  # target already dead (can't happen in Phase 6 but guard anyway)
        store.health[tgt_slot] = np.float32(
            max(0.0, float(store.health[tgt_slot]) - dmg)
        )
        conflict_slots.add(tgt_slot)

    return {"conflict_slots": conflict_slots}


def craft_weapon(
    store: EntityStore,
    idx: np.ndarray,
    cfg: SimConfig,
) -> int:
    """
    Execute CRAFT (weapon) for agents in `idx` (chose CRAFT, living, unarmed).

    For each agent:
      - Must not already have a weapon (store.weapon[slot] == False).
      - Must carry enough inv_wood, inv_stone, inv_minerals per cfg.weapon_cost.
      - On success: deduct cost from inventory, set weapon=True.

    Returns the count of weapons crafted this step.
    """
    if idx.size == 0:
        return 0

    wood_cost     = float(cfg.weapon_cost.get("wood",     0.0))
    stone_cost    = float(cfg.weapon_cost.get("stone",    0.0))
    minerals_cost = float(cfg.weapon_cost.get("minerals", 0.0))

    crafted = 0
    for slot in idx:
        if bool(store.weapon[slot]):
            continue  # already armed — no-op
        if float(store.inv_wood[slot])     < wood_cost:
            continue
        if float(store.inv_stone[slot])    < stone_cost:
            continue
        if float(store.inv_minerals[slot]) < minerals_cost:
            continue

        # Deduct cost and arm the agent
        store.inv_wood[slot]     = np.float32(float(store.inv_wood[slot])     - wood_cost)
        store.inv_stone[slot]    = np.float32(float(store.inv_stone[slot])    - stone_cost)
        store.inv_minerals[slot] = np.float32(float(store.inv_minerals[slot]) - minerals_cost)
        store.weapon[slot]       = True
        crafted += 1

    return crafted
