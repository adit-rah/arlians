# sim/ — the living environment (numpy, no torch)

The mutable simulation layered over the read-only `world/` substrate. Exposes a
PettingZoo-style `Simulation.step(actions) -> StepOut`. **numpy only — never import torch
here** (that line is what keeps the env testable and fast).

## Frozen contracts (do not reshape without a re-lock)

`config.py`, `state.py`, `actions.py` (enum + `build_mask` signature), `observe.py` (channel
layout), and the `Obs`/`Actions`/`StepOut`/`Simulation` signatures in `simulation.py`. Saved
checkpoints and the policy net's input dims depend on these. Forward-declared later-phase
fields exist from the start, zeroed, and are only *used* at their phase.

## Files

- `config.py` — `SimConfig` frozen dataclass: every tunable (drives, food, crops, build,
  combat, predators, repro/genome, reward weights `w_h=1.0`/`w_a=0.05`, curriculum). No magic
  numbers anywhere else — change behavior here.
- `state.py` — `WorldState` (mutable `(H,W)` grids: wild_remaining, crop_*, structure_*,
  stored_food, scent, event_mask) and `EntityStore` (struct-of-arrays length `M`: alive,
  is_predator, y/x, energy/hydration/thermal/health, age, inventory, weapon, lineage_id,
  repro_cd, genome `(M,G)`, last_signal). `create()` allocates zeroed; `Simulation.reset()`
  seeds agents and non-zero neutrals (genome=0.5).
- `actions.py` — `Action` IntEnum (14: NOOP/MOVE/FORAGE/DRINK/PLANT/HARVEST/BUILD/CRAFT/EAT/
  STORE/RETRIEVE/ATTACK/REST/REPRODUCE), `DIRECTIONS` (8 compass), and `build_mask(world,
  state, store, cfg) -> (M,14) bool`. **Masking and the step()-time action gates must stay in
  sync** — e.g. PLANT is masked AND re-checked in step() with the same energy/fertility rule.
- `observe.py` — vectorized `build_observation(...) -> Obs`. 24 spatial channels (0–13 mirror
  `world.get_obs`; 14–19 mutable tile state; 20–23 entity/signal) + a 16-long interoceptive
  vector (10 base + 6 genome). Built with `np.pad` + fancy-index gather — **no per-agent loop**.
  `vector_len(cfg)` and the channel count are consumed by `train/`.
- `dynamics.py` — pure functions on state: decay_drives, update_health, forage, eat, drink,
  regrow_wild, spoil_carried, crop_step, plant, harvest, build, store/retrieve_food,
  structure_decay_step, spoil_stored, resolve_combat, craft_weapon.
- `reproduce.py` — `traits_from_genome` (g0→metab, g1→water_need, g2→cold_tol, g3→capacity;
  g4/g5 reserved), `resolve_deaths(...)` (health≤0 → dead; fills a `deaths_by_cause` dict),
  `reproduce(...)` (mutated genome, child near parent).
- `threats.py` — predators (share `EntityStore` via `is_predator`; climb a blurred `scent`
  gradient), catastrophes (cold_snap/drought/flood/storm via `event_mask`), scent update.
- `curriculum.py` — `Curriculum.D ∈ [0,1]` scales winter/exposure/economy; ramps when stable.
  `exposure_scale` is read in `simulation.step()`.
- `metrics.py` — `MetricsLogger` (per-year Section-H aggregates: pct_calories_farmed,
  occupancies, deaths_by_cause, structures, specialization index, signal↔action MI, …).
  **Heavy** (per-agent Python loops) — used by `scripts/demo.py`, NOT the training hot loop.
  Training uses the lean `RolloutStats` in `train/ppo.py` instead.
- `render.py` — top-down RGB frame (`render_frame(world, state, store)`) for GIFs/PNGs.
- `simulation.py` — the facade. `step()` runs the §2.1 fixed-phase order: MOVE → ATTACK/CRAFT →
  FORAGE/PLANT/HARVEST/BUILD → EAT/DRINK/STORE/RETRIEVE/REPRODUCE/emit → world dynamics →
  predators → catastrophes → drive decay/health/deaths → reward → observe. Also has
  `respawn_dead()` (the Phase 1–4 crutch).

## Reward (frozen, evolution-only)

`comfort = (energy+hydration+thermal)/3`; `r = w_h*comfort + w_a` per living agent, every
step. **No behavior or lineage terms** — civilization is meant to emerge from genome
selection + population-weighted gradients, not from reward shaping. Don't add behavior terms.

## Gotchas

- `respawn_dead` re-seeds at *random* land tiles → looks like teleporting in GIFs and removes
  survival/reproduction pressure. It's a training crutch; off for honest survival runs.
- Step ordering is deterministic (ascending slot id) on purpose — keep it.
- Wild food is intentionally sparse (world layer) → foraging alone is barely survivable.

## Test

```bash
.venv/bin/python -m pytest tests/ -q          # all
.venv/bin/python -m pytest tests/test_contracts.py -q   # contract-shape pins
```
Per-component tests + per-phase gate tests live in `tests/` (one module per component/phase).
