# world/ — the read-only substrate

Procedural world generator. **Finished and frozen.** It is a pure, deterministic function
of `(seed, width, height)`; it holds no mutable simulation state and never changes during a
run. The agent layer (`sim/`) reads from it and layers mutable state on top.

If you're here to change agent behavior, you're probably in the wrong directory — see `sim/`.

## Files

- `config.py` — `WorldConfig` (width, height, seed, `sea_level`, `season_period`, noise/biome
  knobs). Grid size + season period come from here, NOT from `SimConfig`.
- `noise_layers.py` — FFT spectral noise: elevation (with radial island falloff), moisture,
  latitudinal temperature baseline.
- `biome.py` — classifies 10 biomes (`0..9`) from elevation/moisture/temperature. Note: 10
  biomes including `JUNGLE` (the original brief said 9 — it's wrong).
- `rivers.py` — river path tracing downhill + per-tile distance to nearest river.
- `resources.py` — 6 base resource layers (soil_fertility, water_proximity, wood, stone,
  wild_food, minerals) from biome + river proximity. **Wild food is exponentially sparsified**
  to a tiny fraction of tiles, and fertility is highest on wetland/river tiles — this is the
  deliberate bias that makes settling/farming the optimal strategy.
- `seasons.py` — `compute_season_state(t, cfg)` and `apply_seasonal_modifiers(...)`. Season is
  a cheap scalar multiplier per timestep applied to fertility / wild_food / water / temperature.
  Winter crushes foraging (`wild_food_modifier ≈ 0.05`) far harder than farming (`≈ 0.20`).
- `world.py` — the `World` dataclass + generation pipeline. The API `sim/` depends on:
  - `World.generate(cfg) -> World`
  - `world.base_resources["..."]` — the 6 unmodified base resource `(H,W)` arrays.
  - `world.elevation`, `world.temperature_base`, `world.river_mask`, etc.
  - `get_obs(t) -> (14,H,W)`: 7 static geo + 6 season-modified resources + 1 season_phase.
  - `get_window(...)`: **slow** Python double-loop; kept only as the correctness reference for
    the vectorized `sim/observe.py`. **Never call it in the hot loop.**
- `export.py` — write a generated world to `data/world_<seed>/` (used by `generate.py` and
  the `stream/` server).

## Key facts / gotchas

- Channel order is `LAYER_NAMES` in `world.py` (13 atlas channels); `get_obs` adds a 14th
  (season_phase). `sim/observe.py` channels 0–13 mirror these exactly.
- `biome_map` is stored normalized in the atlas as `biome/9.0`.
- Generation is ~0.2s for 1024². Deterministic per seed — same seed ⇒ identical world.
- `sea_level` defines land (`elevation >= sea_level`); agents only spawn/move on land.

## Run / inspect

```bash
.venv/bin/python generate.py --seed 42 --width 256 --height 256   # -> data/world_42/
```

Don't add mutable or per-agent state here. That belongs in `sim/state.py`.
