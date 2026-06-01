# Arlians World

## What the world is

A 1024×1024 tile grid. Each tile is a fixed point in space with a set of numerical attributes describing its geography and resources. The world is the observation environment for ML agents — everything an agent can know about where it stands comes from the tile data beneath and around it.

The world is designed to make **settling and farming the dominant strategy** over nomadic hunting. This is not enforced by rules — it emerges from how resources are distributed geographically.

---

## How it's generated

Run with:
```bash
python generate.py --seed 42
```

Generation takes ~0.2 seconds and outputs to `data/world_42/`.

### Pipeline

```
1. Elevation      — FFT spectral noise + radial island falloff
2. Moisture       — FFT spectral noise (independent)
3. Temperature    — Latitudinal gradient (hot equator, cold poles) + noise
4. Biomes         — Classified from the three fields above
5. Rivers         — Path tracing from high-elevation sources downhill
6. River distance — Per-tile euclidean distance to nearest river
7. Resources      — 6 resource layers built from biome + river proximity
```

#### Terrain (steps 1–3)
Elevation, moisture, and temperature are each generated as independent noise fields using FFT spectral synthesis — white noise passed through a power-law frequency filter. This produces natural fractal terrain in O(N log N) time. Elevation has a radial falloff applied so the map forms an island (ocean at the edges, land rising toward the center).

#### Biomes (step 4)
Each tile is assigned a biome based on its elevation, moisture, and temperature values:

| Biome | Conditions |
|---|---|
| OCEAN | elevation below sea level |
| BEACH | thin strip just above sea level |
| DESERT | low moisture |
| GRASSLAND | moderate moisture |
| FOREST | high moisture |
| TUNDRA | cold temperature, moderate moisture |
| MOUNTAIN | high elevation |
| SNOW_PEAK | very high elevation |
| WETLAND | very high moisture, low-mid elevation |

**WETLAND** is the key biome for civilization formation. It only appears where rivers pool in low-lying wet areas and has the highest soil fertility of any biome.

#### Rivers (step 5–6)
Rivers are computed by tracing steepest-descent paths from the top 10% of land elevation downhill to the ocean. Cells that many paths flow through become river tiles. A per-tile `river_distance` field is then computed — 0.0 on a river cell, 1.0 at the farthest point from any river.

#### Resources (step 7)
Six resource layers are built per tile:

| Resource | Description | Seasonal? |
|---|---|---|
| `soil_fertility` | How farmable the tile is | Yes |
| `water_proximity` | Access to fresh water | Yes |
| `wood` | Timber availability | No |
| `stone` | Stone/quarry quality | No |
| `wild_food` | Huntable food (intentionally sparse) | Yes |
| `minerals` | Ore and mineral deposits | No |

Each resource starts from a biome base value, gets 30% noise variation added, then river proximity bonus applied on top. Wetland tiles have the highest base `soil_fertility` (0.9). `wild_food` is sparsified with an exponential random mask so only ~0.1% of tiles have meaningful wild food — hunting requires travel.

---

## What it generates as

### Files (`data/world_<seed>/`)

| File | Description |
|---|---|
| `atlas.npy` | Shape `(13, 1024, 1024)` float32 — all static layers stacked |
| `atlas_meta.json` | Layer names, value ranges, biome enum, seed, shape |
| `world_config.json` | Full generation parameters |
| `biome_map.npy` | Shape `(1024, 1024)` int8 — biome enum values |
| `river_mask.npy` | Shape `(1024, 1024)` uint8 — 1 where a river exists |

### Atlas channel layout

| Channel | Name | Type | Range |
|---|---|---|---|
| 0 | elevation | float32 | [0, 1] |
| 1 | moisture | float32 | [0, 1] |
| 2 | temperature_base | float32 | [0, 1] |
| 3 | biome | float32 | [0, 1] (normalized from 0–9) |
| 4 | river_mask | float32 | {0, 1} |
| 5 | river_distance | float32 | [0, 1] |
| 6 | flow_accumulation | float32 | [0, 1] (log-normalized) |
| 7 | soil_fertility | float32 | [0, 1] |
| 8 | water_proximity | float32 | [0, 1] |
| 9 | wood | float32 | [0, 1] |
| 10 | stone | float32 | [0, 1] |
| 11 | wild_food | float32 | [0, 1] |
| 12 | minerals | float32 | [0, 1] |

---

## Seasons

The world has a seasonal cycle of 360 timesteps per year. At each timestep `t`, resource layers are modulated by smooth sinusoidal curves:

| Season | Fertility modifier | Wild food modifier |
|---|---|---|
| Winter (t=0) | 0.20 | 0.05 |
| Spring (t=90) | 0.65 | 0.71 |
| Summer (t=180) | 0.98 | 0.85 |
| Fall (t=270) | 0.35 | 0.09 |

Wild food collapses much harder in winter than farming does. This is intentional — an agent that farms and stores food survives winter; one that only hunts does not.

---

## Agent interface

The world exposes three methods for ML agents:

```python
world.get_atlas()              # (13, H, W) — static base map
world.get_obs(t)               # (14, H, W) — seasonal state at timestep t
world.get_window(t, y, x, r)  # (14, 2r+1, 2r+1) — local view centered on (y, x)
```

`get_obs` adds a 14th channel broadcasting `season_phase` (0.0–1.0) across the whole map so agents can learn seasonal patterns. `get_window` is the primary call for agents — they see a local patch of the world around their position, not the whole map.
