# Stream protocol v1

Source of truth for the live demo backend (`stream/server/`) and client (`stream/client/`).

## Bootstrap — `GET /api/v1/bootstrap`

```json
{
  "protocolVersion": 1,
  "H": 256,
  "W": 256,
  "seed": 42,
  "biome": "<base64: H*W int8 row-major>",
  "palette": { "biomes": [[r,g,b], ...], "agent": [255,80,220], ... }
}
```

- `biome`: one byte per tile, biome index 0–9.
- `palette`: matches `sim/render.py` colors.

## Step delta — WebSocket message type `StepDelta`

```json
{
  "type": "StepDelta",
  "protocolVersion": 1,
  "seq": 1,
  "t": 1,
  "seasonPhase": 0.25,
  "nLiving": 96,
  "tiles": [
    { "y": 10, "x": 20, "layers": { "wild": 0.4, "cropStage": 0.01, "cropHealth": 1.0 } }
  ],
  "entities": [
    { "slot": 0, "op": "upsert", "y": 10, "x": 20, "isPredator": false },
    { "slot": 3, "op": "remove" }
  ]
}
```

### Tile `layers` (all optional; omit unchanged fields)

| Field | Type | Meaning |
|-------|------|---------|
| `wild` | float | `wild_remaining` |
| `cropStage` | float | 0 = no crop |
| `cropHealth` | float | |
| `structureType` | int | 0 none, 1 shelter, 2 storage, 3 wall |
| `storedFood` | float | storage tile contents |
| `eventMask` | float | catastrophe overlay |

### Entity ops

| `op` | Fields |
|------|--------|
| `upsert` | `slot`, `y`, `x`, `isPredator` |
| `remove` | `slot` only |

## Control — `POST /api/v1/control`

```json
{ "action": "pause" | "play" | "reset" }
```

## Health — `GET /api/v1/health`

```json
{ "running": true, "paused": false, "t": 120, "nLiving": 94, "seq": 120 }
```

## Poll fallback — `GET /api/v1/delta?after={seq}`

Returns latest `StepDelta` with `seq > after`, or 204 if none.
