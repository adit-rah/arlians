# Arlians live stream

Persistent demo server: runs the sim + policy, pushes **sparse tile/entity deltas** each step. The browser draws a biome basemap once, then patches changed cells.

Core sim code is unchanged; everything lives under `stream/`.

## Prerequisites

From repo root:

```bash
pip install -r requirements.txt -r stream/requirements.txt
python generate.py --seed 42 --width 256 --height 256 --output ./data/world_42
```

Optional trained policy:

```bash
export CHECKPOINT_PATH=runs/ckpt.pt
```

## Run locally

```bash
export WORLD_DIR=./data/world_42
export INIT_AGENTS=96
export MAX_AGENTS=192
export STEPS_PER_SEC=8

uvicorn stream.server.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_DIR` | `data/world_42` | Exported world from `generate.py` |
| `CHECKPOINT_PATH` | (none) | `.pt` checkpoint; random policy if unset |
| `INIT_AGENTS` | `96` | Starting population |
| `MAX_AGENTS` | `192` | Entity slot cap |
| `NO_RESPAWN` | `false` | Set `true` to disable `respawn_dead` |
| `STEPS_PER_SEC` | `8` | Sim steps per second while clients connected |
| `RESET_SEED` | `0` | `Simulation.reset` seed |

## API (protocol v1)

See [protocol.md](protocol.md).

- `GET /api/v1/bootstrap` — static biome + palette
- `GET /api/v1/health`
- `GET /api/v1/delta?after={seq}` — poll fallback
- `WebSocket /api/v1/stream` — `StepDelta` messages
- `POST /api/v1/control` — `pause` | `play` | `reset`

## Tests

```bash
pytest stream/tests/ -q
```

## Deploy notes

- CPU is fine at demo scale (256², ~96 agents).
- Use a GPU only if PyTorch supports your host (T4/L4 OK; P100 + current PyTorch may fail).
- Single process; scale-out = multiple worlds/checkpoints later (protocol v2).

## Layout

```
stream/
  protocol.md
  shared/palette.json
  server/     # FastAPI + SimEngine + delta
  client/     # static canvas UI
  tests/
```
