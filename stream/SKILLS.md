# stream/ — live browser visualization (optional add-on)

A persistent demo server that runs the sim + policy and pushes **sparse tile/entity deltas**
each step to a browser, which draws the biome basemap once and patches changed cells. This is
a presentation layer — **the core sim/world/train code is unchanged and unaware of it.**

If you're working on agent behavior or training, you almost certainly don't need to touch
this. It's for showing the simulation live.

## Orientation

- `README.md` — setup + how to run locally (start here).
- `protocol.md` — the wire protocol for the basemap + per-step deltas.
- `server/` — runs the sim loop and emits deltas. `client/` — browser renderer.
  `shared/` — protocol types shared by both. `tests/` — stream-specific tests.
- `requirements.txt` — extra deps beyond the repo root `requirements.txt`.

## Run (summary — see README.md for the full version)

```bash
pip install -r requirements.txt -r stream/requirements.txt
.venv/bin/python generate.py --seed 42 --width 256 --height 256 --output ./data/world_42
export WORLD_DIR=./data/world_42
# optional: export CHECKPOINT_PATH=runs/ckpt.pt   (else random policy)
# then start the server per README.md
```

## Gotchas

- It loads a world from `WORLD_DIR` (an exported `data/world_<seed>/`) — generate one first.
- With no `CHECKPOINT_PATH` it runs a random policy (expect aimless/teleporting agents — the
  same respawn artifact noted in the top-level `SKILLS.md`).
- Keep protocol changes in lockstep across `shared/` ↔ `server/` ↔ `client/` and `protocol.md`.
