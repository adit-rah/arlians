# train/ — the learner (PyTorch)

Shared-weights, continuing-stream PPO over the `sim/` environment. This is the only layer
that imports torch. It talks to the env solely through the `Simulation` facade and the frozen
obs/action contracts.

## Files

- `policy.py` — `ArlianPolicy`: one shared actor-critic for ALL `M` slots (and all
  generations — a child runs the same net). Spatial CNN (24→32→64→64, 3×3 same-pad over
  15×15) → FC256; vector FC64 over the 16-long interoceptive vector; trunk FC256; four heads:
  `primary(14)`, `param(8)`, `emit(n_symbols)`, `value(1)`. Optional GRU (off by default).
  `act()` (sampling, no-grad) and `evaluate()` (for the surrogate loss). The primary mask is
  added as `-inf` to logits before sampling; an all-masked row falls back to NOOP to avoid
  NaNs. `DEVICE` = CUDA → MPS → CPU. **Input dims (N_SPATIAL=24, WINDOW=15, vector=16) are
  hard-wired from the frozen sim contract** — if you change a sim channel count, update here
  and old checkpoints break.
- `rollout.py` — `SlotRolloutBuffer`: pre-allocated `(M,T)` tensors on DEVICE. GAE per slot
  with `(1-done)` reset at death boundaries (continuing stream, NOT episodic); dead/never-alive
  timesteps masked out of the loss via `valid_mask`. Minibatches iterate valid `(slot,time)`
  pairs only — so longer-lived agents contribute more gradient (population-weighted learning).
- `ppo.py` — `collect()` (roll out T steps, store transitions, bootstrap, GAE), `update()`
  (PPO-clip, clipped value loss, entropy bonus, grad-norm clip), `train()` (loop +
  checkpoint/resume). Hypers at top: clip 0.2, value 0.5, ent 0.01, epochs 4, lr 3e-4,
  γ 0.995, λ 0.95, minibatch 256. Also `save_checkpoint`/`load_checkpoint`.
  - **`RolloutStats`** (in this file): cheap fully-vectorized per-rollout telemetry — action
    mix, `pct_farmed`, fertile/water occupancy, mean drives, **mean/max lifespan**, births,
    deaths-by-cause. This is the diagnostic you actually read; `mean_reward` is near-useless
    (it saturates ~0.6). Logged into each metrics row under `"behavior"` and printed live.
  - **Survival-sim mode**: `train(..., use_respawn=False, reset_floor=N)` lets the cohort
    decline naturally and re-seeds a fresh cohort only when it collapses below `N` (so
    training never stalls at zero agents). `max_age`/`mean_age` is the signal that should rise.
- `run.py` — CLI entrypoint. Flags: `--width/--height/--seed/--agents/--init-agents/--updates/
  --horizon`, `--no-respawn`, `--reset-floor` (defaults to `init_agents//4` when `--no-respawn`),
  `--checkpoint/--checkpoint-every/--resume`, `--log` (JSONL). Prints a per-update line plus a
  `behavior` line.

## Continuing-stream PPO model (the key idea)

The population is `M` parallel auto-resetting sub-envs (one per slot). A **death** = `done=True`
(GAE resets, bootstrap 0); a **birth/respawn/reset** = a fresh trajectory begins in that slot;
an **empty slot** is masked out. This collapses "continuing world with births/deaths" onto
standard vectorized PPO. There is no episode boundary.

## Gotchas

- **Resume needs the population seeded.** On `--resume` the sim is freshly constructed (empty);
  with respawn it's topped up, and with `--no-respawn` the `reset_floor` check re-seeds it.
  Don't add a `sim.reset()` on the resume path — it would discard the continuing stream.
- **CPU-bound.** The bottleneck is numpy env stepping, not the GPU. Throughput ~modest
  (~0.03 upd/s at 512²/256 horizon on a T4). Vectorizing the remaining per-slot Python loops
  in `sim/dynamics.py` is the highest-leverage speedup if you need it.
- **Checkpoints store `update_i` + history**; `--resume` continues from there. Changing the
  policy architecture (or sim channel counts) invalidates old checkpoints.
- Keep `gif fps` out of imageio — use `duration=` (ms); `fps` silently writes 1-frame GIFs.

## Run / test

```bash
.venv/bin/python train/run.py --width 96 --agents 128 --init-agents 64 --updates 4 --horizon 32 --no-respawn
.venv/bin/python -m pytest tests/test_train.py -q
```
