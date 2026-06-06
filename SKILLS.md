# Arlians — repo orientation for agents

You are working on **Arlians**: a procedurally-generated tile world plus a multi-agent
reinforcement-learning simulation whose goal is *emergent* semi-civilization (foraging,
farming, sheltering, storing, reproducing, defending) — never rewarded directly, only
homeostasis + survival.

Read this first, then the `SKILLS.md` in whatever subdirectory you're touching.

## The one-paragraph mental model

Three layers, strictly stacked:

```
world/   read-only substrate   — geography + base resources + season scalars (never mutates)
   │     (World.get_obs(t) -> (14,H,W); base_resources dict; deterministic per seed)
   ▼
sim/     the living environment — mutable per-tile state + entities + step(actions)->obs
   │     (numpy only, no torch; PettingZoo-style Simulation facade)
   ▼
train/   the learner            — shared-weights PPO over the continuing stream (torch)
```

`world/` is finished and frozen. `sim/` + `train/` implement the agent layer (Phases 0–7
of the build, all coded). The current focus is **getting useful behavior out of training**,
not adding mechanics.

## Frozen contracts (the anti-divergence rule — read before editing sim/ or train/)

These signatures/schemas are a **frozen project-wide contract**. Do **not** rename, remove,
or reshape them without an explicit re-lock — many modules and the saved checkpoints depend
on them. You may only start *using* a forward-declared field at its phase.

- `sim/config.py` — `SimConfig` field set (every tunable).
- `sim/state.py` — `WorldState` grid arrays + `EntityStore` SoA fields.
- `sim/actions.py` — `Action` enum (14 primaries), `N_PARAM=8`, `build_mask` signature.
- `sim/observe.py` — 24 spatial channels + 16-long interoceptive vector layout.
- `sim/simulation.py` — `Obs` / `Actions` / `StepOut` shapes and `Simulation` methods.

If a contract looks *wrong*, STOP and flag it rather than working around it.

## Current state (2026-06, survival-sim reframe)

- All Phase 0–7 mechanics are coded and unit-tested. **Behavioral gates are not met** —
  this is research, not a finished result.
- Training runs on a free Kaggle GPU (see `notebooks/kaggle_training.ipynb`,
  `docs/TRAINING_KAGGLE.md`). The env is **CPU-bound** (numpy stepping), so the GPU is
  not the bottleneck; throughput is modest.
- The project was just **reframed from "grow a civilization" to "survival sim"**: accept
  homeostasis-only behavior, turn off the respawn crutch, and make *lifespan* (`max_age`/
  `mean_age` in the new telemetry) the success signal that should rise with training.

## Hard-won lessons (don't relearn these the slow way)

1. **`mean_reward` saturates and is nearly useless.** Reward = `comfort + 0.05`; the best
   an agent can do is sit near its setpoints (~0.6). It plateaus once basic homeostasis is
   learned (~40 updates) and tells you nothing about behavior. Watch the `behavior` block
   (action mix, `pct_farmed`, occupancy, lifespan, deaths-by-cause) instead.
2. **Respawn masks everything.** `respawn_dead` (Phase 1–4 crutch) tops the population back
   up every step at *random* tiles, so you can't observe survival, reproduction never pays
   (it's reward-negative and pointless when the pop is maintained for free), and rendered
   GIFs look like agents *teleporting* (a dead slot's dot reappears across the map). Use
   `--no-respawn` for any honest survival/lifespan/visualization.
3. **Farming will not emerge from scratch** under the current reward + stricter planting
   gates (PLANT costs energy *now* for a payoff ~50 steps later, via a long random action
   chain). The design's curriculum scaffold (pre-seed mature crops to teach
   harvest→eat→comfort) was deferred and never built. Don't expect `pct_farmed > 0` without it.
4. **The world is deliberately starvation-biased toward farming.** Wild food is sparsified
   to a tiny fraction of tiles to *force* farming. If you drop farming (survival-sim), pure
   foraging is brutal — you may need to loosen the food economy (`energy_decay`,
   `forage_regen`, `forage_yield`) for survival to be learnable.
5. **Tune from `deaths_by_cause`, one knob at a time.** Don't blind-tune. The dominant
   death cause names the drive to fix; at random play, *dehydration* can outkill starvation.

## How to run things (use the venv interpreter)

```bash
.venv/bin/python generate.py --seed 42 --width 256 --height 256   # build a world -> data/world_42/
.venv/bin/python train/run.py --width 256 --agents 512 --init-agents 256 --updates 50 --no-respawn
.venv/bin/python scripts/demo.py --steps 360 --render-every 1 --no-respawn   # GIF -> data/demo.gif
.venv/bin/python -m pytest -q                                     # full suite
```

## Conventions

- numpy `float32`; grid arrays `(H,W)`; entity arrays `(M,)` or `(M,k)` with `M=cfg.max_agents`.
- **No magic numbers** — everything flows from `SimConfig`.
- `sim/` imports no torch; `train/` may.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- The Kaggle notebook clones the **`main`** branch — code changes only reach Kaggle once
  pushed to `main`. The notebook's *cells* (config values, commands) live on Kaggle, not
  in the repo, so editing `notebooks/kaggle_training.ipynb` does not change a running
  Kaggle notebook — it must be re-imported or hand-synced.

## Where the design lives

`WORLD.md` (world design), `docs/BUILD_STATUS.md` (build/gate status), `docs/FINDINGS.md`
(simulation insights), `docs/TRAINING_KAGGLE.md` (training walkthrough).

> Note: these are `SKILLS.md` files, which agents must read explicitly. If you want a file
> auto-loaded into context, that's `CLAUDE.md` — consider symlinking this one if desired.
