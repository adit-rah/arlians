# notebooks/

## `kaggle_training.ipynb` — train on a free Kaggle GPU

Self-contained notebook: clones the repo (branch **`main`**), installs deps, generates the
world, trains with checkpointing, and renders a demo GIF. Full prose walkthrough lives in
`docs/TRAINING_KAGGLE.md`.

### The two things that confuse people

1. **The notebook is a separate entity from this repo.** The *code* (`train/run.py`,
   `scripts/demo.py`, …) updates automatically because the notebook `git clone`s `main`. But
   the notebook's *cells* (the `Configuration` cell, the train/demo commands) live on Kaggle —
   editing `notebooks/kaggle_training.ipynb` here does NOT change a running Kaggle notebook.
   Sync via **File → Import Notebook** or hand-edit the cells on Kaggle. So: push code to
   `main` for code changes; re-import the notebook for cell changes.
2. **`/kaggle/working` is wiped between sessions.** To resume, the previous run's checkpoint
   must be re-attached as a **dataset input** (Add Input), then `RESUME_FROM_INPUT=True`. The
   restore cell auto-discovers any `*.pt` under `/kaggle/input` and prints what it found, so an
   exact-path mismatch (wrong filename / nested folder / dataset created-but-not-attached) is
   self-diagnosing.

### Current defaults

- `NO_RESPAWN = True` (survival-sim reframe — lifespan is the signal).
- Demo cell uses `--render-every 1 --no-respawn` (no teleporting).
- Requires phone-verified Kaggle account for GPU + internet; `GPU T4 x2`, Internet On.
- Watch the per-update `behavior` line (action mix, `pct_farmed`, `max_age`, deaths), not
  `mean_reward` (it saturates).
