# docs/ — project documentation index

Authoritative prose docs. Read these for *why*, not just *how*.

- `BUILD_STATUS.md` — what's built, per-phase CODE-gate results, test inventory, how-to-run,
  and which behavioral gates are deferred. Start here for "is X done?".
- `FINDINGS.md` — simulation insights + process findings (the empirical lessons behind the
  top-level `SKILLS.md` gotchas: farming won't emerge without scaffolding, Malthusian
  collapse is real, scripted farmer ≫ forager, etc.).
- `TRAINING_KAGGLE.md` — full beginner walkthrough for training on a free Kaggle GPU
  (companion to `notebooks/kaggle_training.ipynb`).

Also relevant, at the repo root: `WORLD.md` (world generator design).

When you learn something durable about the sim's behavior or the build, record it here
(FINDINGS.md) rather than only in a commit message — that's how the next agent finds it.
