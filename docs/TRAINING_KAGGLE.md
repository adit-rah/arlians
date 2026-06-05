# Training Arlians on Kaggle Notebooks — a beginner's walkthrough

This guide takes you from zero to a running training job on a **free Kaggle GPU**,
explaining every step. No prior Kaggle experience assumed.

## What you're about to do (the mental model)

A **Kaggle Notebook** is a free, hosted Jupyter notebook (cells of Python you run in the
browser) that Kaggle backs with a real machine — including a **free GPU** (~30 hours/week,
sessions up to ~12 hours). You'll:

1. Put this repo somewhere Kaggle can fetch it (GitHub — one-time).
2. Create a notebook, turn on the GPU.
3. Run cells that: clone the repo → install deps → generate the world → **train**, saving
   **checkpoints** so a disconnect never loses progress.
4. Resume across sessions until the behavioral metrics move, then download the trained
   policy and visualize it.

**Reality check before you start:** this is research, not a 10-minute job. Emergence
(farming, winter survival, cooperation) may take many GPU-hours and parameter tuning.
Also, our environment is **CPU-bound** (numpy stepping), so the GPU won't be maxed out —
that's expected; throughput is modest. The point of Kaggle is *free, persistent, hands-off*
compute, which checkpointing makes usable.

---

## Step 1 — Put the code on GitHub (one-time)

Kaggle needs to fetch your code. The cleanest way is a GitHub repo it can `git clone`.

1. Create a free account at <https://github.com> if you don't have one.
2. Make a new **empty** repository: github.com → **+** (top right) → **New repository** →
   name it e.g. `arlians` → **Create repository**. (Private is fine; see the token note
   below if you keep it private.)
3. On your machine, from the repo root, point your local repo at it and push the branch:
   ```bash
   git remote add origin https://github.com/<your-username>/arlians.git
   git push -u origin build/sim-layer
   ```
   (We've been working on the `build/sim-layer` branch — pushing it is enough; no need to
   merge to `main`.)

> **Public vs private:** if the repo is **public**, Kaggle can clone it with no auth. If
> **private**, you'll need a GitHub *personal access token* (github.com → Settings →
> Developer settings → Personal access tokens) and clone with
> `https://<TOKEN>@github.com/<you>/arlians.git`. For a first run, public is simplest.

> **No-GitHub alternative:** zip the repo (`world/ sim/ train/ tests/ scripts/ generate.py
> requirements.txt`), and on Kaggle use **+ Add Input → Upload → Dataset** to upload the
> zip. Your code then lives at `/kaggle/input/<dataset-name>/`. Works, but you must
> re-upload on every code change — GitHub is nicer for iteration.

---

## Step 2 — Create the Kaggle notebook + turn on the GPU

1. Sign up at <https://www.kaggle.com>. **Verify your phone number** (Settings → Phone) —
   Kaggle requires this to unlock GPU/internet access. Without it, the GPU and `git clone`
   won't work.
2. Top-left **Create → New Notebook**. You get a blank notebook with one code cell.
3. Open the right-hand panel (**⋮** or the "Notebook options" / settings sidebar):
   - **Accelerator** → choose **GPU T4 x2** (or **GPU P100**). This is the free GPU.
   - **Internet** → **On** (needed to clone the repo and pip-install).
   - Leave **Persistence** at its default; we rely on checkpoints + version commits instead.

The notebook's working directory is **`/kaggle/working/`** — files you write there are kept
when you "Save Version". `/kaggle/input/` is read-only inputs (datasets).

---

## Step 3 — The notebook cells

Paste each block into its own cell and run them top to bottom (Shift+Enter).

**Cell 1 — confirm the GPU is there**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```
You should see a Tesla T4 (or P100). If it says CUDA not available, the accelerator isn't
set (Step 2).

**Cell 2 — get the code**
```python
# public repo:
!git clone -b build/sim-layer https://github.com/<your-username>/arlians.git
%cd arlians
!git log --oneline -1
```
(For a private repo use the token URL from Step 1. For the zip/dataset path, instead do
`%cd /kaggle/input/<dataset-name>` and skip the clone.)

**Cell 3 — install dependencies**
```python
# torch is already on Kaggle; install the rest. (imageio is usually preinstalled.)
!pip install -q "numpy>=1.26" "scipy>=1.12" "opensimplex>=0.4" imageio
```
Our `world/` + `sim/` are pure numpy/scipy; `train/` uses the preinstalled torch (it will
automatically use CUDA on Kaggle — `train/policy.py` picks CUDA → MPS → CPU).

**Cell 4 — generate the world** (once; it's deterministic per seed)
```python
!python generate.py --seed 42
```

**Cell 5 — train, with checkpointing**
```python
# Checkpoint to /kaggle/working so it survives into the saved version.
# Start MODEST, scale up once you confirm it runs. --resume makes re-runs continue.
!python train/run.py \
    --width 256 --agents 4096 --init-agents 2000 \
    --updates 3000 --horizon 128 \
    --checkpoint /kaggle/working/ckpt.pt --checkpoint-every 20 \
    --log /kaggle/working/metrics.jsonl --resume
```
This prints a live curve (reward / value-loss / entropy / living population) and writes
`ckpt.pt` (resumable) + `metrics.jsonl` every few updates.

> **First time, go small** to confirm everything works before committing GPU-hours, e.g.
> `--width 128 --agents 512 --init-agents 256 --updates 50 --horizon 64`. When the curve
> looks sane, bump to the full-scale command above.

> **Phase 5+ note:** by default training uses `respawn_dead` to keep the population alive
> (the Phase 1–4 crutch). Once you want *reproduction* to sustain the population, add
> `--no-respawn`. Expect this to be harder (Malthusian collapse risk — see FINDINGS.md).

---

## Step 4 — Run it hands-off + survive disconnects

Two ways to actually accumulate hours:

**A. Interactive (you watch):** just run Cell 5. If the browser tab closes or the session
hits its limit, the checkpoint is on disk up to the last save. **Caveat:** an *interactive*
session's `/kaggle/working` is reset when you start a fresh session — so to keep the
checkpoint across sessions, use method B, or "Save Version" before the session ends.

**B. Background run (recommended for long jobs):** top-right **Save Version → Save & Run
All (Commit)**. Kaggle runs the whole notebook to completion **in the background** (up to
~12h) even with your browser closed, and the committed version's `/kaggle/working` outputs
(your `ckpt.pt`, `metrics.jsonl`) are **saved permanently** with that version. This is the
proper way to bank GPU-hours.

**Resuming across sessions / 12h limit.** A long training run spans multiple sessions.
The pattern:
1. After a committed run finishes, open the notebook's **Output** tab — your `ckpt.pt` is
   there. Click **… → Add as Data source** (this turns the output into a dataset input).
2. In the next run, add that dataset, and copy the checkpoint in before training:
   ```python
   !cp /kaggle/input/<your-output-dataset>/ckpt.pt /kaggle/working/ckpt.pt
   ```
   Then run Cell 5 with `--resume` — it picks up exactly where it left off.

Repeat until the metrics plateau or the behavior emerges. Each weekly 30h of GPU chips
away at it.

---

## Step 5 — Watch the right numbers

Training "working" ≠ "civilization emerged". Watch two layers:

- **Learning is healthy** (from Cell 5's live log): `value_loss` falling, `mean_return`
  rising, `entropy` slowly declining, `living` population not collapsing.
- **Behavior is emerging** (the real goal — from `metrics.jsonl` / a periodic eval): the
  Section-H signals climbing over training:
  `pct_calories_farmed` ↑, `water_occupancy`/`fertile_occupancy` ↑, `mean_thermal` ↑ in
  winter, `winter survival` ↑, `specialization_index` ↑, `signal_action_mi` ↑ (language
  being used). Compare against the **untrained baseline** (`pct_farmed≈0.13`,
  `specialization≈0.056`, `MI≈0.007` — see FINDINGS/demo).

---

## Step 6 — Visualize the trained policy

Once you have a `ckpt.pt`, render it the same way as the untrained baseline — the demo
script accepts a checkpoint:
```python
!python scripts/demo.py --checkpoint /kaggle/working/ckpt.pt \
    --steps 360 --agents 800 --size 192 --out /kaggle/working/trained.gif
```
Download `trained.gif` from the **Output** panel and compare it to a baseline GIF from
`python scripts/demo.py` (no `--checkpoint`). If
training worked, you'll see agents clustering on fertile/water tiles, farming, sheltering —
instead of milling around randomly.

---

## Gotchas & tips

- **Phone-verify your Kaggle account** or you get no GPU/internet.
- **GPU quota ~30h/week**, sessions ~9–12h. Plan around checkpoints; don't expect one
  unbroken run.
- **It's CPU-bound.** Kaggle gives ~4 CPU cores; our numpy env stepping (not the GPU) is
  the throughput limiter. The biggest speedups would come from vectorizing the env's
  per-slot Python loops (`forage`/`build`/`reproduce`) — a worthwhile follow-up if
  throughput frustrates you (see BUILD_STATUS.md follow-up #3).
- **Always checkpoint to `/kaggle/working`** and use **Save & Run All (Commit)** for long
  runs, or you'll lose progress on disconnect.
- **Start small, then scale.** Confirm the stack runs at toy scale (50 updates) before
  spending hours at full scale.
- **Cheap paid fallback** if free runs out: vast.ai / RunPod (~$0.2–0.5/hr for a T4) — same
  `train/run.py --checkpoint --resume` flow on a rented box.

---

## TL;DR cheat-sheet

```text
1. git push your repo to GitHub (branch build/sim-layer)
2. Kaggle → verify phone → New Notebook → Accelerator: GPU T4x2, Internet: On
3. Cells: clone repo → pip install → python generate.py --seed 42
4. python train/run.py --width 256 --agents 4096 --init-agents 2000 \
       --updates 3000 --horizon 128 \
       --checkpoint /kaggle/working/ckpt.pt --log /kaggle/working/metrics.jsonl --resume
5. Save Version → Save & Run All (Commit)  (background, banks GPU-hours)
6. Re-add the output ckpt.pt as a dataset next session, --resume, repeat
7. Eval: python scripts/demo.py --checkpoint /kaggle/working/ckpt.pt
```
