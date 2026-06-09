"""
Continuing-stream PPO for the Arlians multi-agent env (build-spec §3.2).

Key design decisions for the continuing stream:
  - done=True marks a DEATH (slot auto-reset via respawn_dead, not episode end).
  - GAE resets at done so no value estimate bleeds across a death boundary.
  - respawn_dead() is called after EVERY step to keep population alive (Phase 1-4).
  - The bootstrap value at horizon end uses the next observation (not zeroed).
  - All M slots are processed jointly; only alive-at-timestep entries enter the loss.

PPO hyperparameters (§3.2):
  clip_eps  = 0.2
  value_coef= 0.5
  ent_coef  = 0.01
  epochs    = 4
  lr        = 3e-4
  gamma     = 0.995
  lam       = 0.95
"""
from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
import torch
import torch.optim as optim

from sim.actions import build_mask, Action, N_PRIMARY
from sim.simulation import Simulation, Actions
from sim.metrics import _compute_specialization_index
from .policy import ArlianPolicy, DEVICE
from .rollout import SlotRolloutBuffer
from sim.observe import vector_len as compute_vector_len


class RolloutStats:
    """Cheap, fully-vectorized behavioral telemetry accumulated over a rollout.

    The mean-reward curve saturates once agents learn basic homeostasis and tells
    you nothing about *what* they do. This records the diagnostic signals — action
    mix, calorie source, settlement occupancy, births/deaths, lifespan — without the
    per-agent Python loops in MetricsLogger (which are too slow for the training hot
    loop). Call `record()` once per step; read `summary()` at the end of the rollout.
    """

    def __init__(self, sim: Simulation):
        self.fert  = sim.world.base_resources["soil_fertility"]
        self.water = sim.world.base_resources["water_proximity"]
        self.n_action = np.zeros(N_PRIMARY, dtype=np.int64)
        self.foraged = self.harvested = 0.0
        self.births = 0
        self.deaths: Dict[str, int] = {}
        self.on_fertile = self.on_water = 0
        self.agentsteps = 0
        self.energy_sum = self.hyd_sum = self.thermal_sum = self.age_sum = 0.0
        self.max_age = 0
        self.pop_sum = 0
        self.pop_steps = 0
        # --- emergence metrics (strategies that are NEVER rewarded; their rise over
        # training is the proof the innate-drive backbone produces civilization) ---
        self.built = 0                            # cumulative structures built this rollout
        self.struct_standing = 0                  # structures currently on the map (last step)
        self.struct_standing_max = 0              # peak standing structures
        M = sim.cfg.max_agents
        self.n_symbols = sim.cfg.n_symbols
        # per-slot action histogram (reset on death) -> specialization (division of labor)
        self._act_hist = np.zeros((M, N_PRIMARY), dtype=np.int32)
        # (signal, action) joint counts -> signal<->action mutual information (proto-language)
        self._sig_act = np.zeros((self.n_symbols, N_PRIMARY), dtype=np.int64)

    def record(self, sim: Simulation, actions: Actions, info: Dict[str, Any]) -> None:
        store = sim.store
        living = store.alive & ~store.is_predator
        idx = np.flatnonzero(living)
        n = idx.size
        self.pop_sum += n
        self.pop_steps += 1
        if n:
            self.n_action += np.bincount(actions.primary[idx], minlength=N_PRIMARY)
            ys, xs = store.y[idx], store.x[idx]
            self.on_fertile += int((self.fert[ys, xs]  >= 0.6).sum())
            self.on_water   += int((self.water[ys, xs] >= 0.6).sum())
            self.agentsteps += n
            self.energy_sum  += float(store.energy[idx].sum())
            self.hyd_sum     += float(store.hydration[idx].sum())
            self.thermal_sum += float(store.thermal[idx].sum())
            self.age_sum     += float(store.age[idx].sum())
            self.max_age = max(self.max_age, int(store.age[idx].max()))
            # division-of-labor: accumulate per-slot action counts. Zero dead/empty slots
            # first so a recycled slot's histogram reflects only its current occupant.
            self._act_hist[~living] = 0
            np.add.at(self._act_hist, (idx, actions.primary[idx]), 1)
            # proto-language: joint (emitted symbol, action) counts. last_signal holds this
            # step's emit (set inside step()), clipped to [0, n_symbols).
            sig = np.clip(store.last_signal[idx].astype(np.int64), 0, self.n_symbols - 1)
            np.add.at(self._sig_act, (sig, actions.primary[idx]), 1)
        cur_struct = int((sim.state.structure_type != 0).sum())
        self.struct_standing = cur_struct
        self.struct_standing_max = max(self.struct_standing_max, cur_struct)
        self.built     += int(info.get("built", 0))
        self.foraged   += float(info.get("foraged", 0.0))
        self.harvested += float(info.get("harvested", 0.0))
        self.births    += int(info.get("births", 0))
        for cause, k in info.get("deaths_by_cause", {}).items():
            self.deaths[cause] = self.deaths.get(cause, 0) + int(k)

    def _signal_action_mi_bits(self) -> float:
        """Mutual information (bits) between emitted signal and primary action, from the
        accumulated joint counts. >0 means signals carry action-predictive structure
        (emergent communication); ~0 means signals are random noise."""
        joint = self._sig_act.astype(np.float64)
        total = joint.sum()
        if total == 0:
            return 0.0
        joint /= total
        p_s = joint.sum(axis=1, keepdims=True)
        p_a = joint.sum(axis=0, keepdims=True)
        expected = p_s * p_a
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(
                (joint > 0) & (expected > 0),
                np.log2(joint / np.where(expected > 0, expected, 1.0)),
                0.0,
            )
        return max(0.0, float((joint * log_ratio).sum()))

    def summary(self) -> Dict[str, Any]:
        a = self.agentsteps or 1
        tot_food = self.foraged + self.harvested
        af = self.n_action / max(1, self.n_action.sum())
        # Emergence metrics computed once at rollout end (per-agent loops too slow per-step).
        hist = {
            i: self._act_hist[i]
            for i in np.flatnonzero(self._act_hist.sum(axis=1) > 0)
        }
        return {
            "pct_farmed":     float(self.harvested / tot_food) if tot_food > 0 else 0.0,
            "fertile_occ":    self.on_fertile / a,
            "water_occ":      self.on_water / a,
            "mean_energy":    self.energy_sum / a,
            "mean_hydration": self.hyd_sum / a,
            "mean_thermal":   self.thermal_sum / a,
            "mean_age":       self.age_sum / a,
            "max_age":        self.max_age,
            "births":         self.births,
            "deaths_total":   int(sum(self.deaths.values())),
            "deaths":         dict(self.deaths),
            "pop_mean":       self.pop_sum / max(1, self.pop_steps),
            "action_frac":    {Action(i).name: round(float(af[i]), 3)
                               for i in range(N_PRIMARY) if af[i] > 0.005},
            # emergent strategies (never rewarded — their rise = the backbone working)
            "structures_built":      int(self.built),
            "structures_standing":   int(self.struct_standing),
            "structures_peak":       int(self.struct_standing_max),
            "specialization_index":  round(_compute_specialization_index(hist), 4),
            "signal_action_mi":      round(self._signal_action_mi_bits(), 4),
        }


# ---- Hyperparameters (§3.2) ----
CLIP_EPS   = 0.2
VALUE_COEF = 0.5
ENT_COEF   = 0.01
EPOCHS     = 4
LR         = 3e-4
GAMMA      = 0.995
LAM        = 0.95
BATCH_SIZE = 256   # minibatch size for update step

# Bump when the reward objective changes incompatibly — guards against resuming an old
# checkpoint (value head trained on a different return distribution) onto the new reward.
REWARD_VERSION = 2


def reward_schedule(update_i: int, cfg) -> Dict[str, float]:
    """Anneal the SCAFFOLDING coefficients over training. The PERMANENT instinct terms
    (w_a / w_h / w_r_birth / w_r_surv) are deliberately NOT scheduled. Returns:
      beta_intr: intrinsic-novelty weight, intr_init -> 0 linearly over intr_anneal_updates.
      ent_coef:  PPO entropy bonus, ent_init -> ent_floor linearly over ent_anneal_updates.
    """
    bi = cfg.intr_init * (1.0 - update_i / max(1, cfg.intr_anneal_updates))
    beta_intr = float(min(cfg.intr_init, max(0.0, bi)))
    frac = min(1.0, update_i / max(1, cfg.ent_anneal_updates))
    ent_coef = float(max(cfg.ent_floor, cfg.ent_init - (cfg.ent_init - cfg.ent_floor) * frac))
    return {"beta_intr": beta_intr, "ent_coef": ent_coef}


def collect(
    sim: Simulation,
    policy: ArlianPolicy,
    T: int,
    seed_offset: int = 0,
    use_respawn: bool = True,
    stats: "RolloutStats | None" = None,
) -> SlotRolloutBuffer:
    """
    Roll out T environment steps, storing all (s, a, r, done, valid) in a buffer.

    Per step:
      1. Build primary_mask via build_mask (§3.1 action masking).
      2. policy.act(...) -> sample actions for all M slots.
      3. Assemble Actions, call sim.step(...).
      4. Store transition in buffer.
      5. Call sim.respawn_dead() so the population stays alive (Phase 1-4, §3.2).

    Bootstrap value is evaluated at the horizon end for non-terminal truncation.
    """
    cfg = sim.cfg
    M   = cfg.max_agents
    vlen = compute_vector_len(cfg)

    buffer = SlotRolloutBuffer(M=M, T=T, vector_len=vlen, n_symbols=cfg.n_symbols)

    # Get current observation (no reset — continuing stream)
    obs = sim.observe()

    for t in range(T):
        # Build primary action mask (M, N_PRIMARY) bool
        prim_mask = build_mask(sim.world, sim.state, sim.store, cfg)  # numpy bool

        # Sample actions
        act_out = policy.act(obs.spatial, obs.vector, prim_mask)

        # Convert to numpy for env
        primary_np = act_out["primary"].cpu().numpy().astype(np.int32)
        param_np   = act_out["param"].cpu().numpy().astype(np.int32)
        emit_np    = act_out["emit"].cpu().numpy().astype(np.int32)

        actions = Actions(primary=primary_np, param=param_np, emit=emit_np)

        # Step environment
        step_out = sim.step(actions)

        # Behavioral telemetry BEFORE respawn (so deaths/lifespan reflect reality)
        if stats is not None:
            stats.record(sim, actions, step_out.info)

        # Store transition
        # alive_mask BEFORE step (the obs we acted on)
        buffer.store(
            spatial      = obs.spatial,
            vector       = obs.vector,
            primary_mask = prim_mask,
            primary      = act_out["primary"],
            param        = act_out["param"],
            emit         = act_out["emit"],
            logp         = act_out["logp"],
            value        = act_out["value"],
            reward       = step_out.reward,
            done         = step_out.done,
            alive_mask   = obs.alive_mask,    # who was alive WHEN we acted
        )

        # Respawn dead agents so population stays trainable (Phase 1-4, §3.2).
        # Disable at Phase 5+ to let REPRODUCE sustain the population instead.
        if use_respawn:
            sim.respawn_dead(seed=seed_offset + t)

        # Advance obs
        obs = step_out.obs

        # Survival-sim early-break: once the whole cohort is dead (and we are NOT
        # topping the population back up), the rest of the horizon is an empty
        # world producing zero valid transitions — stop and save the compute.
        # compute_gae stays correct: the unfilled trailing steps keep
        # valid_mask=False and each slot's GAE already reset at its death boundary.
        if not use_respawn and sim.store.n_living_agents() == 0:
            break

    # ---- Bootstrap value at horizon end ----
    # For alive slots at the end: V(s_T). Dead slots: 0.
    prim_mask_final = build_mask(sim.world, sim.state, sim.store, cfg)
    with torch.no_grad():
        act_bootstrap = policy.act(obs.spatial, obs.vector, prim_mask_final)
    last_value = act_bootstrap["value"]  # (M,) on DEVICE

    # Zero out dead slots at bootstrap
    alive_t = torch.as_tensor(obs.alive_mask, dtype=torch.float32, device=DEVICE)
    last_value = last_value * alive_t

    # Compute GAE
    buffer.compute_gae(last_value, gamma=GAMMA, lam=LAM)

    return buffer


def update(
    buffer: SlotRolloutBuffer,
    policy: ArlianPolicy,
    optimizer: optim.Optimizer,
    ent_coef: float = ENT_COEF,
) -> Dict[str, float]:
    """
    PPO-clip update over the filled buffer (build-spec §3.2).

    Returns dict of mean metrics over all epochs:
      policy_loss, value_loss, entropy, total_loss
    """
    policy.train()

    policy_losses: List[float] = []
    value_losses:  List[float] = []
    entropies:     List[float] = []
    total_losses:  List[float] = []

    for _ in range(EPOCHS):
        for batch in buffer.minibatches(BATCH_SIZE):
            (
                spatial_b, vector_b, prim_mask_b,
                primary_b, param_b, emit_b,
                old_logp_b, old_value_b,
                returns_b, adv_b,
            ) = batch

            # Re-evaluate current policy on this batch
            logp, entropy, value = policy.evaluate(
                spatial_b, vector_b, prim_mask_b,
                primary_b, param_b, emit_b,
            )

            # ---- Policy (surrogate) loss ----
            ratio      = torch.exp(logp - old_logp_b)
            surr1      = ratio * adv_b
            surr2      = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- Value loss (clipped, §3.2) ----
            # Clip value prediction relative to old value estimate
            value_clipped = old_value_b + torch.clamp(
                value - old_value_b, -CLIP_EPS, CLIP_EPS
            )
            v_loss1 = (value - returns_b).pow(2)
            v_loss2 = (value_clipped - returns_b).pow(2)
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            # ---- Entropy bonus ----
            entropy_loss = -entropy.mean()

            # ---- Total loss ----
            loss = policy_loss + VALUE_COEF * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(-entropy_loss.item())
            total_losses.append(loss.item())

    policy.eval()

    if not policy_losses:
        # No valid entries in buffer (shouldn't happen with respawn but be safe)
        return {
            "policy_loss": 0.0,
            "value_loss":  0.0,
            "entropy":     0.0,
            "total_loss":  0.0,
        }

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss":  float(np.mean(value_losses)),
        "entropy":     float(np.mean(entropies)),
        "total_loss":  float(np.mean(total_losses)),
    }


def save_checkpoint(path: str, policy, optimizer, update_i: int, history: list) -> None:
    """Save policy + optimizer + progress so a disconnected run can resume."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "policy":    policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_i":  update_i,          # next update index to run
        "history":   history,
        "reward_version": REWARD_VERSION,
    }, path)


def load_checkpoint(path: str, policy, optimizer) -> tuple:
    """Restore from a checkpoint. Returns (start_update, history).

    Refuses to resume a checkpoint written under a different reward objective: the value
    head was trained to predict a different return distribution, so warm-starting it would
    inject large value-loss and corrupt advantages. Start a fresh run instead.
    """
    ckpt = torch.load(path, map_location=DEVICE)
    ckpt_ver = int(ckpt.get("reward_version", 1))
    if ckpt_ver != REWARD_VERSION:
        raise ValueError(
            f"checkpoint reward_version={ckpt_ver} != current {REWARD_VERSION}: the reward "
            f"objective changed (instinct-backbone overhaul). Start a FRESH run — do not "
            f"resume old weights onto the new objective."
        )
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["update_i"]), list(ckpt.get("history", []))


def train(
    sim: Simulation,
    policy: ArlianPolicy,
    n_updates: int,
    T: int,
    *,
    use_respawn: bool = True,
    reset_floor: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
    resume: bool = False,
    log_fn=None,
) -> List[Dict[str, Any]]:
    """
    Main PPO training loop (§3.2, §5).

    Runs n_updates rounds of collect(T steps) + update(buffer).
    Returns a list of per-update metric dicts:
      mean_reward, policy_loss, value_loss, entropy, n_living, total_loss

    Args:
        sim, policy, n_updates, T: as before.
        use_respawn:     pass-through to collect() — keep True for Phases 1-4, set
                         False at Phase 5+ (or survival-sim mode) so the population is
                         not artificially topped up and lifespan reflects real skill.
        reset_floor:     survival-sim mode. When > 0 (and use_respawn is False), the
                         cohort is left to decline naturally during a rollout; if the
                         living population drops below this floor at a rollout boundary,
                         a fresh cohort is re-seeded. This keeps training alive without
                         the respawn crutch, so mean/max lifespan becomes the signal
                         that rises as the policy learns to survive. 0 disables it.
        checkpoint_path: if set, save a resumable checkpoint here every
                         checkpoint_every updates and at the end (survives Kaggle/Colab
                         disconnects).
        checkpoint_every:update interval between checkpoint saves.
        resume:          if True and checkpoint_path exists, restore and continue.
        log_fn:          optional callback(row: dict) invoked after each update (for
                         live logging / TensorBoard); if None, nothing extra is logged.
    """
    import os
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    start_update = 0
    metrics_history: List[Dict[str, Any]] = []

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        start_update, metrics_history = load_checkpoint(checkpoint_path, policy, optimizer)
        print(f"[train] resumed from {checkpoint_path} at update {start_update}")
    else:
        # Ensure env starts fresh only on a brand-new run
        _ = sim.reset(seed=0)

    for update_i in range(start_update, n_updates):
        # Survival-sim mode: re-seed a fresh cohort if the population has collapsed
        # below the floor (replaces the respawn crutch). Checked at the rollout
        # boundary so GAE/done-masking inside a rollout stays clean.
        if reset_floor > 0 and not use_respawn:
            n_now = int(sim.store.n_living_agents())
            if n_now < reset_floor:
                _ = sim.reset(seed=1_000_000 + update_i)
                print(
                    f"[train] cohort collapsed ({n_now} < floor {reset_floor}); "
                    f"re-seeded fresh cohort at update {update_i}",
                    flush=True,
                )

        # Anneal the scaffolding coefficients for this update. beta_intr is read by
        # sim.step() (mutable attribute — avoids touching the frozen step() signature);
        # ent_coef is threaded into the PPO update below.
        sched = reward_schedule(update_i, sim.cfg)
        sim._beta_intr = sched["beta_intr"]

        print(
            f"[train] {update_i + 1}/{n_updates} collecting rollout ({T} steps)...",
            flush=True,
        )
        stats = RolloutStats(sim)
        buffer = collect(sim, policy, T, seed_offset=update_i * T,
                         use_respawn=use_respawn, stats=stats)

        # Mean reward across valid (alive) timesteps
        valid_rewards = buffer.reward[buffer.valid_mask]
        mean_reward = valid_rewards.mean().item() if valid_rewards.numel() > 0 else 0.0

        # Mean return (episodic-ish, per valid timestep)
        valid_returns = buffer.returns[buffer.valid_mask]
        mean_return = valid_returns.mean().item() if valid_returns.numel() > 0 else 0.0

        n_living = int(sim.store.n_living_agents())

        # ---- update ----
        loss_dict = update(buffer, policy, optimizer, ent_coef=sched["ent_coef"])

        row: Dict[str, Any] = {
            "update":       update_i,
            "mean_reward":  mean_reward,
            "mean_return":  mean_return,
            "n_living":     n_living,
            **loss_dict,
            "behavior":     stats.summary(),
        }
        metrics_history.append(row)
        if log_fn is not None:
            log_fn(row)

        # ---- checkpoint ----
        if checkpoint_path and ((update_i + 1) % checkpoint_every == 0):
            save_checkpoint(checkpoint_path, policy, optimizer, update_i + 1, metrics_history)

    if checkpoint_path:
        save_checkpoint(checkpoint_path, policy, optimizer, n_updates, metrics_history)

    return metrics_history
