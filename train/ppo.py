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

from sim.actions import build_mask
from sim.simulation import Simulation, Actions
from .policy import ArlianPolicy, DEVICE
from .rollout import SlotRolloutBuffer
from sim.observe import vector_len as compute_vector_len

# ---- Hyperparameters (§3.2) ----
CLIP_EPS   = 0.2
VALUE_COEF = 0.5
ENT_COEF   = 0.01
EPOCHS     = 4
LR         = 3e-4
GAMMA      = 0.995
LAM        = 0.95
BATCH_SIZE = 256   # minibatch size for update step


def collect(
    sim: Simulation,
    policy: ArlianPolicy,
    T: int,
    seed_offset: int = 0,
    use_respawn: bool = True,
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
            loss = policy_loss + VALUE_COEF * value_loss + ENT_COEF * entropy_loss

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
    }, path)


def load_checkpoint(path: str, policy, optimizer) -> tuple:
    """Restore from a checkpoint. Returns (start_update, history)."""
    ckpt = torch.load(path, map_location=DEVICE)
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
                         False at Phase 5+ to let reproduction sustain the population.
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
        print(
            f"[train] {update_i + 1}/{n_updates} collecting rollout ({T} steps)...",
            flush=True,
        )
        buffer = collect(sim, policy, T, seed_offset=update_i * T, use_respawn=use_respawn)

        # Mean reward across valid (alive) timesteps
        valid_rewards = buffer.reward[buffer.valid_mask]
        mean_reward = valid_rewards.mean().item() if valid_rewards.numel() > 0 else 0.0

        # Mean return (episodic-ish, per valid timestep)
        valid_returns = buffer.returns[buffer.valid_mask]
        mean_return = valid_returns.mean().item() if valid_returns.numel() > 0 else 0.0

        n_living = int(sim.store.n_living_agents())

        # ---- update ----
        loss_dict = update(buffer, policy, optimizer)

        row: Dict[str, Any] = {
            "update":       update_i,
            "mean_reward":  mean_reward,
            "mean_return":  mean_return,
            "n_living":     n_living,
            **loss_dict,
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
