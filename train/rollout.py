"""
SlotRolloutBuffer — fixed (M, T) storage for continuing-stream PPO (build-spec §3.2).

Key design decisions:
  - Storage is pre-allocated as float32 tensors on DEVICE (MPS/CPU).
  - GAE is computed per slot independently; advantage accumulation RESETS at done=True
    (death boundary) — the continuing stream is not episodic so we must not bleed
    value estimates across a death transition.
  - Dead / never-alive timesteps are masked from all loss computations via `valid_mask`.
  - Minibatches iterate over VALID (alive at that timestep) entries only.

GAE formula per slot s, timestep t (from horizon end T-1 back to 0):
  delta_t   = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
  A_t       = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
  returns_t = A_t + V_t
The (1 - done_t) factor zeroes the bootstrap at death so value leakage is impossible.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch

from .policy import DEVICE, N_PRIMARY

# Type alias for clarity
Tensor = torch.Tensor


class SlotRolloutBuffer:
    """
    Pre-allocated (M, T) buffer for one rollout horizon (build-spec §3.2).

    Slots that are dead at a given timestep are excluded from loss computation
    via the `valid_mask` field.
    """

    def __init__(self, M: int, T: int, vector_len: int, n_symbols: int) -> None:
        """
        Args:
            M:          number of agent slots (cfg.max_agents)
            T:          rollout horizon in steps
            vector_len: interoceptive vector length (sim/observe.vector_len)
            n_symbols:  emit vocabulary size (cfg.n_symbols)
        """
        self.M = M
        self.T = T
        self.vector_len = vector_len
        self.n_symbols = n_symbols
        self._ptr = 0   # current insertion index (0..T-1)

        # ---- Observations ----
        # Stored as float32 on DEVICE to avoid repeated CPU->device copies during update
        self.spatial   = torch.zeros(M, T, 24, 15, 15, dtype=torch.float32, device=DEVICE)
        self.vector    = torch.zeros(M, T, vector_len, dtype=torch.float32, device=DEVICE)

        # ---- Actions (int64 for Categorical.log_prob) ----
        self.primary   = torch.zeros(M, T, dtype=torch.int64, device=DEVICE)
        self.param     = torch.zeros(M, T, dtype=torch.int64, device=DEVICE)
        self.emit      = torch.zeros(M, T, dtype=torch.int64, device=DEVICE)

        # ---- Primary-action mask at each timestep ----
        # Used during evaluate() to re-apply the same mask used at collection time
        self.prim_mask = torch.zeros(M, T, N_PRIMARY, dtype=torch.bool, device=DEVICE)

        # ---- Policy outputs ----
        self.logp      = torch.zeros(M, T, dtype=torch.float32, device=DEVICE)
        self.value     = torch.zeros(M, T, dtype=torch.float32, device=DEVICE)

        # ---- Env outputs ----
        self.reward    = torch.zeros(M, T, dtype=torch.float32, device=DEVICE)
        self.done      = torch.zeros(M, T, dtype=torch.bool, device=DEVICE)

        # ---- Validity mask: True only where agent was alive at that timestep ----
        self.valid_mask = torch.zeros(M, T, dtype=torch.bool, device=DEVICE)

        # ---- GAE outputs (computed after full horizon) ----
        self.returns    = torch.zeros(M, T, dtype=torch.float32, device=DEVICE)
        self.advantages = torch.zeros(M, T, dtype=torch.float32, device=DEVICE)

    # -----------------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all buffers and reset pointer for the next rollout."""
        self._ptr = 0
        self.spatial.zero_()
        self.vector.zero_()
        self.primary.zero_()
        self.param.zero_()
        self.emit.zero_()
        self.prim_mask.zero_()
        self.logp.zero_()
        self.value.zero_()
        self.reward.zero_()
        self.done.zero_()
        self.valid_mask.zero_()
        self.returns.zero_()
        self.advantages.zero_()

    def store(
        self,
        spatial: np.ndarray,    # (M, 24, 15, 15)
        vector: np.ndarray,     # (M, vector_len)
        primary_mask: np.ndarray,  # (M, N_PRIMARY) bool
        primary: Tensor,        # (M,)
        param: Tensor,          # (M,)
        emit: Tensor,           # (M,)
        logp: Tensor,           # (M,)
        value: Tensor,          # (M,)
        reward: np.ndarray,     # (M,)
        done: np.ndarray,       # (M,) bool
        alive_mask: np.ndarray, # (M,) bool — from obs.alive_mask
    ) -> None:
        """Store one timestep at the current pointer position."""
        t = self._ptr
        assert t < self.T, f"Buffer overflow: ptr={t} >= T={self.T}"

        self.spatial[:, t]    = torch.as_tensor(spatial, dtype=torch.float32, device=DEVICE)
        self.vector[:, t]     = torch.as_tensor(vector, dtype=torch.float32, device=DEVICE)
        self.prim_mask[:, t]  = torch.as_tensor(primary_mask, dtype=torch.bool, device=DEVICE)
        self.primary[:, t]    = primary.to(DEVICE)
        self.param[:, t]      = param.to(DEVICE)
        self.emit[:, t]       = emit.to(DEVICE)
        self.logp[:, t]       = logp.to(DEVICE)
        self.value[:, t]      = value.to(DEVICE)
        self.reward[:, t]     = torch.as_tensor(reward, dtype=torch.float32, device=DEVICE)
        self.done[:, t]       = torch.as_tensor(done, dtype=torch.bool, device=DEVICE)
        self.valid_mask[:, t] = torch.as_tensor(alive_mask, dtype=torch.bool, device=DEVICE)

        self._ptr += 1

    # -----------------------------------------------------------------------
    # GAE computation (§3.2)
    # -----------------------------------------------------------------------

    def compute_gae(
        self,
        last_value: Tensor,  # (M,) bootstrap value at end of horizon
        gamma: float = 0.995,
        lam: float = 0.95,
        normalize_adv: bool = True,
    ) -> None:
        """
        Compute per-slot GAE advantages and returns IN-PLACE (§3.2).

        Advantage accumulation resets at done=True (death boundary).
        Dead / never-alive timesteps get returns=0 and advantages=0.

        last_value: V(s_{T}) for non-terminal bootstrap of truncated episodes.
        """
        M, T = self.M, self.T

        # Move last_value to device once
        last_val = last_value.to(DEVICE)  # (M,)

        # gae accumulator per slot
        gae = torch.zeros(M, dtype=torch.float32, device=DEVICE)

        for t in reversed(range(T)):
            # Bootstrap: use last_value for t==T-1, else stored value at t+1
            if t == T - 1:
                next_value = last_val
            else:
                next_value = self.value[:, t + 1]

            # (1 - done_t): zero the bootstrap at death (§3.2 GAE reset)
            not_done = (~self.done[:, t]).float()  # (M,)

            # TD error: delta = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
            delta = (
                self.reward[:, t]
                + gamma * next_value * not_done
                - self.value[:, t]
            )

            # GAE: A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
            # done_t also resets the running gae accumulator (no leakage)
            gae = delta + gamma * lam * not_done * gae  # (M,)

            self.advantages[:, t] = gae
            self.returns[:, t]    = gae + self.value[:, t]

        # Mask out dead/never-alive timesteps
        self.advantages = self.advantages * self.valid_mask.float()
        self.returns    = self.returns    * self.valid_mask.float()

        # Normalise advantages over valid entries only (zero-mean, unit-var)
        # Can be disabled (e.g., for unit tests checking raw GAE values)
        if normalize_adv:
            valid_adv = self.advantages[self.valid_mask]
            if valid_adv.numel() > 1:
                mean = valid_adv.mean()
                std  = valid_adv.std().clamp(min=1e-8)
                # Apply normalisation everywhere; dead-slot values stay 0
                self.advantages = torch.where(
                    self.valid_mask,
                    (self.advantages - mean) / std,
                    torch.zeros_like(self.advantages),
                )

    # -----------------------------------------------------------------------
    # Minibatch iteration (§3.2 — valid entries only)
    # -----------------------------------------------------------------------

    def minibatches(
        self, batch_size: int
    ) -> Iterator[Tuple[Tensor, ...]]:
        """
        Yield flat minibatches over VALID (alive at timestep) (slot, time) pairs.

        Each batch is a tuple:
          (spatial, vector, prim_mask, primary, param, emit,
           old_logp, old_value, returns, advantages)

        Shapes: all (batch_size, ...) except the last possibly-smaller batch.
        """
        # Flat indices of valid (slot, time) entries
        valid_idx = self.valid_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
        N = valid_idx.numel()
        if N == 0:
            return

        # Shuffle for SGD decorrelation
        perm = torch.randperm(N, device=DEVICE)
        valid_idx = valid_idx[perm]

        # Flatten storage for gather
        M, T = self.M, self.T

        spatial_flat   = self.spatial.view(M * T, 24, 15, 15)
        vector_flat    = self.vector.view(M * T, self.vector_len)
        prim_mask_flat = self.prim_mask.view(M * T, N_PRIMARY)
        primary_flat   = self.primary.view(M * T)
        param_flat     = self.param.view(M * T)
        emit_flat      = self.emit.view(M * T)
        logp_flat      = self.logp.view(M * T)
        value_flat     = self.value.view(M * T)
        returns_flat   = self.returns.view(M * T)
        adv_flat       = self.advantages.view(M * T)

        for start in range(0, N, batch_size):
            idx = valid_idx[start : start + batch_size]
            yield (
                spatial_flat[idx],
                vector_flat[idx],
                prim_mask_flat[idx],
                primary_flat[idx],
                param_flat[idx],
                emit_flat[idx],
                logp_flat[idx],
                value_flat[idx],
                returns_flat[idx],
                adv_flat[idx],
            )
