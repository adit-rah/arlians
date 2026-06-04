"""
ArlianPolicy — shared-weights multi-agent actor-critic (build-spec §3.1).

Architecture:
  Spatial head : 3 conv layers (32, 64, 64; 3x3) over (N_SPATIAL, 15, 15) -> FC(256)
  Vector  head : FC(64) over (16,) interoceptive vector
  Trunk        : concat(spatial_feat, vector_feat) -> FC(256)
  Output heads : primary Linear(14), param Linear(8), emit Linear(n_symbols), value Linear(1)

The primary action mask (build_mask output) is applied as a -inf additive mask before
sampling so no illegal action is ever selected (§3.1 action masking).

Device: CUDA if available, else MPS, else CPU (§3.1, §5).
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---- constants from frozen sim contracts ----
N_SPATIAL = 24      # sim/observe.py SPATIAL_CHANNELS
WINDOW_SIZE = 15    # 2 * window_radius(7) + 1
N_PRIMARY = 14      # sim/actions.py N_PRIMARY
N_PARAM = 8         # sim/actions.py N_PARAM

_NEG_INF = float("-inf")


def _get_device() -> torch.device:
    """Return CUDA if available, else MPS, else CPU (§5)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


class ArlianPolicy(nn.Module):
    """
    Shared-weights actor-critic for all M agent slots (§3.1).

    No per-agent state (genome excluded from backprop by design, §3.3).
    GRU is optional and OFF by default in Phase 1.
    """

    def __init__(
        self,
        n_symbols: int = 8,
        use_gru: bool = False,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.n_symbols = n_symbols
        self.use_gru = use_gru
        self.hidden_size = hidden_size

        # ---- Spatial head: 3 conv layers -> flatten -> FC(256) ----
        # Input: (B, 24, 15, 15) float32
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(N_SPATIAL, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # After 3 same-padding conv layers the spatial dims are still (15, 15)
        spatial_flat = 64 * WINDOW_SIZE * WINDOW_SIZE  # 64 * 15 * 15 = 14_400
        self.spatial_fc = nn.Sequential(
            nn.Linear(spatial_flat, hidden_size),
            nn.ReLU(inplace=True),
        )

        # ---- Vector head: FC(64) over (vector_len,) ----
        # vector_len = 10 + genome_dim (default 16 = 10 + 6)
        # We accept any vector_len at runtime via lazy init, but spec says 16.
        self.vector_fc = nn.Sequential(
            nn.Linear(16, 64),  # §3.1: 10 interoceptive + 6 genome = 16
            nn.ReLU(inplace=True),
        )

        # ---- Trunk: concat(256 + 64) -> FC(256) ----
        trunk_in = hidden_size + 64  # 320
        if use_gru:
            self.gru = nn.GRUCell(trunk_in, hidden_size)
            trunk_out = hidden_size
        else:
            self.trunk = nn.Sequential(
                nn.Linear(trunk_in, hidden_size),
                nn.ReLU(inplace=True),
            )
            trunk_out = hidden_size

        # ---- Output heads ----
        self.head_primary = nn.Linear(trunk_out, N_PRIMARY)
        self.head_param   = nn.Linear(trunk_out, N_PARAM)
        self.head_emit    = nn.Linear(trunk_out, n_symbols)
        self.head_value   = nn.Linear(trunk_out, 1)

        # Weight initialisation: orthogonal for linear layers (common PPO practice)
        self._init_weights()
        self.to(DEVICE)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller gain for output heads (PPO convention)
        for head in (self.head_primary, self.head_param, self.head_emit):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.head_value.weight, gain=1.0)

    def _encode(
        self,
        spatial: torch.Tensor,   # (B, 24, 15, 15)
        vector: torch.Tensor,    # (B, 16)
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run both heads and trunk, return (features, new_hx)."""
        sp = self.spatial_conv(spatial)           # (B, 64, 15, 15)
        sp = sp.flatten(1)                        # (B, 14400)
        sp = self.spatial_fc(sp)                  # (B, 256)

        vec = self.vector_fc(vector)              # (B, 64)

        cat = torch.cat([sp, vec], dim=-1)        # (B, 320)

        if self.use_gru:
            new_hx = self.gru(cat, hx)            # (B, 256)
            feat = new_hx
        else:
            feat = self.trunk(cat)                # (B, 256)
            new_hx = None

        return feat, new_hx

    @staticmethod
    def _to_tensor(arr, dtype=torch.float32) -> torch.Tensor:
        """Move a numpy array to DEVICE as a tensor."""
        return torch.as_tensor(arr, dtype=dtype, device=DEVICE)

    def _apply_primary_mask(
        self,
        logits: torch.Tensor,        # (B, N_PRIMARY)
        primary_mask: torch.Tensor,  # (B, N_PRIMARY) bool
    ) -> torch.Tensor:
        """
        Add -inf to masked-out primary action logits (§3.1).

        Dead-slot rows may have all-False mask (no legal actions) — in that case
        we fall back to making action 0 (NOOP) legal so Categorical never receives
        an all-inf row (which would produce NaN). Dead-slot actions are never
        executed by the env so the specific value does not matter.
        """
        logits = logits.clone()
        logits[~primary_mask] = _NEG_INF

        # Guard: rows where ALL actions are -inf -> enable NOOP (index 0)
        all_inf = ~primary_mask.any(dim=-1)  # (B,) bool
        if all_inf.any():
            logits[all_inf, 0] = 0.0

        return logits

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        spatial: "np.ndarray | torch.Tensor",   # (M, 24, 15, 15)
        vector: "np.ndarray | torch.Tensor",    # (M, 16)
        primary_mask: "np.ndarray | torch.Tensor",  # (M, N_PRIMARY) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions for all live slots (§3.1 act interface).

        Returns dict with keys:
          primary : (M,) int64
          param   : (M,) int64
          emit    : (M,) int64
          logp    : (M,) float32  — joint log-prob = sum of three head logps
          value   : (M,) float32
        """
        spatial_t = self._to_tensor(spatial)
        vector_t  = self._to_tensor(vector)
        mask_t    = self._to_tensor(primary_mask, dtype=torch.bool)

        feat, _ = self._encode(spatial_t, vector_t)

        # Primary: mask then sample
        prim_logits = self._apply_primary_mask(self.head_primary(feat), mask_t)
        prim_dist   = Categorical(logits=prim_logits)
        primary     = prim_dist.sample()                # (M,)

        # Param: unconditional
        param_dist  = Categorical(logits=self.head_param(feat))
        param       = param_dist.sample()               # (M,)

        # Emit
        emit_dist   = Categorical(logits=self.head_emit(feat))
        emit        = emit_dist.sample()                # (M,)

        # Value
        value = self.head_value(feat).squeeze(-1)       # (M,)

        # Joint log-prob (sum of independent heads)
        logp = (
            prim_dist.log_prob(primary)
            + param_dist.log_prob(param)
            + emit_dist.log_prob(emit)
        )

        return {
            "primary": primary,
            "param":   param,
            "emit":    emit,
            "logp":    logp,
            "value":   value,
        }

    def evaluate(
        self,
        spatial: torch.Tensor,        # (B, 24, 15, 15)
        vector: torch.Tensor,         # (B, 16)
        primary_mask: torch.Tensor,   # (B, N_PRIMARY) bool
        primary: torch.Tensor,        # (B,) int64
        param: torch.Tensor,          # (B,) int64
        emit: torch.Tensor,           # (B,) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored (a,s) pairs for PPO surrogate loss (§3.2).

        Returns:
          logp     : (B,) — joint log-prob under current policy
          entropy  : (B,) — joint entropy (sum of head entropies)
          value    : (B,)
        """
        feat, _ = self._encode(spatial, vector)

        prim_logits = self._apply_primary_mask(self.head_primary(feat), primary_mask)
        prim_dist   = Categorical(logits=prim_logits)

        param_dist  = Categorical(logits=self.head_param(feat))
        emit_dist   = Categorical(logits=self.head_emit(feat))

        logp = (
            prim_dist.log_prob(primary)
            + param_dist.log_prob(param)
            + emit_dist.log_prob(emit)
        )
        entropy = (
            prim_dist.entropy()
            + param_dist.entropy()
            + emit_dist.entropy()
        )
        value = self.head_value(feat).squeeze(-1)

        return logp, entropy, value
