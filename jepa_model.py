"""
jepa_model.py

A Joint-Embedding Predictive Architecture (JEPA) for short-term
forecasting of atmospheric profiles.

Why JEPA here
-------------
Classic supervised forecasters minimize a loss directly in *pixel/value
space* (e.g. MSE on the raw humidity profile). JEPA instead:

  1. Encodes the *context* (past profiles) with a context encoder f_theta
     into an embedding z_x.
  2. Encodes the *target* (the actual future profile) with a separate
     target encoder f_bar (an EMA copy of f_theta, no gradients) into
     an embedding z_y.
  3. Trains a predictor g_phi to predict z_y from z_x conditioned on
     "which future" (here: the forecast horizon, 3h or 6h), i.e.
     z_y_hat = g_phi(z_x, horizon).
  4. The loss is ||z_y_hat - stopgrad(z_y)||^2 — prediction happens in
     representation space, not value space. This is what keeps the
     embedding from collapsing to a trivial/near-constant vector
     (avoided further by using an EMA target + stopgrad, matching
     I-JEPA/JEPA practice) and is what the assignment calls the
     "hidden context vector".

The assignment also requires an actual RMSE-evaluable forecast, so a
thin linear/MLP *forecast head* sits on top of the predicted embedding
z_y_hat and decodes it into the physical ABS_HUMIDITY profile. The
context/predicted vector is extracted immediately *before* this head,
which is the "final prognostic layer" referenced in task 4.

Everything below is horizon-agnostic: the same context/target/predictor
stack is reused for both the 3h and 6h heads, with the horizon fed in
as a small learned embedding so the model isn't just two independent
networks bolted together.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class SinusoidalLevelEmbedding(nn.Module):
    """Fixed sinusoidal embedding over the (integer) pressure-level index,
    so the encoder knows the vertical ordering of tokens."""

    def __init__(self, dim: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, n: int):
        return self.pe[:n]


class ProfileEncoder(nn.Module):
    """
    Encodes one or several stacked profiles (a context window) into a
    single pooled embedding.

    Input:  [B, T, L, C]  (T context timesteps, L pressure levels, C variables)
    Output: [B, D]        pooled embedding
    """

    def __init__(self, n_channels: int, n_levels: int, embed_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 3, max_ctx_len: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.level_proj = nn.Linear(n_channels, embed_dim)
        self.level_pos = SinusoidalLevelEmbedding(embed_dim, max_len=n_levels + 1)
        self.time_pos = nn.Embedding(max_ctx_len, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, L, C]  (a single future profile arrives as T=1)
        B, T, L, C = x.shape
        tok = self.level_proj(x)                                  # [B, T, L, D]
        tok = tok + self.level_pos(L).view(1, 1, L, -1)
        tok = tok + self.time_pos(torch.arange(T, device=x.device)).view(1, T, 1, -1)
        tok = tok.reshape(B, T * L, self.embed_dim)                # flatten to a token sequence

        cls = self.cls.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        enc = self.encoder(tok)
        pooled = self.norm(enc[:, 0])                              # CLS token -> pooled embedding
        return pooled


class HorizonPredictor(nn.Module):
    """g_phi: predicts the target embedding from the context embedding,
    conditioned on which horizon (3h vs 6h) is being predicted."""

    def __init__(self, embed_dim: int = 128, n_horizons: int = 2, hidden: int = 256):
        super().__init__()
        self.horizon_embed = nn.Embedding(n_horizons, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z_ctx: torch.Tensor, horizon_idx: torch.Tensor) -> torch.Tensor:
        h = self.horizon_embed(horizon_idx)
        return self.net(torch.cat([z_ctx, h], dim=-1))


class ForecastHead(nn.Module):
    """The 'final prognostic layer': decodes a predicted embedding into
    the physical ABS_HUMIDITY profile on the fixed pressure-level grid."""

    def __init__(self, embed_dim: int, n_levels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_levels),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)  # [B, L], standardized ABS_HUMIDITY


HORIZON_TO_IDX = {3: 0, 6: 1}


class WeatherJEPA(nn.Module):
    def __init__(self, n_channels: int, n_levels: int, embed_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 3, max_ctx_len: int = 16,
                 ema_momentum: float = 0.996):
        super().__init__()
        self.context_encoder = ProfileEncoder(
            n_channels, n_levels, embed_dim, n_heads, n_layers, max_ctx_len
        )
        # Target encoder starts as an exact copy, then only ever moves via EMA.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = HorizonPredictor(embed_dim, n_horizons=len(HORIZON_TO_IDX))
        self.forecast_head = ForecastHead(embed_dim, n_levels)
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1 - m)
        for b_t, b_c in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            b_t.data.copy_(b_c.data)

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, T, L, C] -> z_ctx: [B, D]. This is the extractable
        hidden/context vector referenced in task 4 of the assignment."""
        return self.context_encoder(context)

    def forward(self, context: torch.Tensor, future_full: dict[int, torch.Tensor]):
        """
        context: [B, T, L, C]
        future_full: {horizon_hours: [B, L, C]} standardized full future profiles

        Returns a dict with, per horizon:
          - z_ctx_pred: predicted embedding right before the forecast head
                        (this is the vector to extract for the leaderboard/defense)
          - forecast: decoded standardized ABS_HUMIDITY profile [B, L]
          - jepa_loss_terms: (z_hat, stopgrad(z_target)) for the representation loss
        """
        z_ctx = self.encode_context(context)  # [B, D]

        out = {"z_ctx": z_ctx, "per_horizon": {}}
        for hours, idx in HORIZON_TO_IDX.items():
            horizon_idx = torch.full((z_ctx.size(0),), idx, dtype=torch.long, device=z_ctx.device)
            z_hat = self.predictor(z_ctx, horizon_idx)              # [B, D]  <- "extract before final layer"

            with torch.no_grad():
                y = future_full[hours].unsqueeze(1)                 # [B, 1, L, C]
                z_target = self.target_encoder(y)                   # [B, D], stopgrad by construction

            forecast = self.forecast_head(z_hat)                    # [B, L]

            out["per_horizon"][hours] = {
                "z_hat": z_hat,
                "z_target": z_target,
                "forecast": forecast,
            }
        return out


def jepa_and_forecast_loss(model_out: dict, targets: dict[int, torch.Tensor],
                            jepa_weight: float = 1.0, forecast_weight: float = 1.0):
    """
    Combines:
      - representation-space JEPA loss: ||z_hat - stopgrad(z_target)||^2
      - value-space forecast loss: MSE(decoded profile, true standardized profile)

    Both losses share the same predictor output z_hat, so the embedding
    is pulled toward being both (a) predictive of the true future
    representation and (b) decodable into an accurate physical forecast.
    """
    total_jepa, total_fcst = 0.0, 0.0
    per_horizon_fcst = {}
    for hours, d in model_out["per_horizon"].items():
        jepa_term = nn.functional.mse_loss(d["z_hat"], d["z_target"])
        fcst_term = nn.functional.mse_loss(d["forecast"], targets[hours])
        total_jepa = total_jepa + jepa_term
        total_fcst = total_fcst + fcst_term
        per_horizon_fcst[hours] = fcst_term.detach()

    loss = jepa_weight * total_jepa + forecast_weight * total_fcst
    return loss, {"jepa": total_jepa.detach(), "forecast": total_fcst.detach(), "per_horizon": per_horizon_fcst}
