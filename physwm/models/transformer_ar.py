"""Autoregressive next-state Transformer (the paper's baseline architecture).

Takes a context window of ``H`` past states and predicts the next state from the
last position. The **temporal-locality** inductive bias is a sliding attention
window (``attn_window``): with window ``w`` a token attends only to the previous
``w`` tokens, forcing the model toward a low-order (Newtonian) representation.

``hidden_states`` exposes the residual-stream activations (per layer, last token)
for the force-vs-invariant linear probe.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .heads import RegressionHead, ClassificationHead


def _causal_window_mask(H: int, window: Optional[int], device) -> torch.Tensor:
    """Additive attention mask (H,H): 0 where allowed, -inf where masked."""
    i = torch.arange(H, device=device)[:, None]
    j = torch.arange(H, device=device)[None, :]
    allowed = (j <= i)
    if window is not None:
        allowed = allowed & (j > i - window)
    mask = torch.zeros(H, H, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ARTransformer(nn.Module):
    def __init__(self, cfg, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, 4096, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.attn_window = cfg.attn_window
        if cfg.head == "classification":
            self.head = ClassificationHead(d, state_dim, cfg.n_bins, cfg.bin_range)
        else:
            self.head = RegressionHead(d, state_dim)

    def _trunk(self, context: torch.Tensor):
        """context (B,H,F) -> (final_hidden (B,H,d), per-layer hidden list)."""
        B, H, _ = context.shape
        x = self.in_proj(context) + self.pos[:, :H]
        mask = _causal_window_mask(H, self.attn_window, context.device)
        hiddens = [x]
        for blk in self.blocks:
            x = blk(x, mask)
            hiddens.append(x)
        x = self.ln_f(x)
        return x, hiddens

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        h, _ = self._trunk(context)
        return h[:, -1]                       # last-token hidden (B,d)

    def loss(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head.loss(self.forward(context), target)

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        return self.head.predict(self.forward(context))

    @torch.no_grad()
    def hidden_states(self, context: torch.Tensor) -> torch.Tensor:
        """Stacked residual-stream activations at the last token: (B, n_layers+1, d)."""
        _, hiddens = self._trunk(context)
        return torch.stack([h[:, -1] for h in hiddens], dim=1)
