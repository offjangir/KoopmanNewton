"""DiT backbone shared by the generative methods (diffusion, flow, drifting).

Models a conditional map over the *next state*: a single "target token" (the noisy
state / flow interpolant / pure noise) is appended to ``H`` context tokens (the
history), and a stack of adaLN-Zero DiT blocks predicts the F-dim output for the
target token (noise eps / velocity / sample, depending on the method).

Conditioning (adaLN-Zero) is driven by ``cond = time_emb + context_summary``:
the generative/flow time (absent for drifting -> zeros) plus a global pooled
context embedding. This lets the **temporal-locality** knob (context length) and
history condition every block, exactly as in the AR baseline.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from .embeddings import TimeEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.ln1(x), shift1, scale1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * a
        h = modulate(self.ln2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class DiTBackbone(nn.Module):
    def __init__(self, cfg, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        d = cfg.d_model
        self.ctx_proj = nn.Linear(state_dim, d)
        self.tgt_proj = nn.Linear(state_dim, d)
        self.ctx_pos = nn.Parameter(torch.zeros(1, 4096, d))
        self.tgt_type = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.tgt_type, std=0.02)
        self.time_emb = TimeEmbedding(d)
        self.ctx_summary = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [DiTBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_f[-1].weight)
        nn.init.zeros_(self.ada_f[-1].bias)
        self.out = nn.Linear(d, state_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _embed_context(self, context):
        B, H, _ = context.shape
        c = self.ctx_proj(context) + self.ctx_pos[:, :H]
        summary = self.ctx_summary(c.mean(dim=1))          # (B,d)
        return c, summary

    def _run(self, x_t, context, t):
        """x_t (B,F), context (B,H,F), t (B,) or None -> (tokens (B,H+1,d), cond)."""
        B = x_t.shape[0]
        c, summary = self._embed_context(context)
        cond = self.time_emb(t, B, x_t.device) + summary
        tgt = self.tgt_proj(x_t).unsqueeze(1) + self.tgt_type   # (B,1,d)
        tokens = torch.cat([c, tgt], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        return tokens, cond

    def forward(self, x_t, context, t=None):
        tokens, cond = self._run(x_t, context, t)
        tgt = tokens[:, -1]
        shift, scale = self.ada_f(cond).chunk(2, dim=-1)
        tgt = modulate(self.ln_f(tgt).unsqueeze(1), shift, scale).squeeze(1)
        return self.out(tgt)

    @torch.no_grad()
    def hidden_states(self, context: torch.Tensor) -> torch.Tensor:
        """Residual-stream activations at the last *context* token, per layer.

        Uses a zeroed target token and t=0 so the readout reflects the dynamics
        encoded from the history -- directly comparable to the AR probe.
        """
        B, H, F = context.shape
        x0 = torch.zeros(B, F, device=context.device)
        c, summary = self._embed_context(context)
        cond = self.time_emb(None, B, context.device) + summary
        tgt = self.tgt_proj(x0).unsqueeze(1) + self.tgt_type
        tokens = torch.cat([c, tgt], dim=1)
        hiddens = [tokens[:, H - 1]]
        for blk in self.blocks:
            tokens = blk(tokens, cond)
            hiddens.append(tokens[:, H - 1])
        return torch.stack(hiddens, dim=1)     # (B, n_layers+1, d)
