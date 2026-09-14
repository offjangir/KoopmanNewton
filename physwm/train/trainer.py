"""Generic training loop shared by every method.

Handles optimizer/schedule, AMP (CUDA only), grad clipping, checkpointing, and
JSONL logging. Heavy evaluation (rollout MDE, probing) lives in ``physwm.eval``
and is called from scripts; the trainer only tracks train/val loss.
"""

from __future__ import annotations

import json
import math
import os
from itertools import cycle

import torch


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


class Trainer:
    def __init__(self, cfg, method, train_loader, val_loader=None):
        self.cfg = cfg
        self.tc = cfg.train
        want_cuda = "cuda" in self.tc.device and torch.cuda.is_available()
        self.device = torch.device(self.tc.device if want_cuda else "cpu")
        self.method = method.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = torch.optim.AdamW(method.parameters(), lr=self.tc.lr,
                                     weight_decay=self.tc.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: _lr_lambda(s, self.tc.warmup, self.tc.steps))
        self.use_amp = self.tc.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        os.makedirs(self.tc.out_dir, exist_ok=True)
        self.history = []

    def _log(self, row):
        self.history.append(row)
        with open(os.path.join(self.tc.out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    @torch.no_grad()
    def validate(self, max_batches: int = 20):
        """Held-out metrics.

        Returns both:
        * ``val_loss``     -- the method's OWN training objective on val data.
          This is the only quantity comparable to ``train loss`` for an
          overfitting gap (for diffusion/flow the objective is an eps/velocity
          regression, which is a different quantity from prediction error).
        * ``val_pred_mse`` -- one-step prediction MSE via ``sample()``, which IS
          comparable across methods.
        """
        if self.val_loader is None:
            return None
        self.method.eval()
        tot_loss, tot_mse, nb = 0.0, 0.0, 0
        for batch in self.val_loader:
            batch = _to(batch, self.device)
            tot_loss += float(self.method.loss(batch).item())
            pred = self.method.sample(batch["context"])
            tot_mse += float(((pred - batch["target"]) ** 2).mean().item())
            nb += 1
            if nb >= max_batches:
                break
        self.method.train()
        nb = max(nb, 1)
        return {"val_loss": tot_loss / nb, "val_pred_mse": tot_mse / nb}

    @torch.no_grad()
    def train_loss_eval(self, max_batches: int = 20):
        """Training-objective value on TRAIN data in eval mode (for the gap)."""
        self.method.eval()
        tot, nb = 0.0, 0
        for batch in self.train_loader:
            tot += float(self.method.loss(_to(batch, self.device)).item())
            nb += 1
            if nb >= max_batches:
                break
        self.method.train()
        return tot / max(nb, 1)

    def fit(self):
        self.method.train()
        data = cycle(self.train_loader)
        for step in range(1, self.tc.steps + 1):
            batch = _to(next(data), self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.method.loss(batch)
            self.scaler.scale(loss).backward()
            if self.tc.grad_clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.method.parameters(), self.tc.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()

            if step % self.tc.log_every == 0 or step == 1:
                self._log(dict(step=step, loss=float(loss.item()),
                               lr=self.sched.get_last_lr()[0]))
            if self.tc.eval_every and step % self.tc.eval_every == 0:
                vl = self.validate()
                if vl is not None:
                    tr_loss = self.train_loss_eval()
                    # overfitting gap on the SAME objective, in eval mode
                    self._log(dict(step=step, train_loss_eval=tr_loss,
                                   generalization_gap=vl["val_loss"] - tr_loss, **vl))
            if self.tc.ckpt_every and step % self.tc.ckpt_every == 0:
                self.save_ckpt(step)
        self.save_ckpt(self.tc.steps, final=True)
        return self.history

    def save_ckpt(self, step, final=False):
        name = "final.pt" if final else f"step{step}.pt"
        torch.save({"model": self.method.state_dict(), "step": step,
                    "cfg": self.cfg.to_dict()}, os.path.join(self.tc.out_dir, name))
