"""
Trainer
=======
Main training loop for all DTI model variants.

Features:
- AdamW optimizer with cosine+warmup LR schedule (§8.2–8.3)
- Gradient clipping (§8.4)
- Best checkpoint saving (lowest val MSE)
- Per-epoch metric logging to console and CSV
"""
import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.training.metrics import compute_all_metrics


def _cosine_warmup_schedule(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup → cosine decay (§8.3)."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Generic trainer compatible with all four DTI variants.

    Parameters
    ----------
    model       : nn.Module — one of the four DTI variants
    variant_name: str       — name used for checkpoint/log filenames
    output_dir  : str       — root output directory
    lr          : float     — peak learning rate
    weight_decay: float
    max_epochs  : int
    patience    : int       — early stopping patience (epochs without val MSE improvement)
    grad_clip   : float     — gradient clipping norm
    device      : str       — 'cuda' | 'mps' | 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        variant_name: str,
        output_dir: str = "outputs",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 30,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        self.variant_name = variant_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = model.to(device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Output paths
        self.ckpt_dir = os.path.join(output_dir, "checkpoints", variant_name)
        self.log_dir = os.path.join(output_dir, "logs")
        self.results_dir = os.path.join(output_dir, "results")
        for d in [self.ckpt_dir, self.log_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

        self.results_path = os.path.join(self.results_dir, "results.csv")
        self._best_val_mse = float("inf")
        self._patience_counter = 0
        self._scheduler: Optional[LambdaLR] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        split_name: str = "random",
        seed: int = 42,
    ) -> Dict:
        """Run full training loop. Returns dict of final metrics."""
        n_train_steps = len(train_loader) * self.max_epochs
        n_warmup = max(1, int(0.05 * n_train_steps))
        self._scheduler = _cosine_warmup_schedule(self.optimizer, n_warmup, n_train_steps)

        history: List[Dict] = []
        self._best_val_mse = float("inf")
        self._patience_counter = 0

        print(f"\n{'─'*60}")
        print(f"  Variant : {self.variant_name}")
        print(f"  Device  : {self.device}")
        print(f"  Split   : {split_name}  |  Seed: {seed}")
        print(f"  Params  : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'─'*60}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._eval_epoch(val_loader)
            elapsed = time.time() - t0

            lr_now = self._scheduler.get_last_lr()[0]
            row = {
                "variant": self.variant_name,
                "split": split_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": lr_now,
                "time_s": round(elapsed, 2),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)

            # Console log
            print(
                f"Ep {epoch:3d}/{self.max_epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_mse={val_metrics['mse']:.4f} | "
                f"ci={val_metrics['ci']:.4f} | "
                f"r={val_metrics['pearson']:.4f} | "
                f"lr={lr_now:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_metrics["mse"] < self._best_val_mse:
                self._best_val_mse = val_metrics["mse"]
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    print(f"  [EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
                    break

        # Write final summary row to results CSV
        best_row = {
            "variant": self.variant_name,
            "split": split_name,
            "seed": seed,
            "best_val_mse": self._best_val_mse,
        }
        self._append_results_csv(best_row)
        self._save_epoch_log(history)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for drug_tokens, drug_mask, prot_tokens, prot_mask, affinities in loader:
            drug_tokens = drug_tokens.to(self.device)
            drug_mask = drug_mask.to(self.device)
            prot_tokens = prot_tokens.to(self.device)
            prot_mask = prot_mask.to(self.device)
            affinities = affinities.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(drug_tokens, drug_mask, prot_tokens, prot_mask)
            loss = nn.functional.mse_loss(preds, affinities)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds, all_true = [], []

        for drug_tokens, drug_mask, prot_tokens, prot_mask, affinities in loader:
            drug_tokens = drug_tokens.to(self.device)
            drug_mask = drug_mask.to(self.device)
            prot_tokens = prot_tokens.to(self.device)
            prot_mask = prot_mask.to(self.device)

            preds = self.model(drug_tokens, drug_mask, prot_tokens, prot_mask)
            all_preds.append(preds.cpu().numpy())
            all_true.append(affinities.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        return compute_all_metrics(y_pred, y_true)

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        path = os.path.join(self.ckpt_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def _append_results_csv(self, row: Dict) -> None:
        file_exists = os.path.isfile(self.results_path)
        with open(self.results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _save_epoch_log(self, history: List[Dict]) -> None:
        if not history:
            return
        log_path = os.path.join(self.log_dir, f"{self.variant_name}_history.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"  [Trainer] Epoch log saved → {log_path}")
