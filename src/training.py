import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from sklearn.metrics import roc_auc_score, equal_error_rate  # type: ignore


def equal_error_rate(y_true, y_scores):
    """Compute EER (lower is better)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str = "auto",
        checkpoint_dir: str = "models",
        use_amp: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_eer": []}
        self.best_val_auc = 0.0
        self.best_ckpt_path = None

    # ------------------------------------------------------------------
    # Core train / eval loops
    # ------------------------------------------------------------------

    def _train_epoch(self, optimizer, criterion):
        self.model.train()
        total_loss, n = 0.0, 0

        for audio, labels in self.train_loader:
            audio  = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits = self.model(audio)
                loss = criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)
            n += len(labels)

        return total_loss / n

    @torch.no_grad()
    def _eval_epoch(self, criterion):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_probs, all_labels = [], []

        for audio, labels in self.val_loader:
            audio  = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(audio)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            total_loss += loss.item() * len(labels)
            n += len(labels)

        val_loss = total_loss / n
        val_auc  = roc_auc_score(all_labels, all_probs)
        val_eer  = equal_error_rate(all_labels, all_probs)
        return val_loss, val_auc, val_eer

    # ------------------------------------------------------------------
    # Two-phase training
    # ------------------------------------------------------------------

    def train(
        self,
        # Phase 1 — head only
        phase1_epochs: int = 10,
        phase1_lr: float = 1e-4,
        # Phase 2 — unfreeze top encoder layers
        phase2_epochs: int = 10,
        phase2_head_lr: float = 5e-5,
        phase2_encoder_lr: float = 1e-5,
        n_unfreeze_layers: int = 4,
        # Shared
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
    ):
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )

        # ---- Phase 1 ------------------------------------------------
        print("\n" + "="*50)
        print("Phase 1: training classification head only")
        print("="*50)

        optimizer = AdamW(self.model.classifier.parameters(), lr=phase1_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-6)

        for epoch in range(1, phase1_epochs + 1):
            self._run_epoch(epoch, phase1_epochs, optimizer, scheduler, criterion, phase=1)

        # ---- Phase 2 ------------------------------------------------
        print("\n" + "="*50)
        print(f"Phase 2: unfreezing top {n_unfreeze_layers} encoder layers")
        print("="*50)

        self.model.unfreeze_top_layers(n_unfreeze_layers)
        param_groups = self.model.get_param_groups(
            head_lr=phase2_head_lr, encoder_lr=phase2_encoder_lr
        )
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)

        for epoch in range(1, phase2_epochs + 1):
            self._run_epoch(epoch, phase2_epochs, optimizer, scheduler, criterion, phase=2)

        print("\nTraining complete.")
        print(f"Best val AUC: {self.best_val_auc:.4f}  |  checkpoint: {self.best_ckpt_path}")
        self._save_history()

    def _run_epoch(self, epoch, total_epochs, optimizer, scheduler, criterion, phase):
        t0 = time.time()
        train_loss = self._train_epoch(optimizer, criterion)
        val_loss, val_auc, val_eer = self._eval_epoch(criterion)
        scheduler.step()

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_auc"].append(val_auc)
        self.history["val_eer"].append(val_eer)

        elapsed = time.time() - t0
        print(
            f"[P{phase}] Epoch {epoch:3d}/{total_epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"AUC: {val_auc:.4f} | EER: {val_eer:.4f} | {elapsed:.1f}s"
        )

        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self._save_checkpoint("best_model.pt")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "encoder_name": self.model.encoder_name,
                "best_val_auc": self.best_val_auc,
            },
            path,
        )
        self.best_ckpt_path = str(path)
        print(f"  -> Saved checkpoint: {path}")

    def _save_history(self):
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")