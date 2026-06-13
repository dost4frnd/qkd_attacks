#!/usr/bin/env python3
"""
qkd_train_pipeline.py

Unified training + evaluation pipeline for:
1) HybridQLSTMClassifier   (quantum-inspired recurrent baseline; optional PennyLane hook can be added later)
2) LSTMClassifier
3) TinyTransformerClassifier
4) SequenceAutoencoder      (one-class anomaly detector; trained on normal class only)

Expected dataset format:
- One row per window/sequence sample
- A label column: label
- Optional split column: split with values train/val/test
- Feature columns flattened as time-suffixed names, e.g.:
    key_length_t00, qber_t00, ..., arrival_dev_t15
  where each timestep shares the same base feature set.

If your dataset is in "long" format, convert it to the flattened window format first.
The earlier synthetic generator in this conversation already outputs a compatible flat format.

Outputs:
- metrics.csv
- per-model confusion matrices
- best model checkpoints
- scaler.pkl, label_encoder.pkl, threshold.pkl (for the autoencoder)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Data utilities
# ----------------------------
TIMESTEP_RE = re.compile(r"^(?P<base>.+)_t(?P<t>\d+)$")


def infer_feature_layout(columns: Sequence[str]) -> Tuple[List[str], int]:
    """
    Return (base_feature_names, sequence_length) from flattened window columns.
    """
    groups: Dict[str, List[int]] = {}
    for col in columns:
        m = TIMESTEP_RE.match(col)
        if not m:
            continue
        base = m.group("base")
        t = int(m.group("t"))
        groups.setdefault(base, []).append(t)

    if not groups:
        raise ValueError(
            "Could not infer flattened sequence layout. Expected columns like feature_t00, feature_t01, ..."
        )

    base_features = sorted(groups.keys())
    seq_lens = {len(sorted(set(ts))) for ts in groups.values()}
    if len(seq_lens) != 1:
        raise ValueError(f"Inconsistent sequence lengths across features: {seq_lens}")
    seq_len = seq_lens.pop()
    return base_features, seq_len


def flatten_to_3d(df: pd.DataFrame, base_features: List[str], seq_len: int) -> np.ndarray:
    """
    Convert flattened rows to shape [N, T, F].
    """
    X = np.zeros((len(df), seq_len, len(base_features)), dtype=np.float32)
    for fi, feat in enumerate(base_features):
        for t in range(seq_len):
            col = f"{feat}_t{t:02d}"
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
            X[:, t, fi] = df[col].to_numpy(dtype=np.float32)
    return X


def stratified_split_indices(y: np.ndarray, train_frac: float, val_frac: float, seed: int):
    """
    Split indices into train/val/test in stratified fashion.
    """
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=seed
    )
    val_size = val_frac / (1.0 - train_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, stratify=y[temp_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDatasetUnlabeled(Dataset):
    def __init__(self, X: np.ndarray, indices: np.ndarray):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.X[idx]


def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler over all time steps concatenated together.
    """
    n, t, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n * t, f))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    X2 = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f).astype(np.float32)
    return X2


# ----------------------------
# Positional encoding
# ----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


# ----------------------------
# Models
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)


class TinyTransformerClassifier(nn.Module):
    """
    Tiny Transformer classifier using PyTorch's TransformerEncoderLayer.
    PyTorch's encoder layer is the standard self-attention + feedforward block.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class QuantumInspiredCell(nn.Module):
    """
    A lightweight "quantum-inspired" recurrent cell for reproducible experiments
    without external quantum libraries.

    It is NOT a true quantum circuit. It uses a richer feature map
    (x, sin(x), cos(x), x^2) to mimic the compact nonlinearity often
    sought in hybrid quantum-classical models.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        feat_dim = input_dim * 4
        self.x_map = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.Tanh(),
        )
        self.gate = nn.Linear(feat_dim + hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        return torch.cat([x, torch.sin(x), torch.cos(x), x * x], dim=-1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xm = self.feature_map(x)
        xm = self.dropout(xm)
        combined = torch.cat([xm, h], dim=-1)
        gates = self.gate(combined)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class HybridQLSTMClassifier(nn.Module):
    """
    Hybrid QLSTM-like sequence classifier.
    For full quantum-circuit replacement, this cell can later be swapped
    with a PennyLane-backed variational circuit if you install that dependency.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.cell = QuantumInspiredCell(input_dim, hidden_dim, dropout=dropout)
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        b, t, f = x.shape
        h = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for i in range(t):
            h, c = self.cell(x[:, i, :], h, c)
        z = self.post(h)
        return self.head(z)


class SequenceAutoencoder(nn.Module):
    """
    One-class anomaly detector:
    train only on normal samples, reconstruct sequences, use reconstruction error.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.recon_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        enc_out, (h_n, c_n) = self.encoder(x)
        latent = self.to_latent(h_n[-1])  # [B, latent_dim]
        dec_init = self.latent_to_hidden(latent)  # [B, hidden_dim]
        # Repeat a learned context vector across the sequence.
        dec_in = torch.zeros_like(x)
        dec_out, _ = self.decoder(dec_in, (dec_init.unsqueeze(0), torch.zeros_like(dec_init).unsqueeze(0)))
        recon = self.recon_head(dec_out)
        return recon


# ----------------------------
# Training and evaluation
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 7
    anomaly_threshold_quantile: float = 0.95


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    ckpt_path: Path,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs))
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(epoch - 1 + 1e-6)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        print(f"[classifier] epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if no_improve >= patience:
            print("[classifier] early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    ckpt_path: Path,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs))
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb in train_loader:
            Xb = Xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(Xb)
            loss = criterion(recon, Xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb in val_loader:
                Xb = Xb.to(device)
                recon = model(Xb)
                loss = criterion(recon, Xb)
                val_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(epoch - 1 + 1e-6)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        print(f"[autoencoder] epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if no_improve >= patience:
            print("[autoencoder] early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    all_probs = []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=-1)
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_y.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_probs), np.concatenate(all_y)


@torch.no_grad()
def reconstruction_errors(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    errs = []
    for Xb in loader:
        Xb = Xb.to(device)
        recon = model(Xb)
        per_sample = torch.mean((recon - Xb) ** 2, dim=(1, 2))
        errs.append(per_sample.cpu().numpy())
    return np.concatenate(errs)


def evaluate_multiclass(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    logits, probs, y_true = predict_classifier(model, loader, device)
    y_pred = np.argmax(logits, axis=1)

    metrics = {
        "model": name,
        "task": "multiclass",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_ovr_macro"] = np.nan

    print(f"\n=== {name} (multiclass) ===")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def evaluate_autoencoder(
    name: str,
    model: nn.Module,
    train_normal_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_test_binary: np.ndarray,
    threshold_quantile: float,
    device: torch.device,
) -> Dict[str, float]:
    # Threshold from normal validation reconstruction errors
    val_errs = reconstruction_errors(model, train_normal_loader if len(train_normal_loader.dataset) > 0 else val_loader, device)
    threshold = float(np.quantile(val_errs, threshold_quantile))

    test_errs = reconstruction_errors(model, test_loader, device)
    y_pred_binary = (test_errs > threshold).astype(int)

    metrics = {
        "model": name,
        "task": "one_class",
        "threshold": threshold,
        "accuracy": accuracy_score(y_test_binary, y_pred_binary),
        "precision_macro": precision_score(y_test_binary, y_pred_binary, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test_binary, y_pred_binary, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test_binary, y_pred_binary, average="macro", zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_test_binary, test_errs)
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["average_precision"] = average_precision_score(y_test_binary, test_errs)
    except Exception:
        metrics["average_precision"] = np.nan

    print(f"\n=== {name} (one-class) ===")
    print("Threshold:", threshold)
    print("Confusion matrix:")
    print(confusion_matrix(y_test_binary, y_pred_binary))
    print(classification_report(y_test_binary, y_pred_binary, target_names=["normal", "attack"], zero_division=0))

    return metrics


# ----------------------------
# Main
# ----------------------------
def build_model(model_name: str, input_dim: int, num_classes: int, seq_len: int, args) -> nn.Module:
    if model_name == "qlstm":
        return HybridQLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    if model_name == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if model_name == "transformer":
        d_model = max(args.hidden_dim, 32)
        # keep d_model divisible by num_heads
        if d_model % args.num_heads != 0:
            d_model = args.num_heads * ((d_model // args.num_heads) + 1)
        return TinyTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=args.num_heads,
            num_layers=args.transformer_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            max_len=max(seq_len, 512),
        )
    if model_name == "autoencoder":
        return SequenceAutoencoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to flattened QKD CSV/Parquet")
    parser.add_argument("--outdir", type=str, default="runs_qkd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    if "label" not in df.columns:
        raise KeyError("Dataset must contain a 'label' column.")

    base_features, seq_len = infer_feature_layout([c for c in df.columns if "_t" in c])
    X = flatten_to_3d(df, base_features, seq_len)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"].astype(str).values)

    # Splitting
    if "split" in df.columns and set(df["split"].dropna().unique()).issuperset({"train", "val", "test"}):
        train_idx = np.where(df["split"].astype(str).values == "train")[0]
        val_idx = np.where(df["split"].astype(str).values == "val")[0]
        test_idx = np.where(df["split"].astype(str).values == "test")[0]
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(
            y, train_frac=0.70, val_frac=0.15, seed=args.seed
        )

    # Scale using training data only
    scaler = fit_scaler_on_train(X[train_idx])
    X = apply_scaler(X, scaler)

    with open(outdir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(outdir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(outdir / "feature_layout.json", "w", encoding="utf-8") as f:
        json.dump({"base_features": base_features, "seq_len": seq_len}, f, indent=2)

    num_classes = len(label_encoder.classes_)
    input_dim = X.shape[-1]

    # class datasets
    train_ds = SequenceDataset(X, y, train_idx)
    val_ds = SequenceDataset(X, y, val_idx)
    test_ds = SequenceDataset(X, y, test_idx)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    # normal-only datasets for the autoencoder
    normal_label_name = None
    for c in label_encoder.classes_:
        if c.lower() == "normal":
            normal_label_name = c
            break
    if normal_label_name is None:
        raise ValueError("Could not find a class named 'normal' for one-class training.")
    normal_class_id = int(label_encoder.transform([normal_label_name])[0])

    train_normal_idx = train_idx[y[train_idx] == normal_class_id]
    val_normal_idx = val_idx[y[val_idx] == normal_class_id]
    test_binary = (y[test_idx] != normal_class_id).astype(int)

    train_normal_ds = SequenceDatasetUnlabeled(X, train_normal_idx)
    val_normal_ds = SequenceDatasetUnlabeled(X, val_normal_idx)
    test_unlabeled_ds = SequenceDatasetUnlabeled(X, test_idx)

    train_normal_loader = make_loader(train_normal_ds, args.batch_size, shuffle=True)
    val_normal_loader = make_loader(val_normal_ds, args.batch_size, shuffle=False)
    test_unlabeled_loader = make_loader(test_unlabeled_ds, args.batch_size, shuffle=False)

    model_specs = [
        ("qlstm", "qlstm"),
        ("lstm", "lstm"),
        ("transformer", "transformer"),
        ("autoencoder", "autoencoder"),
    ]

    metrics_rows = []

    for model_key, model_name in model_specs:
        ckpt_path = outdir / f"{model_key}.pt"
        print(f"\n==============================")
        print(f"Training {model_key}")
        print(f"==============================")

        model = build_model(model_key, input_dim, num_classes, seq_len, args).to(device)

        if model_key == "autoencoder":
            model = train_autoencoder(
                model,
                train_normal_loader,
                val_normal_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                ckpt_path=ckpt_path,
            )
            row = evaluate_autoencoder(
                name=model_key,
                model=model,
                train_normal_loader=train_normal_loader,
                val_loader=val_normal_loader,
                test_loader=test_unlabeled_loader,
                y_test_binary=test_binary,
                threshold_quantile=0.95,
                device=device,
            )
        else:
            model = train_classifier(
                model,
                train_loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                ckpt_path=ckpt_path,
            )
            row = evaluate_multiclass(
                name=model_key,
                model=model,
                loader=test_loader,
                device=device,
                label_encoder=label_encoder,
            )

        metrics_rows.append(row)

        # optional predictions
        if args.save_predictions and model_key != "autoencoder":
            logits, probs, y_true = predict_classifier(model, test_loader, device)
            pred_df = pd.DataFrame({
                "y_true": y_true,
                "y_pred": np.argmax(logits, axis=1),
            })
            for i, cls in enumerate(label_encoder.classes_):
                pred_df[f"p_{cls}"] = probs[:, i]
            pred_df.to_csv(outdir / f"{model_key}_predictions.csv", index=False)

        if args.save_predictions and model_key == "autoencoder":
            errs = reconstruction_errors(model, test_unlabeled_loader, device)
            pred_df = pd.DataFrame({
                "y_true_binary": test_binary,
                "recon_error": errs,
            })
            pred_df.to_csv(outdir / f"{model_key}_scores.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)
    print("\nSaved metrics to:", outdir / "metrics.csv")
    print(metrics_df.to_string(index=False))

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
