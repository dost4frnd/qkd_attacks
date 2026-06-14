
#!/usr/bin/env python3
"""
TF-QKD analysis pipeline (PDF-first, Transactions-safe).

Key changes versus the earlier compact script:
- saves figures as PDF for LaTeX embedding
- separates supervised classification metrics from anomaly-detection metrics
- handles unknown_attack rows safely
- produces per-dataset named outputs
- creates cross-domain comparison plots from analysis directories
- keeps the core training code untouched

Expected training artifacts in --run-dir:
- qlstm.pt
- lstm.pt
- transformer.pt
- autoencoder.pt
- scaler.pkl
- label_encoder.pkl
- feature_layout.json
- run_config.json (optional)

Expected input data:
- flattened TF-QKD CSV/Parquet with columns feature_t00..feature_tNN
- optional split column
- optional long file in the same folder for drift plots
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from qkd_train_pipeline_tfqkd import (
    HybridQLSTMClassifier,
    LSTMClassifier,
    SequenceAutoencoder,
    TinyTransformerClassifier,
)

TIMESTEP_RE = re.compile(r"^(?P<base>.+)_t(?P<t>\d+)$")
CORE_DRIFT_FEATURES = [
    "phase_lock_error_rad",
    "visibility",
    "ref_power_t_dbm",
    "ref_wavelength_t_nm",
    "coincidence_rate",
    "qber_phase",
    "qber_bit",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_name_from_path(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if stem.endswith("_flat"):
        stem = stem[:-5]
    return parent if parent else stem


def infer_feature_layout(columns: Sequence[str]) -> Tuple[List[str], int]:
    groups: Dict[str, List[int]] = {}
    for c in columns:
        m = TIMESTEP_RE.match(c)
        if not m:
            continue
        groups.setdefault(m.group("base"), []).append(int(m.group("t")))
    if not groups:
        raise ValueError("Could not infer flattened sequence layout.")
    bases = sorted(groups.keys())
    seq_lens = {len(set(ts)) for ts in groups.values()}
    if len(seq_lens) != 1:
        raise ValueError(f"Inconsistent sequence lengths across features: {seq_lens}")
    return bases, seq_lens.pop()


def flatten_to_3d(df: pd.DataFrame, bases: List[str], seq_len: int) -> np.ndarray:
    X = np.zeros((len(df), seq_len, len(bases)), dtype=np.float32)
    for fi, feat in enumerate(bases):
        for t in range(seq_len):
            col = f"{feat}_t{t:02d}"
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
            X[:, t, fi] = df[col].astype(np.float32).values
    return X


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    n, t, f = X_train.shape
    sc = StandardScaler()
    sc.fit(X_train.reshape(n * t, f))
    return sc


def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    return sc.transform(X.reshape(n * t, f)).reshape(n, t, f).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray):
        self.X = torch.tensor(X[idx], dtype=torch.float32)
        self.y = torch.tensor(y[idx], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class UnlabeledSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, idx: np.ndarray):
        self.X = torch.tensor(X[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def make_loader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def load_artifacts(run_dir: Path):
    with open(run_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(run_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(run_dir / "feature_layout.json", "r", encoding="utf-8") as f:
        fl = json.load(f)
    return scaler, label_encoder, fl["base_features"], int(fl["seq_len"])


def build_models(input_dim: int, num_classes: int, seq_len: int, device: torch.device, run_dir: Path):
    cfg = {
        "hidden_dim": 64,
        "latent_dim": 16,
        "dropout": 0.1,
        "num_layers": 1,
        "transformer_layers": 2,
        "num_heads": 4,
    }
    run_cfg = run_dir / "run_config.json"
    if run_cfg.exists():
        try:
            cfg.update(json.loads(run_cfg.read_text()))
        except Exception:
            pass

    d_model = max(cfg["hidden_dim"], 32)
    if d_model % cfg["num_heads"] != 0:
        d_model = cfg["num_heads"] * ((d_model // cfg["num_heads"]) + 1)

    qlstm = HybridQLSTMClassifier(input_dim, cfg["hidden_dim"], num_classes, dropout=cfg["dropout"]).to(device)
    lstm = LSTMClassifier(input_dim, cfg["hidden_dim"], num_classes, num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    transformer = TinyTransformerClassifier(
        input_dim,
        d_model,
        cfg["num_heads"],
        cfg["transformer_layers"],
        num_classes,
        dropout=cfg["dropout"],
        max_len=max(seq_len, 512),
    ).to(device)
    autoencoder = SequenceAutoencoder(input_dim, cfg["hidden_dim"], cfg["latent_dim"], dropout=cfg["dropout"]).to(device)
    return qlstm, lstm, transformer, autoencoder


def try_load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> bool:
    if not ckpt_path.exists():
        return False
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return True


def save_pdf(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def maybe_find_long_file(data_path: Path) -> Optional[Path]:
    candidates = [
        data_path.parent / "tfqkd_long.csv",
        data_path.parent / "tfqkd_long.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# -----------------------------------------------------------------------------
# Feature / attention / plotting helpers
# -----------------------------------------------------------------------------
def extract_transformer_attention(model: TinyTransformerClassifier, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        h = model.input_proj(x)
        h = model.posenc(h)
        attn_mats = []
        out = h
        for layer in model.encoder.layers:
            x1 = layer.norm1(out) if layer.norm_first else out
            attn_out, attn_w = layer.self_attn(
                x1, x1, x1,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_mats.append(attn_w.detach().cpu().numpy())  # [B, heads, T, T]
            out = out + layer.dropout1(attn_out)
            x2 = layer.norm2(out) if layer.norm_first else out
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            out = out + layer.dropout2(ff)
        # average over layers, batch, heads
        attn = np.mean(np.stack(attn_mats, axis=0), axis=(0, 1, 2))
        return attn


def plot_attention_heatmap(attn: np.ndarray, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key timestep")
    ax.set_ylabel("Query timestep")
    save_pdf(fig, outpath)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, class_names: Sequence[str], outpath: Path, title: str) -> None:
    n = len(embeddings)
    if n > 2500:
        idx = np.random.choice(n, 2500, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    pca = PCA(n_components=min(30, embeddings.shape[1]), random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    perp = max(5, min(40, len(embeddings) // 20))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=42)
    xy = tsne.fit_transform(emb_pca)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        m = labels == i
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.72, label=name)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    save_pdf(fig, outpath)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    save_pdf(fig, outpath)


def plot_metrics_supervised(metrics_df: pd.DataFrame, outpath: Path, title: str) -> None:
    df = metrics_df[metrics_df["task"] == "supervised"].copy()
    if df.empty:
        return
    models = ["qlstm", "lstm", "transformer"]
    df = df.set_index("model").reindex(models)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col in zip(axes, ["accuracy", "f1_macro", "roc_auc_ovr_macro"]):
        if col not in df.columns:
            ax.axis("off")
            continue
        ax.bar(models, df[col].fillna(0.0).values)
        ax.set_title(col)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    save_pdf(fig, outpath)


def plot_metrics_anomaly(metrics_df: pd.DataFrame, outpath: Path, title: str) -> None:
    df = metrics_df[metrics_df["task"] == "anomaly"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    cols = ["accuracy", "f1_macro", "roc_auc", "average_precision"]
    row = df.iloc[0]
    for ax, col in zip(axes, cols):
        val = float(row[col]) if col in row and pd.notna(row[col]) else 0.0
        ax.bar(["autoencoder"], [val])
        ax.set_title(col)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    save_pdf(fig, outpath)


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, outdir: Path, prefix: str) -> Tuple[float, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {prefix}")
    ax.legend()
    save_pdf(fig, outdir / f"{prefix}_roc.pdf")

    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {prefix}")
    ax.legend()
    save_pdf(fig, outdir / f"{prefix}_pr.pdf")
    return float(auc), float(ap)


def plot_anomaly_hist(errors: np.ndarray, threshold: float, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=40, alpha=0.9)
    ax.axvline(threshold, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("count")
    save_pdf(fig, outpath)


def plot_feature_drift(long_df: pd.DataFrame, outdir: Path, dataset_label: str) -> None:
    if not {"label", "t"}.issubset(long_df.columns):
        return
    features = [f for f in CORE_DRIFT_FEATURES if f in long_df.columns]
    if not features:
        return

    grouped = long_df.groupby(["label", "t"])[features].mean().reset_index()
    labels = [x for x in grouped["label"].unique().tolist()]
    for feat in features:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for label in labels:
            tmp = grouped[grouped["label"] == label]
            ax.plot(tmp["t"], tmp[feat], label=label, linewidth=1.2)
        ax.set_title(f"{dataset_label} — {feat}")
        ax.set_xlabel("timestep")
        ax.set_ylabel(feat)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.25)
        save_pdf(fig, outdir / f"{dataset_label}_drift_{feat}.pdf")


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def supervised_label_mask(y_labels: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    known = set(label_encoder.classes_.tolist())
    return np.array([label in known for label in y_labels], dtype=bool)


def evaluate_supervised(
    model: nn.Module,
    loader: DataLoader,
    y_true: np.ndarray,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    probs_all = []
    feats_all = []
    y_all = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=-1)
            if hasattr(model, "features"):
                feats = model.features(Xb)
            else:
                feats = probs
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            feats_all.append(feats.cpu().numpy())
            y_all.append(yb.numpy())

    logits = np.concatenate(logits_all)
    probs = np.concatenate(probs_all)
    feats = np.concatenate(feats_all)
    y_pred = np.argmax(logits, axis=1)
    y_true = np.concatenate(y_all)

    metrics = {
        "task": "supervised",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_ovr_macro"] = np.nan

    return metrics, y_true, y_pred, probs, feats


def evaluate_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_test_binary: np.ndarray,
    device: torch.device,
    threshold_quantile: float = 0.95,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    model.eval()

    def _errs(loader):
        errs = []
        with torch.no_grad():
            for Xb in loader:
                Xb = Xb.to(device)
                recon = model(Xb)
                errs.append(torch.mean((recon - Xb) ** 2, dim=(1, 2)).cpu().numpy())
        return np.concatenate(errs)

    tr_err = _errs(train_loader)
    val_err = _errs(val_loader)
    threshold = float(np.quantile(tr_err if len(tr_err) else val_err, threshold_quantile))
    test_err = _errs(test_loader)
    y_pred = (test_err > threshold).astype(int)

    metrics = {
        "task": "anomaly",
        "accuracy": accuracy_score(y_test_binary, y_pred),
        "precision_macro": precision_score(y_test_binary, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test_binary, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test_binary, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test_binary, test_err)
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["average_precision"] = average_precision_score(y_test_binary, test_err)
    except Exception:
        metrics["average_precision"] = np.nan

    return metrics, test_err, threshold


# -----------------------------------------------------------------------------
# Single-dataset analysis
# -----------------------------------------------------------------------------
def run_analysis(data_path: Path, run_dir: Path, outdir: Path, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_label = dataset_name_from_path(data_path)

    df = load_dataset(data_path)
    if "label" not in df.columns:
        raise KeyError("Dataset must contain a label column.")
    bases, seq_len = infer_feature_layout([c for c in df.columns if TIMESTEP_RE.match(c)])
    X = flatten_to_3d(df, bases, seq_len)

    scaler, label_encoder, saved_bases, saved_seq_len = load_artifacts(run_dir)
    X = apply_scaler(X, scaler)

    # label handling: known classes for supervised evaluation only
    labels = df["label"].astype(str).values
    #patched
    # labels_arr = label_encoder.transform(labels) if len(labels) else np.array([], dtype=int)
    labels_arr = (
        np.array(
            [
                label_encoder.transform([lab])[0]
                if lab in label_encoder.classes_
                else -1
                for lab in labels
            ],
            dtype=int,
        )
        if len(labels)
        else np.array([], dtype=int)
    )
    
    known_mask = np.array([lab in label_encoder.classes_ for lab in labels], dtype=bool)
    y_known = label_encoder.transform(labels[known_mask]) if known_mask.any() else np.array([], dtype=int)

    if "split" in df.columns and set(df["split"].astype(str).unique()).issuperset({"train", "val", "test"}):
        train_idx = np.where(df["split"].astype(str).values == "train")[0]
        val_idx = np.where(df["split"].astype(str).values == "val")[0]
        test_idx = np.where(df["split"].astype(str).values == "test")[0]
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        train_idx, temp_idx = train_test_split(idx, train_size=0.70, stratify=labels, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, stratify=labels[temp_idx], random_state=seed)

    # For supervised metrics, only known labels are valid.
    test_known_mask = known_mask[test_idx]
    test_known_idx = test_idx[test_known_mask]
    y_test_known = label_encoder.transform(df.iloc[test_known_idx]["label"].astype(str).values) if len(test_known_idx) else np.array([], dtype=int)

    test_ds_known = SequenceDataset(X, labels_arr, test_known_idx)
    test_loader_known = make_loader(test_ds_known, 64, shuffle=False)

    # anomaly labels: everything except normal is attack
    normal_label = "normal" if "normal" in label_encoder.classes_ else label_encoder.classes_[0]
    y_test_binary = (df.iloc[test_idx]["label"].astype(str).values != normal_label).astype(int)
    test_loader_all = make_loader(UnlabeledSequenceDataset(X, test_idx), 64, shuffle=False)
    train_normal_idx = train_idx[df.iloc[train_idx]["label"].astype(str).values == normal_label]
    val_normal_idx = val_idx[df.iloc[val_idx]["label"].astype(str).values == normal_label]
    train_normal_loader = make_loader(UnlabeledSequenceDataset(X, train_normal_idx), 64, shuffle=False)
    val_normal_loader = make_loader(UnlabeledSequenceDataset(X, val_normal_idx), 64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qlstm, lstm, transformer, autoencoder = build_models(X.shape[-1], len(label_encoder.classes_), seq_len, device, run_dir)

    models = [
        ("qlstm", qlstm),
        ("lstm", lstm),
        ("transformer", transformer),
    ]

    rows = []
    figure_index = []

    # ------------------
    # supervised models
    # ------------------
    for name, model in models:
        if not try_load_checkpoint(model, run_dir / f"{name}.pt", device):
            continue

        metrics, y_true, y_pred, probs, feats = evaluate_supervised(model, test_loader_known, y_test_known, label_encoder, device)
        metrics.update({
            "dataset": dataset_label,
            "model": name,
            "train_run_dir": run_dir.name,
            "test_known_rows": int(len(test_known_idx)),
            "unknown_rows_in_dataset": int((~known_mask).sum()),
        })
        rows.append(metrics)

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
        cm_path = outdir / f"{dataset_label}_{name}_confusion.pdf"
        plot_confusion_matrix(cm, label_encoder.classes_, cm_path, f"{dataset_label} — {name} confusion matrix")
        figure_index.append({"file": cm_path.name, "kind": "confusion_matrix", "model": name, "dataset": dataset_label})

        if name == "transformer":
            plot_tsne(
                feats,
                y_true,
                label_encoder.classes_,
                outdir / f"{dataset_label}_{name}_tsne.pdf",
                f"{dataset_label} — transformer embeddings",
            )
            figure_index.append({"file": f"{dataset_label}_{name}_tsne.pdf", "kind": "tsne", "model": name, "dataset": dataset_label})

            # attention heatmap
            sample_batch = next(iter(test_loader_known))[0].to(device)
            sample_batch = sample_batch[: min(32, sample_batch.size(0))]
            attn = extract_transformer_attention(model, sample_batch)
            attn_path = outdir / f"{dataset_label}_{name}_attention.pdf"
            plot_attention_heatmap(attn, attn_path, f"{dataset_label} — transformer attention")
            figure_index.append({"file": attn_path.name, "kind": "attention", "model": name, "dataset": dataset_label})

        # model distribution figures are saved in a single metrics comparison later

    # ------------------
    # anomaly detector
    # ------------------
    if try_load_checkpoint(autoencoder, run_dir / "autoencoder.pt", device):
        anom_metrics, scores, threshold = evaluate_autoencoder(
            autoencoder, train_normal_loader, val_normal_loader, test_loader_all, y_test_binary, device
        )
        anom_metrics.update({
            "dataset": dataset_label,
            "model": "autoencoder",
            "train_run_dir": run_dir.name,
            "test_rows": int(len(test_idx)),
            "normal_rows": int((df.iloc[test_idx]["label"].astype(str).values == normal_label).sum()),
            "attack_rows": int((df.iloc[test_idx]["label"].astype(str).values != normal_label).sum()),
            "threshold": threshold,
        })
        rows.append(anom_metrics)

        auc, ap = plot_roc_pr(y_test_binary, scores, outdir, f"{dataset_label}_autoencoder")
        plot_anomaly_hist(scores, threshold, outdir / f"{dataset_label}_autoencoder_scores.pdf", f"{dataset_label} — autoencoder reconstruction errors")
        figure_index.extend([
            {"file": f"{dataset_label}_autoencoder_roc.pdf", "kind": "roc", "model": "autoencoder", "dataset": dataset_label},
            {"file": f"{dataset_label}_autoencoder_pr.pdf", "kind": "pr", "model": "autoencoder", "dataset": dataset_label},
            {"file": f"{dataset_label}_autoencoder_scores.pdf", "kind": "score_hist", "model": "autoencoder", "dataset": dataset_label},
        ])

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    metrics[metrics["task"] == "supervised"].to_csv(outdir / "metrics_supervised.csv", index=False)
    metrics[metrics["task"] == "anomaly"].to_csv(outdir / "metrics_anomaly.csv", index=False)

    # ------------------
    # summary plots
    # ------------------
    plot_metrics_supervised(metrics, outdir / f"{dataset_label}_supervised_metrics.pdf", f"{dataset_label} — supervised metrics")
    plot_metrics_anomaly(metrics, outdir / f"{dataset_label}_anomaly_metrics.pdf", f"{dataset_label} — anomaly metrics")
    figure_index.append({"file": f"{dataset_label}_supervised_metrics.pdf", "kind": "metrics", "model": "supervised", "dataset": dataset_label})
    figure_index.append({"file": f"{dataset_label}_anomaly_metrics.pdf", "kind": "metrics", "model": "autoencoder", "dataset": dataset_label})

    # ------------------
    # drift plots from long file if available
    # ------------------
    long_file = maybe_find_long_file(data_path)
    if long_file:
        long_df = load_dataset(long_file)
        plot_feature_drift(long_df, outdir, dataset_label)
        for feat in [f for f in CORE_DRIFT_FEATURES if f in long_df.columns]:
            figure_index.append({"file": f"{dataset_label}_drift_{feat}.pdf", "kind": "drift", "model": "all", "dataset": dataset_label})

    with open(outdir / "figure_index.json", "w", encoding="utf-8") as f:
        json.dump(figure_index, f, indent=2)

    # brief markdown summary for the directory
    md = [
        f"# Analysis summary: {dataset_label}",
        "",
        f"- dataset: `{data_path}`",
        f"- run dir: `{run_dir}`",
        f"- rows: {len(df)}",
        f"- supervised known-label test rows: {len(test_known_idx)}",
        f"- unknown rows in dataset: {int((~known_mask).sum())}",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
    ]
    (outdir / "analysis_summary.md").write_text("\n".join(md), encoding="utf-8")
    return metrics


# -----------------------------------------------------------------------------
# Cross-domain comparison
# -----------------------------------------------------------------------------
def find_metrics_csv(d: Path) -> Optional[Path]:
    direct = d / "metrics.csv"
    if direct.exists():
        return direct
    # fallback: search one level down for analysis outputs
    for child in d.iterdir() if d.exists() else []:
        if child.is_dir() and (child / "metrics.csv").exists():
            return child / "metrics.csv"
    return None


def compare_dirs(dirs: List[Path], outdir: Path, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in dirs:
        mpath = find_metrics_csv(d)
        if not mpath:
            print(f"[warn] missing metrics.csv for {d}")
            continue
        df = pd.read_csv(mpath)
        df["source_dir"] = d.name
        df["source_path"] = str(d)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No metrics.csv found in provided directories.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(outdir / "combined_metrics.csv", index=False)

    # supervised comparison
    sup = combined[combined["task"] == "supervised"].copy()
    if not sup.empty:
        for col in ["accuracy", "f1_macro", "roc_auc_ovr_macro"]:
            if col not in sup.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.8))
            pivot = sup.pivot_table(index="source_dir", columns="model", values=col, aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.index))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"Cross-domain supervised {col}")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            save_pdf(fig, outdir / f"cross_domain_supervised_{col}.pdf")

    # anomaly comparison
    anom = combined[combined["task"] == "anomaly"].copy()
    if not anom.empty:
        for col in ["accuracy", "f1_macro", "roc_auc", "average_precision"]:
            if col not in anom.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.8))
            pivot = anom.pivot_table(index="source_dir", columns="model", values=col, aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.index))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"Cross-domain anomaly {col}")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            save_pdf(fig, outdir / f"cross_domain_anomaly_{col}.pdf")

    # overall combined summary
    fig, ax = plt.subplots(figsize=(11, 5))
    if "f1_macro" in combined.columns:
        tmp = combined[combined["model"].isin(["qlstm", "lstm", "transformer"]) & (combined["task"] == "supervised")]
        if not tmp.empty:
            pivot = tmp.pivot_table(index="source_dir", columns="model", values="f1_macro", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Cross-domain supervised F1 macro")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            save_pdf(fig, outdir / "cross_domain_f1_macro.pdf")
    else:
        plt.close(fig)

    return combined


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to flattened TF-QKD CSV/Parquet")
    p.add_argument("--run-dir", type=str, default=".", help="Directory containing trained checkpoints/artifacts")
    p.add_argument("--outdir", type=str, default="analysis_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-dirs", nargs="*", default=None, help="Analysis directories to compare")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.compare_dirs:
        compare_dirs([Path(x) for x in args.compare_dirs], outdir, seed=args.seed)
        print(f"Cross-domain comparison saved to {outdir}")
        return

    if not args.data:
        raise SystemExit("--data is required unless --compare-dirs is used.")

    metrics = run_analysis(Path(args.data), Path(args.run_dir), outdir, seed=args.seed)
    print(metrics.to_string(index=False))
    print(f"Saved analysis to {outdir}")


if __name__ == "__main__":
    main()
