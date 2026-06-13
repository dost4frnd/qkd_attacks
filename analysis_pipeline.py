#!/usr/bin/env python3
"""
TF-QKD analysis pipeline.
"""
from __future__ import annotations
import argparse, json, math, pickle, re
from pathlib import Path
from typing import List, Sequence, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
TIMESTEP_RE = re.compile(r'^(?P<base>.+)_t(?P<t>\d+)$')
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def infer_feature_layout(columns: Sequence[str]) -> Tuple[List[str], int]:
    groups: Dict[str, List[int]] = {}
    for c in columns:
        m = TIMESTEP_RE.match(c)
        if not m: continue
        groups.setdefault(m.group('base'), []).append(int(m.group('t')))
    bases = sorted(groups.keys())
    lengths = {len(set(ts)) for ts in groups.values()}
    if len(lengths) != 1: raise ValueError(f'inconsistent seq lengths: {lengths}')
    return bases, lengths.pop()
def flatten_to_3d(df: pd.DataFrame, bases: List[str], seq_len: int) -> np.ndarray:
    X = np.zeros((len(df), seq_len, len(bases)), dtype=np.float32)
    for fi, feat in enumerate(bases):
        for t in range(seq_len):
            X[:, t, fi] = df[f'{feat}_t{t:02d}'].astype(np.float32).values
    return X
class SequenceDataset(Dataset):
    def __init__(self, X, y, idx): self.X=torch.tensor(X[idx], dtype=torch.float32); self.y=torch.tensor(y[idx], dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
class UnlabeledSequenceDataset(Dataset):
    def __init__(self, X, idx): self.X=torch.tensor(X[idx], dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]
def make_loader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def save_pdf(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__(); pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model)); pe[:,0::2] = torch.sin(position*div_term); pe[:,1::2] = torch.cos(position*div_term); self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.1):
        super().__init__(); self.lstm=nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0); self.head=nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, num_classes))
    def forward(self, x): _, (h_n, _) = self.lstm(x); return self.head(h_n[-1])
    def features(self, x): _, (h_n, _) = self.lstm(x); return h_n[-1]
class QuantumInspiredCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__(); self.hidden_dim=hidden_dim; self.gate=nn.Linear(input_dim*4 + hidden_dim, 4*hidden_dim); self.dropout=nn.Dropout(dropout)
    def feature_map(self, x): return torch.cat([x, torch.sin(x), torch.cos(x), x*x], dim=-1)
    def forward(self, x, h, c): xm=self.dropout(self.feature_map(x)); gates=self.gate(torch.cat([xm,h], dim=-1)); i,f,g,o=gates.chunk(4, dim=-1); i=torch.sigmoid(i); f=torch.sigmoid(f); g=torch.tanh(g); o=torch.sigmoid(o); c_new=f*c+i*g; h_new=o*torch.tanh(c_new); return h_new, c_new
class HybridQLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__(); self.cell=QuantumInspiredCell(input_dim, hidden_dim, dropout=dropout); self.post=nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)); self.head=nn.Linear(hidden_dim, num_classes); self.hidden_dim=hidden_dim
    def forward(self, x):
        b, t, _ = x.shape; h=torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype); c=torch.zeros_like(h)
        for i in range(t): h, c = self.cell(x[:,i,:], h, c)
        return self.head(self.post(h))
    def features(self, x):
        b, t, _ = x.shape; h=torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype); c=torch.zeros_like(h)
        for i in range(t): h, c = self.cell(x[:,i,:], h, c)
        return self.post(h)
class TinyTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, dropout=0.1, max_len=512):
        super().__init__(); self.input_proj=nn.Linear(input_dim, d_model); self.posenc=SinusoidalPositionalEncoding(d_model, max_len=max_len); enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True); self.encoder=nn.TransformerEncoder(enc_layer, num_layers=num_layers); self.head=nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, num_classes)); self.d_model=d_model; self.num_heads=num_heads
    def forward(self, x): h=self.input_proj(x); h=self.posenc(h); h=self.encoder(h); return self.head(h.mean(dim=1))
    def features(self, x): h=self.input_proj(x); h=self.posenc(h); h=self.encoder(h); return h.mean(dim=1)
class SequenceAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, dropout=0.1):
        super().__init__(); self.encoder=nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True); self.to_latent=nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim), nn.GELU(), nn.Dropout(dropout)); self.latent_to_hidden=nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU()); self.decoder=nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True); self.recon_head=nn.Linear(hidden_dim, input_dim)
    def forward(self, x): _, (h_n, _) = self.encoder(x); latent=self.to_latent(h_n[-1]); init_h=self.latent_to_hidden(latent).unsqueeze(0); init_c=torch.zeros_like(init_h); dec_in=torch.zeros_like(x); dec_out, _ = self.decoder(dec_in, (init_h, init_c)); return self.recon_head(dec_out)
def load_dataset(path: Path): return pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)
def fit_scaler_on_train(X_train): n,t,f = X_train.shape; sc = StandardScaler(); sc.fit(X_train.reshape(n*t, f)); return sc
def apply_scaler(X, sc): n,t,f=X.shape; return sc.transform(X.reshape(n*t,f)).reshape(n,t,f).astype(np.float32)
def get_splits(df, y, seed):
    if 'split' in df.columns and set(df['split'].astype(str).unique()).issuperset({'train','val','test'}):
        return np.where(df['split'].astype(str).values=='train')[0], np.where(df['split'].astype(str).values=='val')[0], np.where(df['split'].astype(str).values=='test')[0]
    from sklearn.model_selection import train_test_split
    idx=np.arange(len(df)); train_idx, temp_idx = train_test_split(idx, train_size=0.70, stratify=y, random_state=seed); val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, stratify=y[temp_idx], random_state=seed); return train_idx, val_idx, test_idx
def load_artifacts(run_dir: Path):
    with open(run_dir / 'scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open(run_dir / 'label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
    with open(run_dir / 'feature_layout.json', 'r', encoding='utf-8') as f: fl = json.load(f)
    return scaler, label_encoder, fl['base_features'], int(fl['seq_len'])
def build_models(input_dim, num_classes, seq_len, device, run_dir):
    cfg = {'hidden_dim':64, 'latent_dim':16, 'dropout':0.1, 'num_layers':1, 'transformer_layers':2, 'num_heads':4}
    if (run_dir / 'run_config.json').exists():
        try: cfg.update(json.loads((run_dir / 'run_config.json').read_text()))
        except Exception: pass
    d_model = max(cfg['hidden_dim'], 32)
    if d_model % cfg['num_heads'] != 0: d_model = cfg['num_heads'] * ((d_model // cfg['num_heads']) + 1)
    return (
        HybridQLSTMClassifier(input_dim, cfg['hidden_dim'], num_classes, dropout=cfg['dropout']).to(device),
        LSTMClassifier(input_dim, cfg['hidden_dim'], num_classes, num_layers=cfg['num_layers'], dropout=cfg['dropout']).to(device),
        TinyTransformerClassifier(input_dim, d_model, cfg['num_heads'], cfg['transformer_layers'], num_classes, dropout=cfg['dropout'], max_len=max(seq_len, 512)).to(device),
        SequenceAutoencoder(input_dim, cfg['hidden_dim'], cfg['latent_dim'], dropout=cfg['dropout']).to(device)
    )
def try_load_checkpoint(model, ckpt_path, device):
    if not ckpt_path.exists(): return False
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device).eval(); return True
def save_confusion_matrix(cm, labels, outpath, title):
    fig, ax = plt.subplots(figsize=(8,6)); im=ax.imshow(cm, interpolation='nearest'); fig.colorbar(im, ax=ax); ax.set_title(title); ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels); thresh = cm.max()/2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', color='white' if cm[i,j] > thresh else 'black', fontsize=8)
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label'); save_pdf(fig, outpath)
def plot_feature_drift(long_df: pd.DataFrame, outdir: Path, prefix: str):
    features = ['phase_lock_error_rad', 'visibility', 'ref_power_t_dbm', 'ref_wavelength_t_nm', 'coincidence_rate', 'qber_phase']
    features = [f for f in features if f in long_df.columns]
    if not features:
        return
    grouped = long_df.groupby(['label', 't'])[features].mean().reset_index()
    labels = grouped['label'].unique().tolist()
    for feat in features:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in labels:
            tmp = grouped[grouped['label'] == label]
            ax.plot(tmp['t'], tmp[feat], label=label)
        ax.set_title(f'{prefix}: drift over time — {feat}')
        ax.set_xlabel('timestep')
        ax.set_ylabel(feat)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        save_pdf(fig, outdir / f'{prefix}_drift_{feat}.pdf')
def plot_tsne(embeddings, labels, label_names, outpath, title):
    n = len(embeddings)
    if n > 2500:
        idx = np.random.choice(n, 2500, replace=False); embeddings = embeddings[idx]; labels = labels[idx]
    pca = PCA(n_components=min(30, embeddings.shape[1]), random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=max(5, min(40, len(embeddings)//20)), init='pca', learning_rate='auto', random_state=42)
    xy = tsne.fit_transform(emb_pca)
    fig, ax = plt.subplots(figsize=(8,6))
    for i, name in enumerate(label_names):
        m = labels == i
        if m.any(): ax.scatter(xy[m,0], xy[m,1], s=8, alpha=0.7, label=name)
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); save_pdf(fig, outpath)
def plot_roc_pr(y_true, scores, outdir, prefix='autoencoder'):
    fpr, tpr, _ = roc_curve(y_true, scores); auc = roc_auc_score(y_true, scores); fig, ax = plt.subplots(figsize=(6,5)); ax.plot(fpr, tpr, label=f'AUC={auc:.3f}'); ax.plot([0,1],[0,1], linestyle='--'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title(f'ROC — {prefix}'); ax.legend(); fig.tight_layout(); fig.savefig(outdir / f'{prefix}_roc.pdf', dpi=200, bbox_inches='tight'); plt.close(fig)
    precision, recall, _ = precision_recall_curve(y_true, scores); ap = average_precision_score(y_true, scores); fig, ax = plt.subplots(figsize=(6,5)); ax.plot(recall, precision, label=f'AP={ap:.3f}'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(f'PR — {prefix}'); ax.legend(); fig.tight_layout(); fig.savefig(outdir / f'{prefix}_pr.pdf', dpi=200, bbox_inches='tight'); plt.close(fig)
def plot_metrics(metrics_df, outpath):
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(metrics_df)); w = 0.22
    if 'accuracy' in metrics_df.columns: ax.bar(x-w, metrics_df['accuracy'], width=w, label='accuracy')
    if 'f1_macro' in metrics_df.columns: ax.bar(x, metrics_df['f1_macro'], width=w, label='f1')
    if 'roc_auc_ovr_macro' in metrics_df.columns: ax.bar(x+w, metrics_df['roc_auc_ovr_macro'].fillna(0.0), width=w, label='roc_auc_ovr')
    if 'roc_auc' in metrics_df.columns: ax.bar(x+w*2, metrics_df['roc_auc'].fillna(0.0), width=w, label='roc_auc')
    ax.set_xticks(x); ax.set_xticklabels(metrics_df['model'].tolist(), rotation=30, ha='right'); ax.set_ylim(0,1.05); ax.legend(); ax.set_title('Model metrics comparison'); save_pdf(fig, outpath)
def extract_transformer_attention(model: TinyTransformerClassifier, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        h = model.input_proj(x); h = model.posenc(h)
        attn_mats = []
        out = h
        for layer in model.encoder.layers:
            x1 = layer.norm1(out) if layer.norm_first else out
            attn_out, attn_w = layer.self_attn(x1, x1, x1, need_weights=True, average_attn_weights=False)
            attn_mats.append(attn_w.detach().cpu().numpy())
            out = out + layer.dropout1(attn_out)
            x2 = layer.norm2(out) if layer.norm_first else out
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            out = out + layer.dropout2(ff)
        attn = np.mean(np.stack(attn_mats, axis=0), axis=(0,1,2))
        return attn
def plot_attention(attn, outpath, title='Transformer attention'):
    fig, ax = plt.subplots(figsize=(7,6)); im=ax.imshow(attn, aspect='auto'); fig.colorbar(im, ax=ax); ax.set_title(title); ax.set_xlabel('Key timestep'); ax.set_ylabel('Query timestep'); save_pdf(fig, outpath)
@torch.no_grad()
def predict_classifier(model, loader, device):
    model.eval(); probs_all=[]; logits_all=[]; y_all=[]; feats_all=[]
    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = model(Xb); probs = torch.softmax(logits, dim=-1)
        logits_all.append(logits.cpu().numpy()); probs_all.append(probs.cpu().numpy()); feats_all.append(model.features(Xb).cpu().numpy() if hasattr(model, 'features') else probs.cpu().numpy())
        if yb is not None: y_all.append(yb.numpy())
    return np.concatenate(logits_all), np.concatenate(probs_all), np.concatenate(y_all), np.concatenate(feats_all)
@torch.no_grad()
def reconstruction_errors(model, loader, device):
    model.eval(); errs=[]
    for Xb in loader:
        Xb = Xb.to(device)
        recon = model(Xb)
        errs.append(torch.mean((recon - Xb) ** 2, dim=(1,2)).cpu().numpy())
    return np.concatenate(errs)
def run_one_dataset(data_path: Path, run_dir: Path, outdir: Path, seed=42):
    set_seed(seed); outdir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    bases, seq_len = infer_feature_layout([c for c in df.columns if '_t' in c])
    X = flatten_to_3d(df, bases, seq_len)
    scaler, label_encoder, saved_bases, saved_seq_len = load_artifacts(run_dir)
    X = apply_scaler(X, scaler)
    labels_str = df['label'].astype(str).values

    known_classes = set(label_encoder.classes_)

    mapped_labels = []

    for lbl in labels_str:

        if lbl in known_classes:
            mapped_labels.append(
                label_encoder.transform([lbl])[0]
            )

        else:
            # unknown class handling
            mapped_labels.append(-1)

    y = np.array(mapped_labels)
    train_idx, val_idx, test_idx = get_splits(df, y, seed)
    test_ds = SequenceDataset(X, y, test_idx)
    test_loader = make_loader(test_ds, 64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qlstm, lstm, transformer, autoencoder = build_models(X.shape[-1], len(label_encoder.classes_), seq_len, device, run_dir)
    ckpts = {'qlstm': run_dir / 'qlstm.pt', 'lstm': run_dir / 'lstm.pt', 'transformer': run_dir / 'transformer.pt', 'autoencoder': run_dir / 'autoencoder.pt'}
    loaded = {name: try_load_checkpoint(model, ckpts[name], device) for name, model in [('qlstm', qlstm), ('lstm', lstm), ('transformer', transformer), ('autoencoder', autoencoder)]}
    print('Loaded:', loaded)
    results = []
    preds = {}
    for name, model in [('qlstm', qlstm), ('lstm', lstm), ('transformer', transformer)]:
        logits, probs, y_true, feats = predict_classifier(model, test_loader, device)
        y_pred = probs.argmax(axis=1)
        preds[name] = (y_true, y_pred, probs, feats)
        metrics = dict(model=name, accuracy=accuracy_score(y_true, y_pred), precision_macro=precision_score(y_true, y_pred, average='macro', zero_division=0), recall_macro=recall_score(y_true, y_pred, average='macro', zero_division=0), f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0))
        try: metrics['roc_auc_ovr_macro'] = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        except Exception: metrics['roc_auc_ovr_macro'] = np.nan
        results.append(metrics)
        save_confusion_matrix(confusion_matrix(y_true, y_pred), label_encoder.classes_, outdir / f'{name}_confusion_matrix.pdf', f'{name} confusion matrix')
        known_mask = y_true != -1

        print(
            f"\n[{name}]\n"
            f"{classification_report(
                y_true[known_mask],
                y_pred[known_mask],
                target_names=label_encoder.classes_,
                zero_division=0
            )}"
        )
    normal_label = label_encoder.transform(['normal'])[0] if 'normal' in label_encoder.classes_ else 0
    train_normal_idx = train_idx[y[train_idx] == normal_label]
    val_normal_idx = val_idx[y[val_idx] == normal_label]
    train_normal_loader = make_loader(UnlabeledSequenceDataset(X, train_normal_idx), 64, shuffle=False)
    val_normal_loader = make_loader(UnlabeledSequenceDataset(X, val_normal_idx), 64, shuffle=False)
    test_all_loader = make_loader(UnlabeledSequenceDataset(X, test_idx), 64, shuffle=False)
    if loaded['autoencoder']:
        tr_err = reconstruction_errors(autoencoder, train_normal_loader, device)
        val_err = reconstruction_errors(autoencoder, val_normal_loader, device)
        test_err = reconstruction_errors(autoencoder, test_all_loader, device)
        threshold = float(np.quantile(tr_err if len(tr_err) else val_err, 0.95))
        y_true_binary = (y[test_idx] != normal_label).astype(int)
        y_pred_binary = (test_err > threshold).astype(int)
        metrics = dict(model='autoencoder', accuracy=accuracy_score(y_true_binary, y_pred_binary), precision_macro=precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0), recall_macro=recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0), f1_macro=f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0))
        try: metrics['roc_auc'] = roc_auc_score(y_true_binary, test_err); metrics['average_precision'] = average_precision_score(y_true_binary, test_err)
        except Exception: metrics['roc_auc'] = np.nan; metrics['average_precision'] = np.nan
        results.append(metrics)
        plot_roc_pr(y_true_binary, test_err, outdir, prefix='autoencoder')
        fig, ax = plt.subplots(figsize=(8,5)); ax.hist(test_err, bins=40, alpha=0.8); ax.axvline(threshold, linestyle='--'); ax.set_title('Autoencoder reconstruction error distribution'); ax.set_xlabel('MSE'); ax.set_ylabel('count'); save_pdf(fig, outdir / 'autoencoder_scores_hist.pdf')
        score_df = pd.DataFrame({'label': df.iloc[test_idx]['label'].values, 'recon_error': test_err}); score_df.groupby('label')['recon_error'].mean().sort_values(ascending=False).to_csv(outdir / 'anomaly_scores_by_class.csv')
    metrics_df = pd.DataFrame(results); metrics_df.to_csv(outdir / 'metrics.csv', index=False)
    plot_metrics(metrics_df, outdir / 'metrics_comparison.pdf')
    # drift plots
    long_path = data_path.parent / 'tfqkd_long.csv'
    if not long_path.exists(): long_path = data_path.parent / 'tfqkd_long.parquet'
    if long_path.exists():
        long_df = pd.read_parquet(long_path) if long_path.suffix == '.parquet' else pd.read_csv(long_path)
        plot_feature_drift(long_df, outdir, prefix=data_path.parent.name)
    # t-SNE
    if 'transformer' in preds:
        _, _, _, feats = preds['transformer']
        plot_tsne(feats, y[test_idx], label_encoder.classes_, outdir / 'tsne_transformer_embeddings.pdf', 't-SNE of transformer embeddings')
    # attention
    if loaded['transformer']:
        xb, _ = next(iter(test_loader)); xb = xb.to(device)[: min(32, xb.size(0))]
        attn = extract_transformer_attention(transformer, xb)
        plot_attention(attn, outdir / 'transformer_attention.pdf', 'Average transformer attention')
    for name, (y_true, y_pred, probs, feats) in preds.items():
        pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        for i, cls in enumerate(label_encoder.classes_): pred_df[f'p_{cls}'] = probs[:, i]
        pred_df.to_csv(outdir / f'{name}_predictions.csv', index=False)
    return metrics_df
def compare_runs(run_dirs: List[Path], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in run_dirs:
        mpath = rd / 'metrics.csv'
        if not mpath.exists():
            print(f'[warn] missing {mpath}')
            continue
        df = pd.read_csv(mpath); df['run_dir'] = rd.name; rows.append(df)
    if not rows: raise FileNotFoundError('No metrics.csv found in the provided run directories.')
    allm = pd.concat(rows, ignore_index=True); allm.to_csv(outdir / 'combined_metrics.csv', index=False)
    cols = [c for c in ['accuracy','precision_macro','recall_macro','f1_macro','roc_auc','average_precision','roc_auc_ovr_macro'] if c in allm.columns]
    models = allm['model'].unique().tolist()
    for col in cols:
        fig, ax = plt.subplots(figsize=(10,5))
        for model in models:
            tmp = allm[allm['model']==model]
            ax.plot(tmp['run_dir'], tmp[col], marker='o', label=model)
        ax.set_title(f'Cross-domain robustness: {col}'); ax.set_ylabel(col); ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=8); ax.grid(alpha=0.3); save_pdf(fig, outdir / f'cross_domain_{col}.pdf')
    fig, ax = plt.subplots(figsize=(12,6))
    for model in models:
        tmp = allm[allm['model']==model]
        if 'f1_macro' in tmp.columns: ax.plot(tmp['run_dir'], tmp['f1_macro'], marker='o', label=model)
    ax.set_title('Cross-domain F1 macro comparison'); ax.set_ylabel('F1 macro'); ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=8); ax.grid(alpha=0.3); save_pdf(fig, outdir / 'cross_domain_f1_macro.pdf')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--run-dir', type=str, default='.')
    p.add_argument('--outdir', type=str, default='analysis_out')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compare-dirs', nargs='*', default=None)
    args = p.parse_args()
    outdir = Path(args.outdir)
    if args.compare_dirs:
        compare_runs([Path(x) for x in args.compare_dirs], outdir)
        print(f'Cross-domain comparison saved to {outdir}')
        return
    if not args.data:
        raise SystemExit('--data is required unless --compare-dirs is used')
    metrics_df = run_one_dataset(Path(args.data), Path(args.run_dir), outdir, seed=args.seed)
    print('\nSaved analysis to:', outdir.resolve())
    print(metrics_df.to_string(index=False))
if __name__ == '__main__': main()
