#!/usr/bin/env python3
"""
TF-QKD dataset factory.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_BASE = [
    'normal',
    'phase_drift_attack',
    'reference_light_tamper',
    'wavelength_switching_attack',
    'asymmetric_loss_attack',
    'synchronization_jitter_attack',
    'detector_blinding_attack',
    'combined_attack',
]
LABELS_UNKNOWN = LABELS_BASE + ['unknown_attack']
CORE_FEATURES = [
    'phase_lock_error_rad','phase_drift_rad','visibility','ref_power_t_dbm','ref_wavelength_t_nm',
    'sync_offset_ps','asym_loss_db','coincidence_rate','qber_phase','qber_bit','single_click_a','single_click_b'
]

@dataclass
class Config:
    variant: str = 'tfqkd_clean'
    seed: int = 42
    samples_per_class: int = 800
    unknown_samples: int = 800
    window_len: int = 32
    outdir: str = 'tfqkd_datasets'
    save_format: str = 'csv'


def clip01(x): return float(np.clip(x, 0.0, 1.0))

def binary_entropy(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return float(-(p*np.log2(p)+(1-p)*np.log2(1-p)))


def make_variant_params(cfg: Config):
    if cfg.variant == 'tfqkd_clean':
        return dict(rho_phi=0.992, sigma_phi=0.010, visibility_base=0.989, fiber_min=40.0, fiber_max=250.0, asym_loss_extra=1.0, lock_fail_base=0.003)
    if cfg.variant == 'tfqkd_drift':
        return dict(rho_phi=0.975, sigma_phi=0.035, visibility_base=0.984, fiber_min=40.0, fiber_max=350.0, asym_loss_extra=1.5, lock_fail_base=0.015)
    if cfg.variant == 'tfqkd_asym':
        return dict(rho_phi=0.988, sigma_phi=0.018, visibility_base=0.986, fiber_min=80.0, fiber_max=600.0, asym_loss_extra=8.0, lock_fail_base=0.006)
    if cfg.variant == 'tfqkd_unknown':
        return dict(rho_phi=0.985, sigma_phi=0.022, visibility_base=0.986, fiber_min=60.0, fiber_max=500.0, asym_loss_extra=4.0, lock_fail_base=0.010)
    raise ValueError(cfg.variant)


def sample_run_params(cfg, rng, vp):
    fiber_a = rng.uniform(vp['fiber_min'], vp['fiber_max'])
    fiber_b = rng.uniform(vp['fiber_min'], vp['fiber_max'])
    loss_a = 0.20*fiber_a + rng.normal(0, 0.5)
    loss_b = 0.20*fiber_b + rng.normal(0, 0.5)
    if cfg.variant == 'tfqkd_asym':
        if rng.random() < 0.7:
            loss_a += rng.uniform(2.0, 12.0)
        else:
            loss_b += rng.uniform(2.0, 12.0)
    if cfg.variant == 'tfqkd_drift':
        loss_a += rng.normal(0, 1.0); loss_b += rng.normal(0, 1.0)
    asym_loss = abs(loss_a - loss_b)
    return dict(
        fiber_len_a_km=fiber_a, fiber_len_b_km=fiber_b,
        loss_a_db=loss_a, loss_b_db=loss_b, asym_loss_db=asym_loss,
        eta_a=clip01(rng.normal(0.72,0.03)), eta_b=clip01(rng.normal(0.72,0.03)),
        dark_count_rate=abs(rng.normal(1e-6,5e-7)), afterpulse_rate=abs(rng.normal(1e-4,2e-5)),
        ctrl_gain=clip01(rng.normal(0.80,0.05)), ctrl_bandwidth_hz=abs(rng.normal(1e6,2e5)),
        ref_power_dbm=rng.normal(-15.0,0.5), ref_wavelength_nm=rng.normal(1550.12,0.02),
        base_sync_offset_ps=rng.normal(0,5.0), base_pol_mismatch_deg=abs(rng.normal(0,2.0)),
        mu_sig_a=rng.uniform(0.40,0.60), mu_sig_b=rng.uniform(0.40,0.60), mu_decoy_a=rng.uniform(0.08,0.20), mu_decoy_b=rng.uniform(0.08,0.20),
    )


def attack_profile(label, t, T, rng, vp):
    phase = 2*np.pi*t/max(T-1,1)
    a = dict(phi_bias=0.0, sigma_phi_scale=1.0, ref_power_bias=0.0, ref_lambda_bias=0.0, sync_bias=0.0, loss_bias=0.0, detector_bias=0.0, lock_fail_prob=vp['lock_fail_base'], unknown_mode=0)
    if label == 'normal': return a
    if label == 'phase_drift_attack': a['phi_bias'] = 0.04*np.sin(2*phase); a['sigma_phi_scale'] = 2.0 if vp['rho_phi'] < 0.99 else 1.6; a['lock_fail_prob'] += 0.015
    elif label == 'reference_light_tamper': a['ref_power_bias'] = 0.8*rng.choice([-1,1]); a['ref_lambda_bias'] = rng.normal(0.01,0.003); a['phi_bias'] = rng.normal(0.0,0.01)
    elif label == 'wavelength_switching_attack': delta = rng.choice([-1,1])*rng.uniform(0.02,0.15); a['ref_lambda_bias'] = delta; a['phi_bias'] = 0.05*delta; a['ref_power_bias'] = 0.15*np.sign(delta)
    elif label == 'asymmetric_loss_attack': a['loss_bias'] = rng.uniform(2.0,10.0) + vp['asym_loss_extra']
    elif label == 'synchronization_jitter_attack': a['sync_bias'] = rng.normal(0.0,20.0); a['phi_bias'] = 0.01*np.sin(4*phase)
    elif label == 'detector_blinding_attack': a['detector_bias'] = rng.uniform(0.3,0.8); a['lock_fail_prob'] += 0.02
    elif label == 'combined_attack': a['phi_bias'] = rng.normal(0.0,0.02); a['sigma_phi_scale'] = 1.5; a['ref_lambda_bias'] = rng.normal(0.03,0.01); a['sync_bias'] = rng.normal(0.0,10.0); a['loss_bias'] = rng.uniform(1.0,5.0) + 0.5*vp['asym_loss_extra']; a['detector_bias'] = rng.uniform(0.1,0.5); a['lock_fail_prob'] += 0.01
    elif label == 'unknown_attack': a['phi_bias'] = 0.03*np.sin(3*phase) + rng.normal(0.0,0.01); a['sigma_phi_scale'] = 1.8; a['ref_lambda_bias'] = rng.normal(0.06,0.02); a['sync_bias'] = rng.normal(0.0,18.0); a['loss_bias'] = rng.uniform(0.5,3.0); a['detector_bias'] = rng.uniform(0.05,0.25); a['lock_fail_prob'] += 0.03; a['unknown_mode'] = 1
    else: raise ValueError(label)
    return a


def evolve_step(cfg, rng, state, run, att, vp):
    rho = vp['rho_phi']; sigma = vp['sigma_phi']*att['sigma_phi_scale']
    phi = rho*state['phi'] + sigma*rng.normal() + att['phi_bias']
    phase_lock_error = phi - run['ctrl_gain']*phi
    lock_ok = 1
    if rng.random() < att['lock_fail_prob']:
        lock_ok = 0; phase_lock_error += rng.uniform(-np.pi, np.pi)
    ref_power_t = run['ref_power_dbm'] + rng.normal(0,0.1) + att['ref_power_bias']
    ref_lambda_t = run['ref_wavelength_nm'] + rng.normal(0,0.002) + att['ref_lambda_bias']
    sync_offset = run['base_sync_offset_ps'] + rng.normal(0,3) + att['sync_bias']
    pol = run['base_pol_mismatch_deg'] + rng.normal(0,0.3)
    loss_a = run['loss_a_db'] + att['loss_bias']; loss_b = run['loss_b_db']; asym_loss = abs(loss_a-loss_b)
    vis = vp['visibility_base']*np.exp(-0.35*abs(phase_lock_error)-0.015*abs(sync_offset)/10.0-0.01*abs(pol))
    vis *= np.exp(-0.03*abs(att['ref_lambda_bias'])); vis *= np.exp(-0.08*abs(att['ref_power_bias'])); vis = clip01(vis)
    coincidence_rate = 0.5*np.exp(-asym_loss/20.0)*vis*run['eta_a']*run['eta_b']; coincidence_rate += rng.normal(0,0.01); coincidence_rate = clip01(coincidence_rate)
    single_click_a = clip01(coincidence_rate*(1-att['detector_bias'])); single_click_b = clip01(coincidence_rate*(1-att['detector_bias']))
    qber_phase = float(np.clip(0.5*(1-vis)+0.01, 0, 0.5))
    qber_bit = float(np.clip(0.01 + 0.15*(1-vis) + 0.002*abs(sync_offset) + 0.001*run['dark_count_rate']*1e6, 0, 0.5))
    raw_key_rate = 1e6*coincidence_rate; secret_key_rate = raw_key_rate*max(0.0, 1.0-2.0*qber_phase)
    p1 = np.clip(0.5 + 0.5*(qber_bit-0.05), 1e-6, 1-1e-6); entropy = binary_entropy(p1)
    sifted_key_len = int(max(0, round(256*coincidence_rate*(1-qber_bit) + rng.normal(0,5))))
    anomaly_score_gt = float(0.45*(1-vis) + 0.25*min(asym_loss/15.0,1.0) + 0.15*min(abs(sync_offset)/50.0,1.0) + 0.15*min(abs(att['ref_lambda_bias'])/0.15,1.0))
    state['phi'] = phi
    return dict(phase_drift_rad=phi, phase_lock_error_rad=phase_lock_error, phase_lock_state=lock_ok, ref_power_t_dbm=ref_power_t, ref_wavelength_t_nm=ref_lambda_t, sync_offset_ps=sync_offset, polarization_mismatch_deg=pol, visibility=vis, coincidence_rate=coincidence_rate, single_click_a=single_click_a, single_click_b=single_click_b, qber_phase=qber_phase, qber_bit=qber_bit, sifted_key_len=sifted_key_len, raw_key_rate_bps=raw_key_rate, secret_key_rate_bps=secret_key_rate, measurement_entropy=entropy, asym_loss_db=asym_loss, attack_strength=float(abs(att['phi_bias']) + abs(att['ref_lambda_bias']) + abs(att['sync_bias'])/20.0 + abs(att['loss_bias'])/10.0 + att['detector_bias']), attack_mode_flag=0 if att['unknown_mode']==0 else 1, lock_loss_flag=0 if lock_ok else 1, anomaly_score_gt=anomaly_score_gt)

def make_sample(cfg, rng, label, sample_id, vp):
    run = sample_run_params(cfg, rng, vp)
    state = {"phi": 0.0}
    rows_long = []
    seq_rows = []

    for t in range(cfg.window_len):
        att = attack_profile(label, t, cfg.window_len, rng, vp)
        telem = evolve_step(cfg, rng, state, run, att, vp)

        basis_a = rng.integers(0, 2)
        basis_b = rng.integers(0, 2)
        send_flag_a = 1
        send_flag_b = 1
        intensity_a = run["mu_sig_a"] if rng.random() < 0.7 else run["mu_decoy_a"]
        intensity_b = run["mu_sig_b"] if rng.random() < 0.7 else run["mu_decoy_b"]
        phase_encode_a = rng.uniform(-np.pi, np.pi)
        phase_encode_b = rng.uniform(-np.pi, np.pi)

        run_copy = run.copy()
        if "asym_loss_db" in run_copy:
            run_copy["base_asym_loss_db"] = run_copy.pop("asym_loss_db")

        row = {
            "sample_id": sample_id,
            "run_id": sample_id,
            "window_id": sample_id,
            "t": t,
            "label": label,
            "protocol_variant": "SNS-TF-QKD",
            "basis_a": basis_a,
            "basis_b": basis_b,
            "send_flag_a": send_flag_a,
            "send_flag_b": send_flag_b,
            "intensity_a": intensity_a,
            "intensity_b": intensity_b,
            "phase_encode_a_rad": phase_encode_a,
            "phase_encode_b_rad": phase_encode_b,
        }

        row.update(run_copy)
        row.update(telem)

        if label == "phase_drift_attack":
            row["anomaly_score_gt"] += 0.05
        elif label == "unknown_attack":
            row["anomaly_score_gt"] += 0.12

        rows_long.append(row)
        seq_rows.append(row)

    flat = {
        "sample_id": sample_id,
        "label": label,
        "protocol_variant": "SNS-TF-QKD",
    }

    for t in range(cfg.window_len):
        for feat in CORE_FEATURES:
            flat[f"{feat}_t{t:02d}"] = seq_rows[t][feat]

    return rows_long, flat

def save_dataset(df_long, df_flat, cfg):
    outdir = Path(cfg.outdir) / cfg.variant; outdir.mkdir(parents=True, exist_ok=True)
    if cfg.save_format == 'parquet':
        df_long.to_parquet(outdir / 'tfqkd_long.parquet', index=False); df_flat.to_parquet(outdir / 'tfqkd_flat.parquet', index=False)
    else:
        df_long.to_csv(outdir / 'tfqkd_long.csv', index=False); df_flat.to_csv(outdir / 'tfqkd_flat.csv', index=False)
    with open(outdir / 'config.json', 'w', encoding='utf-8') as f: json.dump(asdict(cfg), f, indent=2)
    with open(outdir / 'label_map.json', 'w', encoding='utf-8') as f: json.dump({lab:i for i,lab in enumerate(sorted(df_flat['label'].unique().tolist()))}, f, indent=2)
    manifest = {'variant': cfg.variant, 'n_samples_flat': int(len(df_flat)), 'n_rows_long': int(len(df_long)), 'labels': sorted(df_flat['label'].unique().tolist()), 'core_features': CORE_FEATURES, 'window_len': cfg.window_len}
    with open(outdir / 'manifest.json', 'w', encoding='utf-8') as f: json.dump(manifest, f, indent=2)
    return outdir


def generate_one(cfg):
    rng = np.random.default_rng(cfg.seed); vp = make_variant_params(cfg)
    labels = LABELS_UNKNOWN if cfg.variant == 'tfqkd_unknown' else LABELS_BASE
    rows_long = []; rows_flat = []; sample_id = 0
    for label in labels:
        n_samples = cfg.unknown_samples if (cfg.variant == 'tfqkd_unknown' and label == 'unknown_attack') else cfg.samples_per_class
        for _ in range(n_samples):
            sample_id += 1
            long_rows, flat_row = make_sample(cfg, rng, label, sample_id, vp)
            rows_long.extend(long_rows); rows_flat.append(flat_row)
    df_long = pd.DataFrame(rows_long); df_flat = pd.DataFrame(rows_flat)
    if cfg.variant != 'tfqkd_unknown':
        idx = np.arange(len(df_flat)); y = df_flat['label'].values
        train_idx, temp_idx = train_test_split(idx, train_size=0.70, stratify=y, random_state=cfg.seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, stratify=y[temp_idx], random_state=cfg.seed)
        split = np.array(['test']*len(df_flat), dtype=object); split[train_idx]='train'; split[val_idx]='val'; split[test_idx]='test'; df_flat['split']=split
    else:
        known_mask = df_flat['label'] != 'unknown_attack'; idx = np.where(known_mask)[0]; y = df_flat.loc[known_mask, 'label'].values
        train_idx, temp_idx = train_test_split(idx, train_size=0.70, stratify=y, random_state=cfg.seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, stratify=df_flat.loc[temp_idx,'label'].values, random_state=cfg.seed)
        split = np.array(['test']*len(df_flat), dtype=object); split[train_idx]='train'; split[val_idx]='val'; split[test_idx]='test'; split[df_flat['label'].values=='unknown_attack']='test'; df_flat['split']=split
    outdir = save_dataset(df_long, df_flat, cfg)
    print(f'[{cfg.variant}] saved to {outdir}')
    print(df_flat['label'].value_counts().sort_index().to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', type=str, default='tfqkd_clean', choices=['tfqkd_clean','tfqkd_drift','tfqkd_asym','tfqkd_unknown'])
    p.add_argument('--all', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples-per-class', type=int, default=800)
    p.add_argument('--unknown-samples', type=int, default=800)
    p.add_argument('--window-len', type=int, default=32)
    p.add_argument('--outdir', type=str, default='tfqkd_datasets')
    p.add_argument('--save-format', type=str, default='csv', choices=['csv','parquet'])
    args = p.parse_args()
    if args.all:
        for v in ['tfqkd_clean','tfqkd_drift','tfqkd_asym','tfqkd_unknown']:
            generate_one(Config(variant=v, seed=args.seed, samples_per_class=args.samples_per_class, unknown_samples=args.unknown_samples, window_len=args.window_len, outdir=args.outdir, save_format=args.save_format))
    else:
        generate_one(Config(variant=args.variant, seed=args.seed, samples_per_class=args.samples_per_class, unknown_samples=args.unknown_samples, window_len=args.window_len, outdir=args.outdir, save_format=args.save_format))

if __name__ == '__main__': main()
