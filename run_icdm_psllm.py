# -*- coding: utf-8 -*-
"""
ICDM-style Pattern- and Semantics-Augmented LLM Forecasting Benchmark.

Working directory:
    ~/LLM

Expected data:
    ~/LLM/data_icdm_tsf/
        ETTh1/ETTh1.csv
        ETTh2/ETTh2.csv
        ETTm1/ETTm1.csv
        ETTm2/ETTm2.csv
        Weather/Weather.csv
        Traffic/Traffic.csv
        Electricity/Electricity.csv
        Exchange/Exchange.csv
        ILI/ILI.csv
        M4/Train/Hourly-train.csv etc. optional

Expected models:
    /home/tahiti/Malashin_Projects/hf_models/
        opt_350m/
        opt_1p3b/
        phi3p5_mini_instruct/
        pythia_410m/
        pythia_1b/
        qwen2p5_0p5b_instruct/
        qwen2p5_1p5b/
        qwen2p5_1p5b_instruct/
        qwen2p5_3b/
        smollm2_1p7b/
        tinyllama_1p1b_chat/
        bert_tiny/
        bert_mini/
        distilbert_base_uncased/
        electra_small_discriminator/

Implemented models:
    last
    dlinear
    nlinear
    patchtst
    itransformer
    llm_direct
    psllm

Implemented plans:
    single
    main
    ablations
    backbones
    semantic
    noise
    fewshot
    m4

Main method:
    PS-LLM = pattern encoder + semantic encoder + alignment + gated fusion
             + frozen local pretrained LLM + forecasting head.

Important:
    The code uses local_files_only=True for HuggingFace models.
"""

import os
import gc
import json
import math
import time
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ============================================================
# Dataset and model registries
# ============================================================

MAIN_DATASETS = {
    "ETTh1": "ETTh1/ETTh1.csv",
    "ETTh2": "ETTh2/ETTh2.csv",
    "ETTm1": "ETTm1/ETTm1.csv",
    "ETTm2": "ETTm2/ETTm2.csv",
    "Weather": "Weather/Weather.csv",
    "Traffic": "Traffic/Traffic.csv",
    "Electricity": "Electricity/Electricity.csv",
    "Exchange": "Exchange/Exchange.csv",
    "ILI": "ILI/ILI.csv",
}

DEFAULT_HORIZONS = {
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "Weather": [96, 192, 336, 720],
    "Traffic": [96, 192, 336, 720],
    "Electricity": [96, 192, 336, 720],
    "Exchange": [96, 192, 336, 720],
    "ILI": [24, 36, 48, 60],
}

M4_HORIZONS = {
    "Hourly": 48,
    "Daily": 14,
    "Weekly": 13,
    "Monthly": 18,
    "Quarterly": 8,
    "Yearly": 6,
}

LLM_DIRS = {
    "opt_350m": "opt_350m",
    "opt_1p3b": "opt_1p3b",
    "phi3p5_mini_instruct": "phi3p5_mini_instruct",
    "pythia_410m": "pythia_410m",
    "pythia_1b": "pythia_1b",
    "qwen2p5_0p5b_instruct": "qwen2p5_0p5b_instruct",
    "qwen2p5_1p5b": "qwen2p5_1p5b",
    "qwen2p5_1p5b_instruct": "qwen2p5_1p5b_instruct",
    "qwen2p5_3b": "qwen2p5_3b",
    "smollm2_1p7b": "smollm2_1p7b",
    "tinyllama_1p1b_chat": "tinyllama_1p1b_chat",
}

ENCODER_DIRS = {
    "bert_tiny": "bert_tiny",
    "bert_mini": "bert_mini",
    "distilbert_base_uncased": "distilbert_base_uncased",
    "electra_small_discriminator": "electra_small_discriminator",
}


# ============================================================
# Config
# ============================================================

@dataclass
class ExpConfig:
    data_root: str = "data_icdm_tsf"
    hf_root: str = "/home/tahiti/Malashin_Projects/hf_models"
    out_dir: str = "runs_icdm_psllm"

    dataset: str = "ETTh1"
    task: str = "main"
    mode: str = "M"

    model: str = "psllm"
    llm_name: str = "qwen2p5_1p5b"
    semantic_encoder: str = "mlp"

    seq_len: int = 96
    pred_len: int = 96

    train_ratio: float = 0.7
    val_ratio: float = 0.1

    batch_size: int = 8
    epochs: int = 10
    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    rank: int = 64

    cnn_kernels: str = "3,5,7,13"
    align_lambda: float = 0.05

    use_pattern: int = 1
    use_semantic: int = 1
    use_align: int = 1
    use_gate: int = 1
    use_llm: int = 1
    freeze_llm: int = 1

    noise_std: float = 0.0
    fewshot_ratio: float = 1.0

    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    amp: int = 1

    max_train_batches: int = -1
    max_eval_batches: int = -1

    m4_freq: str = "Hourly"
    m4_max_series: int = -1


# ============================================================
# General utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def resolve_existing_root(path: str) -> Path:
    candidates = [
        Path(path),
        Path("/home/tahiti/Malashin_Projects/hf_models"),
        Path("../Malashin_Projects/hf_models"),
        Path("Malashin_Projects/hf_models"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(path)


def resolve_model_dir(hf_root: Path, name: str, is_encoder: bool = False) -> Path:
    table = ENCODER_DIRS if is_encoder else LLM_DIRS
    dirname = table.get(name, name)
    p = hf_root / dirname

    if p.exists() and (p / "config.json").exists():
        return p

    if p.exists():
        found = list(p.rglob("config.json"))
        if found:
            return found[0].parent

    raise FileNotFoundError(
        f"Cannot find local model directory for '{name}'. Tried: {p}"
    )


def cfg_from_args(args, **updates):
    d = vars(args).copy()
    d.update(updates)
    allowed = set(ExpConfig.__dataclass_fields__.keys())
    return ExpConfig(**{k: v for k, v in d.items() if k in allowed})


# ============================================================
# Data
# ============================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray):
        return x * self.std + self.mean


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def choose_columns(df: pd.DataFrame, mode: str) -> Tuple[List[str], List[str]]:
    nums = get_numeric_columns(df)
    if len(nums) == 0:
        raise RuntimeError("No numeric columns found")

    if mode == "M":
        return nums, nums

    if mode == "MS":
        target = "OT" if "OT" in nums else nums[-1]
        return nums, [target]

    if mode == "S":
        target = "OT" if "OT" in nums else nums[-1]
        return [target], [target]

    raise ValueError("mode must be M, MS, or S")


class MainTSFDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        seq_len: int,
        pred_len: int,
        mode: str,
        split: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        scaler: Optional[Tuple[StandardScaler, StandardScaler]] = None,
        fewshot_ratio: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.split = split
        self.noise_std = noise_std

        df = pd.read_csv(csv_path)
        x_cols, y_cols = choose_columns(df, mode)

        self.input_cols = x_cols
        self.target_cols = y_cols

        x = df[x_cols].values.astype(np.float32)
        y = df[y_cols].values.astype(np.float32)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if scaler is None:
            sx = StandardScaler()
            sy = StandardScaler()
            sx.fit(x[:train_end])
            sy.fit(y[:train_end])
            self.scaler = (sx, sy)
        else:
            sx, sy = scaler
            self.scaler = scaler

        x = sx.transform(x).astype(np.float32)
        y = sy.transform(y).astype(np.float32)

        if split == "train":
            start, end = 0, train_end
        elif split == "val":
            start, end = max(0, train_end - seq_len), val_end
        elif split == "test":
            start, end = max(0, val_end - seq_len), n
        else:
            raise ValueError("split must be train/val/test")

        self.x = x[start:end]
        self.y = y[start:end]

        max_start = len(self.x) - seq_len - pred_len + 1
        self.indices = list(range(max(0, max_start)))

        if split == "train" and fewshot_ratio < 1.0:
            keep = max(1, int(len(self.indices) * fewshot_ratio))
            self.indices = self.indices[:keep]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len
        r = e + self.pred_len

        x = self.x[s:e].copy()
        y = self.y[e:r].copy()

        if self.noise_std > 0:
            x += np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }


class M4Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        freq: str,
        seq_len: int,
        pred_len: int,
        split: str,
        scaler: Optional[Tuple[StandardScaler, StandardScaler]] = None,
        fewshot_ratio: float = 1.0,
        noise_std: float = 0.0,
        max_series: int = -1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.noise_std = noise_std

        path = root / "M4" / "Train" / f"{freq}-train.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        if max_series > 0:
            df = df.iloc[:max_series].copy()

        raw = []
        for _, row in df.iterrows():
            vals = row.iloc[1:].values.astype(np.float32)
            vals = vals[~np.isnan(vals)]
            if len(vals) >= seq_len + pred_len + 10:
                raw.append(vals)

        if len(raw) == 0:
            raise RuntimeError(f"No valid M4 series for frequency {freq}")

        if scaler is None:
            fit_vals = []
            for vals in raw:
                fit_vals.append(vals[: int(len(vals) * 0.7)])
            fit_vals = np.concatenate(fit_vals).reshape(-1, 1)
            sx = StandardScaler()
            sx.fit(fit_vals)
            self.scaler = (sx, sx)
        else:
            self.scaler = scaler

        sx, _ = self.scaler
        self.windows = []

        for vals in raw:
            vals = sx.transform(vals.reshape(-1, 1)).reshape(-1).astype(np.float32)
            n = len(vals)
            tr = int(n * 0.7)
            va = int(n * 0.8)

            if split == "train":
                start, end = 0, tr
            elif split == "val":
                start, end = max(0, tr - seq_len), va
            elif split == "test":
                start, end = max(0, va - seq_len), n
            else:
                raise ValueError(split)

            local = vals[start:end]
            max_start = len(local) - seq_len - pred_len + 1
            for s in range(max(0, max_start)):
                self.windows.append((local, s))

        if split == "train" and fewshot_ratio < 1.0:
            keep = max(1, int(len(self.windows) * fewshot_ratio))
            self.windows = self.windows[:keep]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vals, s = self.windows[idx]
        e = s + self.seq_len
        r = e + self.pred_len

        x = vals[s:e].reshape(-1, 1).copy()
        y = vals[e:r].reshape(-1, 1).copy()

        if self.noise_std > 0:
            x += np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }


def build_loaders(cfg: ExpConfig):
    root = Path(cfg.data_root)

    if cfg.task == "main":
        if cfg.dataset not in MAIN_DATASETS:
            raise ValueError(f"Unknown dataset {cfg.dataset}")

        csv_path = root / MAIN_DATASETS[cfg.dataset]
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        train_ds = MainTSFDataset(
            csv_path=csv_path,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            mode=cfg.mode,
            split="train",
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            scaler=None,
            fewshot_ratio=cfg.fewshot_ratio,
            noise_std=cfg.noise_std,
        )

        scaler = train_ds.scaler

        val_ds = MainTSFDataset(
            csv_path=csv_path,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            mode=cfg.mode,
            split="val",
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            scaler=scaler,
            fewshot_ratio=1.0,
            noise_std=0.0,
        )

        test_ds = MainTSFDataset(
            csv_path=csv_path,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            mode=cfg.mode,
            split="test",
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            scaler=scaler,
            fewshot_ratio=1.0,
            noise_std=0.0,
        )

        input_dim = len(train_ds.input_cols)
        output_dim = len(train_ds.target_cols)

    elif cfg.task == "m4":
        if cfg.pred_len <= 0:
            cfg.pred_len = M4_HORIZONS[cfg.m4_freq]

        train_ds = M4Dataset(
            root=root,
            freq=cfg.m4_freq,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            split="train",
            scaler=None,
            fewshot_ratio=cfg.fewshot_ratio,
            noise_std=cfg.noise_std,
            max_series=cfg.m4_max_series,
        )

        scaler = train_ds.scaler

        val_ds = M4Dataset(
            root=root,
            freq=cfg.m4_freq,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            split="val",
            scaler=scaler,
            max_series=cfg.m4_max_series,
        )

        test_ds = M4Dataset(
            root=root,
            freq=cfg.m4_freq,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            split="test",
            scaler=scaler,
            max_series=cfg.m4_max_series,
        )

        input_dim = 1
        output_dim = 1

    else:
        raise ValueError("task must be main or m4")

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Empty dataset split. train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, input_dim, output_dim


# ============================================================
# Baseline models
# ============================================================

class LastValueModel(nn.Module):
    def __init__(self, pred_len: int, output_dim: int):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(self, x):
        y = x[:, -1:, :]
        if y.shape[-1] != self.output_dim:
            y = y[..., -self.output_dim:]
        y = y.repeat(1, self.pred_len, 1)
        return y, torch.tensor(0.0, device=x.device)


class DLinearModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        output_dim: int,
        moving_avg: int = 25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.moving_avg = moving_avg

        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)

        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def moving_mean(self, x):
        xt = x.transpose(1, 2)
        pad = self.moving_avg // 2
        xp = F.pad(xt, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xp, kernel_size=self.moving_avg, stride=1)
        trend = trend[:, :, :x.shape[1]].transpose(1, 2)
        return trend

    def forward(self, x):
        trend = self.moving_mean(x)
        seasonal = x - trend

        ys = self.seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        yt = self.trend(trend.transpose(1, 2)).transpose(1, 2)
        y = ys + yt

        if self.out_proj is not None:
            y = self.out_proj(y)

        return y, torch.tensor(0.0, device=x.device)


class NLinearModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, input_dim: int, output_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(seq_len, pred_len)
        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        last = x[:, -1:, :]
        z = x - last
        y = self.linear(z.transpose(1, 2)).transpose(1, 2)
        y = y + last

        if self.out_proj is not None:
            y = self.out_proj(y)

        return y, torch.tensor(0.0, device=x.device)


class PatchTSTLite(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_len = min(patch_len, seq_len)
        self.stride = stride

        self.n_patches = 1 + max(0, (seq_len - self.patch_len) // stride)

        self.patch_proj = nn.Linear(self.patch_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        b, l, c = x.shape

        patches = []
        for s in range(0, l - self.patch_len + 1, self.stride):
            patches.append(x[:, s:s + self.patch_len, :])

        xp = torch.stack(patches, dim=2)
        xp = xp.permute(0, 3, 2, 1).reshape(b * c, self.n_patches, self.patch_len)

        z = self.patch_proj(xp)
        h = self.encoder(z)

        y = self.head(h.reshape(b * c, -1))
        y = y.reshape(b, c, self.pred_len).permute(0, 2, 1)

        if self.out_proj is not None:
            y = self.out_proj(y)

        return y, torch.tensor(0.0, device=x.device)


class iTransformerLite(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.var_embed = nn.Linear(seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, pred_len)
        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        z = self.var_embed(x.transpose(1, 2))
        h = self.encoder(z)
        y = self.head(h).transpose(1, 2)

        if self.out_proj is not None:
            y = self.out_proj(y)

        return y, torch.tensor(0.0, device=x.device)


# ============================================================
# PS-LLM components
# ============================================================

class MultiScalePatternEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, kernels: List[int], dropout: float):
        super().__init__()

        branch_dim = max(8, d_model // max(1, len(kernels)))

        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Conv1d(branch_dim, branch_dim, kernel_size=1),
                    nn.GELU(),
                )
            )

        self.proj = nn.Sequential(
            nn.Linear(branch_dim * len(kernels) + input_dim * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        xt = x.transpose(1, 2)

        outs = []
        for branch in self.branches:
            y = branch(xt).transpose(1, 2)
            y = y[:, :x.shape[1], :]
            outs.append(y)

        low = F.avg_pool1d(xt, kernel_size=5, stride=1, padding=2).transpose(1, 2)
        low = low[:, :x.shape[1], :]
        high = x - low

        diff = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        z = torch.cat(outs + [low, high, diff], dim=-1)
        return self.proj(z)


class SemanticStatsEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()

        self.stats_per_var = 9

        self.net = nn.Sequential(
            nn.Linear(input_dim * self.stats_per_var, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        mn = x.min(dim=1).values
        mx = x.max(dim=1).values
        last = x[:, -1, :]
        first = x[:, 0, :]
        trend = last - first

        diff = x[:, 1:, :] - x[:, :-1, :]
        volatility = diff.std(dim=1)
        energy = (x ** 2).mean(dim=1)

        if x.shape[1] > 2:
            a = x[:, :-1, :] - x[:, :-1, :].mean(dim=1, keepdim=True)
            b = x[:, 1:, :] - x[:, 1:, :].mean(dim=1, keepdim=True)
            ac = (a * b).mean(dim=1) / (a.std(dim=1) * b.std(dim=1) + 1e-6)
        else:
            ac = torch.zeros_like(mean)

        stats = torch.cat(
            [mean, std, mn, mx, last, trend, volatility, energy, ac],
            dim=-1,
        )

        z = self.net(stats)
        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        return z


class TextSemanticEncoder(nn.Module):
    def __init__(self, hf_root: Path, encoder_name: str, d_model: int):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers is not installed")

        p = resolve_model_dir(hf_root, encoder_name, is_encoder=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            p,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.encoder = AutoModel.from_pretrained(
            p,
            local_files_only=True,
            trust_remote_code=True,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden = getattr(self.encoder.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.encoder.config, "dim", None)
        if hidden is None:
            raise RuntimeError(f"Cannot infer encoder hidden size for {encoder_name}")

        self.proj = nn.Linear(hidden, d_model)

    @staticmethod
    def make_texts(x_cpu: torch.Tensor) -> List[str]:
        arr = x_cpu.numpy()
        texts = []

        for z in arr:
            mean = float(np.mean(z))
            std = float(np.std(z))
            first = float(np.mean(z[0]))
            last = float(np.mean(z[-1]))
            trend = last - first
            vol = float(np.std(np.diff(z, axis=0)))

            if trend > 0.05:
                direction = "increasing"
            elif trend < -0.05:
                direction = "decreasing"
            else:
                direction = "stable"

            if vol > 1.0:
                volatility = "high"
            elif vol > 0.3:
                volatility = "moderate"
            else:
                volatility = "low"

            text = (
                f"The multivariate time series window is {direction}. "
                f"The average normalized level is {mean:.3f}. "
                f"The volatility is {volatility}, with standard deviation {std:.3f}. "
                f"The recent mean change is {trend:.3f}."
            )
            texts.append(text)

        return texts

    def forward(self, x):
        device = x.device
        texts = self.make_texts(x.detach().cpu())

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            out = self.encoder(**tok)
            pooled = out.last_hidden_state.mean(dim=1)

        z = self.proj(pooled.float())
        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        return z


class TrainableTransformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.hidden_size = d_model

    def forward(self, inputs_embeds):
        return self.encoder(inputs_embeds)


class FrozenLLMBackbone(nn.Module):
    def __init__(self, hf_root: Path, llm_name: str, freeze: bool = True):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers is not installed")

        p = resolve_model_dir(hf_root, llm_name, is_encoder=False)

        self.config = AutoConfig.from_pretrained(
            p,
            local_files_only=True,
            trust_remote_code=True,
        )

        if getattr(self.config, "pad_token_id", None) is None:
            eos = getattr(self.config, "eos_token_id", None)
            self.config.pad_token_id = 0 if eos is None else eos

        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModel.from_pretrained(
            p,
            config=self.config,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        hidden = getattr(self.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.config, "word_embed_proj_dim", None)
        if hidden is None:
            hidden = getattr(self.config, "n_embd", None)
        if hidden is None:
            raise RuntimeError(f"Cannot infer LLM hidden size for {llm_name}")

        self.hidden_size = hidden

    def forward(self, inputs_embeds):
        dtype = next(self.model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        return out.last_hidden_state


class LowRankForecastHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pred_len: int,
        output_dim: int,
        rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.output_dim = output_dim
        self.rank = rank

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.time_head = nn.Linear(hidden_dim, pred_len * rank)
        self.var_embed = nn.Parameter(torch.randn(output_dim, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, h):
        target_dtype = self.time_head.weight.dtype
        h = h.to(target_dtype)

        summary = h.mean(dim=1)
        summary = self.drop(self.norm(summary))

        z = self.time_head(summary)
        z = z.reshape(summary.shape[0], self.pred_len, self.rank)

        y = torch.einsum("bhr,cr->bhc", z, self.var_embed) + self.bias
        return y


class PSLLMModel(nn.Module):
    def __init__(self, cfg: ExpConfig, input_dim: int, output_dim: int, hf_root: Path):
        super().__init__()

        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.use_pattern = bool(cfg.use_pattern)
        self.use_semantic = bool(cfg.use_semantic)
        self.use_align = bool(cfg.use_align)
        self.use_gate = bool(cfg.use_gate)
        self.use_llm = bool(cfg.use_llm)

        kernels = [int(k) for k in cfg.cnn_kernels.split(",") if k.strip()]

        self.raw_proj = nn.Linear(input_dim, cfg.d_model)

        self.pattern = MultiScalePatternEncoder(
            input_dim=input_dim,
            d_model=cfg.d_model,
            kernels=kernels,
            dropout=cfg.dropout,
        )

        if cfg.semantic_encoder == "mlp":
            self.semantic = SemanticStatsEncoder(
                input_dim=input_dim,
                d_model=cfg.d_model,
                dropout=cfg.dropout,
            )
        else:
            self.semantic = TextSemanticEncoder(
                hf_root=hf_root,
                encoder_name=cfg.semantic_encoder,
                d_model=cfg.d_model,
            )

        self.pattern_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.semantic_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.Sigmoid(),
        )

        if self.use_llm:
            self.backbone = FrozenLLMBackbone(
                hf_root=hf_root,
                llm_name=cfg.llm_name,
                freeze=bool(cfg.freeze_llm),
            )
            self.to_backbone = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, self.backbone.hidden_size),
            )
            hidden = self.backbone.hidden_size
        else:
            self.backbone = TrainableTransformerBackbone(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
            )
            self.to_backbone = nn.Identity()
            hidden = cfg.d_model

        self.head = LowRankForecastHead(
            hidden_dim=hidden,
            pred_len=cfg.pred_len,
            output_dim=output_dim,
            rank=cfg.rank,
            dropout=cfg.dropout,
        )

    def forward(self, x):
        raw = self.raw_proj(x)

        if self.use_pattern:
            zp = self.pattern_proj(self.pattern(x))
        else:
            zp = raw

        if self.use_semantic:
            zs = self.semantic_proj(self.semantic(x))
        else:
            zs = raw

        if self.use_pattern and self.use_semantic:
            if self.use_gate:
                g = self.gate(torch.cat([zp, zs], dim=-1))
                z = g * zp + (1.0 - g) * zs
            else:
                z = 0.5 * zp + 0.5 * zs
        elif self.use_pattern:
            z = zp
        elif self.use_semantic:
            z = zs
        else:
            z = raw

        emb = self.to_backbone(z)
        h = self.backbone(emb)
        y = self.head(h)

        align_loss = torch.tensor(0.0, device=x.device)

        if self.use_align and self.use_pattern and self.use_semantic:
            a = F.normalize(zp.mean(dim=1), dim=-1)
            b = F.normalize(zs.mean(dim=1), dim=-1)
            align_loss = 1.0 - (a * b).sum(dim=-1).mean()

        return y, align_loss


def build_model(cfg: ExpConfig, input_dim: int, output_dim: int, hf_root: Path):
    if cfg.model == "last":
        return LastValueModel(cfg.pred_len, output_dim)

    if cfg.model == "dlinear":
        return DLinearModel(cfg.seq_len, cfg.pred_len, input_dim, output_dim)

    if cfg.model == "nlinear":
        return NLinearModel(cfg.seq_len, cfg.pred_len, input_dim, output_dim)

    if cfg.model == "patchtst":
        return PatchTSTLite(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )

    if cfg.model == "itransformer":
        return iTransformerLite(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )

    if cfg.model == "llm_direct":
        d = asdict(cfg)
        d.update({
            "use_pattern": 0,
            "use_semantic": 0,
            "use_align": 0,
            "use_gate": 0,
            "use_llm": 1,
            "model": "psllm",
        })
        return PSLLMModel(ExpConfig(**d), input_dim, output_dim, hf_root)

    if cfg.model == "psllm":
        return PSLLMModel(cfg, input_dim, output_dim, hf_root)

    raise ValueError(f"Unknown model {cfg.model}")


# ============================================================
# Training / evaluation
# ============================================================

def metric_dict(pred: torch.Tensor, y: torch.Tensor):
    mse = torch.mean((pred - y) ** 2).item()
    mae = torch.mean(torch.abs(pred - y)).item()

    denom = torch.mean(torch.abs(y)).item() + 1e-8
    nmae = mae / denom

    return {
        "mse": mse,
        "mae": mae,
        "nmae": nmae,
    }


def train_epoch(model, loader, optimizer, scaler, device, cfg: ExpConfig):
    model.train()

    sums = {
        "loss": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "nmae": 0.0,
    }
    n = 0

    use_amp = bool(cfg.amp) and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        if cfg.max_train_batches > 0 and batch_idx >= cfg.max_train_batches:
            break

        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, align = model(x)
            pred = pred.float()
            loss_pred = F.mse_loss(pred, y)
            loss = loss_pred + (cfg.align_lambda * align if cfg.use_align else 0.0)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        m = metric_dict(pred.detach(), y.detach())

        sums["loss"] += loss.item()
        sums["mse"] += m["mse"]
        sums["mae"] += m["mae"]
        sums["nmae"] += m["nmae"]
        n += 1

    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, cfg: ExpConfig):
    model.eval()

    sums = {
        "loss": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "nmae": 0.0,
    }
    n = 0

    use_amp = bool(cfg.amp) and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        if cfg.max_eval_batches > 0 and batch_idx >= cfg.max_eval_batches:
            break

        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, align = model(x)
            pred = pred.float()
            loss_pred = F.mse_loss(pred, y)
            loss = loss_pred + (cfg.align_lambda * align if cfg.use_align else 0.0)

        m = metric_dict(pred, y)

        sums["loss"] += loss.item()
        sums["mse"] += m["mse"]
        sums["mae"] += m["mae"]
        sums["nmae"] += m["nmae"]
        n += 1

    return {k: v / max(1, n) for k, v in sums.items()}


def run_one(cfg: ExpConfig):
    set_seed(cfg.seed)

    hf_root = resolve_existing_root(cfg.hf_root)
    device = get_device(cfg.device)

    run_name = (
        f"{cfg.task}_{cfg.dataset if cfg.task == 'main' else 'M4-' + cfg.m4_freq}"
        f"_model-{cfg.model}"
        f"_llm-{cfg.llm_name}"
        f"_sem-{cfg.semantic_encoder}"
        f"_mode-{cfg.mode}"
        f"_L{cfg.seq_len}_H{cfg.pred_len}"
        f"_pat{cfg.use_pattern}_sem{cfg.use_semantic}_align{cfg.use_align}_gate{cfg.use_gate}_llm{cfg.use_llm}"
        f"_fs{cfg.fewshot_ratio}_noise{cfg.noise_std}_seed{cfg.seed}"
    )

    run_dir = Path(cfg.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader, val_loader, test_loader, input_dim, output_dim = build_loaders(cfg)

    model = build_model(cfg, input_dim, output_dim, hf_root)
    model = model.to(device)

    total_params, trainable_params = count_params(model)

    print("=" * 120)
    print("RUN:", run_name)
    print("device:", device)
    print("data_root:", cfg.data_root)
    print("hf_root:", hf_root)
    print("dataset:", cfg.dataset if cfg.task == "main" else f"M4-{cfg.m4_freq}")
    print("model:", cfg.model)
    print("llm:", cfg.llm_name)
    print("semantic_encoder:", cfg.semantic_encoder)
    print("input_dim:", input_dim, "output_dim:", output_dim)
    print("params_total:", total_params, "params_trainable:", trainable_params)
    print("batches train/val/test:", len(train_loader), len(val_loader), len(test_loader))
    print("run_dir:", run_dir)
    print("=" * 120)

    trainable = [p for p in model.parameters() if p.requires_grad]

    history = []
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    t0 = time.time()

    if len(trainable) == 0:
        val_metrics = evaluate(model, val_loader, device, cfg)
        test_metrics = evaluate(model, test_loader, device, cfg)
        history.append({
            "epoch": 0,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_nmae": val_metrics["nmae"],
        })
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
            val_metrics = evaluate(model, val_loader, device, cfg)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_nmae": train_metrics["nmae"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_nmae": val_metrics["nmae"],
            }
            history.append(row)

            print(
                f"epoch {epoch:03d} | "
                f"train mse {train_metrics['mse']:.6f} mae {train_metrics['mae']:.6f} | "
                f"val mse {val_metrics['mse']:.6f} mae {val_metrics['mae']:.6f}"
            )

            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_epoch = epoch
                bad = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            else:
                bad += 1

            if bad >= cfg.patience:
                print(f"early stopping at epoch {epoch}, best_epoch={best_epoch}")
                break

        if (run_dir / "best_model.pt").exists():
            model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

        test_metrics = evaluate(model, test_loader, device, cfg)

    elapsed = time.time() - t0

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    result = {
        "task": cfg.task,
        "dataset": cfg.dataset if cfg.task == "main" else f"M4-{cfg.m4_freq}",
        "model": cfg.model,
        "llm_name": cfg.llm_name,
        "semantic_encoder": cfg.semantic_encoder,
        "mode": cfg.mode,
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "use_pattern": cfg.use_pattern,
        "use_semantic": cfg.use_semantic,
        "use_align": cfg.use_align,
        "use_gate": cfg.use_gate,
        "use_llm": cfg.use_llm,
        "fewshot_ratio": cfg.fewshot_ratio,
        "noise_std": cfg.noise_std,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_nmae": test_metrics["nmae"],
        "params_total": total_params,
        "params_trainable": trainable_params,
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame([result]).to_csv(run_dir / "result.csv", index=False)

    print("TEST RESULT")
    print(json.dumps(result, indent=2))

    del model
    cleanup()

    return result


# ============================================================
# Plans
# ============================================================

def run_main(args):
    results = []
    datasets = parse_list(args.datasets)
    models = parse_list(args.models)

    for ds in datasets:
        horizons = DEFAULT_HORIZONS[ds] if args.horizons == "auto" else parse_int_list(args.horizons)

        for h in horizons:
            for model_name in models:
                cfg = cfg_from_args(
                    args,
                    task="main",
                    dataset=ds,
                    pred_len=h,
                    model=model_name,
                )

                if model_name not in ["psllm", "llm_direct"]:
                    cfg.use_pattern = 0
                    cfg.use_semantic = 0
                    cfg.use_align = 0
                    cfg.use_gate = 0
                    cfg.use_llm = 0

                res = run_one(cfg)
                results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_main.csv", index=False)
    print("Saved:", out / "summary_main.csv")


def run_ablations(args):
    results = []
    datasets = parse_list(args.datasets)
    horizons = [96, 336] if args.horizons == "auto" else parse_int_list(args.horizons)

    variants = [
        ("full", 1, 1, 1, 1, 1, "psllm"),
        ("wo_pattern", 0, 1, 0, 0, 1, "psllm"),
        ("wo_semantic", 1, 0, 0, 0, 1, "psllm"),
        ("wo_align", 1, 1, 0, 1, 1, "psllm"),
        ("wo_gate", 1, 1, 1, 0, 1, "psllm"),
        ("wo_llm", 1, 1, 1, 1, 0, "psllm"),
        ("llm_direct", 0, 0, 0, 0, 1, "llm_direct"),
    ]

    for ds in datasets:
        for h in horizons:
            if ds == "ILI" and h > 60:
                continue

            for variant, pat, sem, align, gate, llm, model_name in variants:
                cfg = cfg_from_args(
                    args,
                    task="main",
                    dataset=ds,
                    pred_len=h,
                    model=model_name,
                    use_pattern=pat,
                    use_semantic=sem,
                    use_align=align,
                    use_gate=gate,
                    use_llm=llm,
                    out_dir=str(Path(args.out_dir) / "ablations" / variant),
                )

                res = run_one(cfg)
                res["variant"] = variant
                results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_ablations.csv", index=False)
    print("Saved:", out / "summary_ablations.csv")


def run_backbones(args):
    results = []
    datasets = parse_list(args.datasets)
    backbones = parse_list(args.backbones)

    for ds in datasets:
        for llm_name in backbones:
            cfg = cfg_from_args(
                args,
                task="main",
                dataset=ds,
                pred_len=args.pred_len,
                model="psllm",
                llm_name=llm_name,
                use_pattern=1,
                use_semantic=1,
                use_align=1,
                use_gate=1,
                use_llm=1,
                out_dir=str(Path(args.out_dir) / "backbones"),
            )

            res = run_one(cfg)
            results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_backbones.csv", index=False)
    print("Saved:", out / "summary_backbones.csv")


def run_semantic(args):
    results = []
    datasets = parse_list(args.datasets)
    encoders = parse_list(args.semantic_encoders)

    for ds in datasets:
        for enc in encoders:
            cfg = cfg_from_args(
                args,
                task="main",
                dataset=ds,
                pred_len=args.pred_len,
                model="psllm",
                semantic_encoder=enc,
                use_pattern=1,
                use_semantic=1,
                use_align=1,
                use_gate=1,
                use_llm=1,
                out_dir=str(Path(args.out_dir) / "semantic"),
            )

            res = run_one(cfg)
            results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_semantic_encoders.csv", index=False)
    print("Saved:", out / "summary_semantic_encoders.csv")


def run_noise(args):
    results = []
    datasets = parse_list(args.datasets)
    models = parse_list(args.models)
    noises = parse_float_list(args.noise_levels)

    for ds in datasets:
        for noise in noises:
            for model_name in models:
                cfg = cfg_from_args(
                    args,
                    task="main",
                    dataset=ds,
                    pred_len=args.pred_len,
                    model=model_name,
                    noise_std=noise,
                    out_dir=str(Path(args.out_dir) / "noise"),
                )

                if model_name not in ["psllm", "llm_direct"]:
                    cfg.use_pattern = 0
                    cfg.use_semantic = 0
                    cfg.use_align = 0
                    cfg.use_gate = 0
                    cfg.use_llm = 0

                res = run_one(cfg)
                results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_noise.csv", index=False)
    print("Saved:", out / "summary_noise.csv")


def run_fewshot(args):
    results = []
    datasets = parse_list(args.datasets)
    models = parse_list(args.models)
    fractions = parse_float_list(args.fewshot_levels)

    for ds in datasets:
        for frac in fractions:
            for model_name in models:
                cfg = cfg_from_args(
                    args,
                    task="main",
                    dataset=ds,
                    pred_len=args.pred_len,
                    model=model_name,
                    fewshot_ratio=frac,
                    out_dir=str(Path(args.out_dir) / "fewshot"),
                )

                if model_name not in ["psllm", "llm_direct"]:
                    cfg.use_pattern = 0
                    cfg.use_semantic = 0
                    cfg.use_align = 0
                    cfg.use_gate = 0
                    cfg.use_llm = 0

                res = run_one(cfg)
                results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_fewshot.csv", index=False)
    print("Saved:", out / "summary_fewshot.csv")


def run_m4(args):
    results = []
    freqs = parse_list(args.m4_freqs)
    models = parse_list(args.models)

    for freq in freqs:
        h = M4_HORIZONS[freq] if args.pred_len <= 0 else args.pred_len

        for model_name in models:
            cfg = cfg_from_args(
                args,
                task="m4",
                dataset="M4",
                mode="S",
                m4_freq=freq,
                pred_len=h,
                model=model_name,
                out_dir=str(Path(args.out_dir) / "m4"),
            )

            if model_name not in ["psllm", "llm_direct"]:
                cfg.use_pattern = 0
                cfg.use_semantic = 0
                cfg.use_align = 0
                cfg.use_gate = 0
                cfg.use_llm = 0

            res = run_one(cfg)
            results.append(res)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "summary_m4.csv", index=False)
    print("Saved:", out / "summary_m4.csv")


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--plan",
        type=str,
        default="single",
        choices=["single", "main", "ablations", "backbones", "semantic", "noise", "fewshot", "m4"],
    )

    p.add_argument("--data_root", type=str, default="data_icdm_tsf")
    p.add_argument("--hf_root", type=str, default="/home/tahiti/Malashin_Projects/hf_models")
    p.add_argument("--out_dir", type=str, default="runs_icdm_psllm")

    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument(
        "--datasets",
        type=str,
        default="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Traffic,Electricity,Exchange,ILI",
    )

    p.add_argument("--task", type=str, default="main", choices=["main", "m4"])
    p.add_argument("--mode", type=str, default="M", choices=["M", "MS", "S"])

    p.add_argument(
        "--model",
        type=str,
        default="psllm",
        choices=["last", "dlinear", "nlinear", "patchtst", "itransformer", "llm_direct", "psllm"],
    )

    p.add_argument(
        "--models",
        type=str,
        default="last,dlinear,nlinear,patchtst,itransformer,llm_direct,psllm",
    )

    p.add_argument("--llm_name", type=str, default="qwen2p5_1p5b")

    p.add_argument(
        "--backbones",
        type=str,
        default=(
            "opt_350m,opt_1p3b,pythia_410m,pythia_1b,"
            "qwen2p5_0p5b_instruct,qwen2p5_1p5b,qwen2p5_1p5b_instruct,"
            "smollm2_1p7b,tinyllama_1p1b_chat"
        ),
    )

    p.add_argument("--semantic_encoder", type=str, default="mlp")

    p.add_argument(
        "--semantic_encoders",
        type=str,
        default="mlp,bert_tiny,bert_mini,distilbert_base_uncased,electra_small_discriminator",
    )

    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--horizons", type=str, default="auto")

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rank", type=int, default=64)

    p.add_argument("--cnn_kernels", type=str, default="3,5,7,13")
    p.add_argument("--align_lambda", type=float, default=0.05)

    p.add_argument("--use_pattern", type=int, default=1)
    p.add_argument("--use_semantic", type=int, default=1)
    p.add_argument("--use_align", type=int, default=1)
    p.add_argument("--use_gate", type=int, default=1)
    p.add_argument("--use_llm", type=int, default=1)
    p.add_argument("--freeze_llm", type=int, default=1)

    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--noise_levels", type=str, default="0.0,0.05,0.1,0.2")

    p.add_argument("--fewshot_ratio", type=float, default=1.0)
    p.add_argument("--fewshot_levels", type=str, default="0.05,0.1,0.25,0.5,1.0")

    p.add_argument("--m4_freq", type=str, default="Hourly")
    p.add_argument("--m4_freqs", type=str, default="Hourly,Daily,Weekly,Monthly")
    p.add_argument("--m4_max_series", type=int, default=-1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", type=int, default=1)

    p.add_argument("--max_train_batches", type=int, default=-1)
    p.add_argument("--max_eval_batches", type=int, default=-1)

    return p


def main():
    args = build_parser().parse_args()

    if args.plan == "single":
        cfg = cfg_from_args(args)
        run_one(cfg)
    elif args.plan == "main":
        run_main(args)
    elif args.plan == "ablations":
        run_ablations(args)
    elif args.plan == "backbones":
        run_backbones(args)
    elif args.plan == "semantic":
        run_semantic(args)
    elif args.plan == "noise":
        run_noise(args)
    elif args.plan == "fewshot":
        run_fewshot(args)
    elif args.plan == "m4":
        run_m4(args)
    else:
        raise ValueError(args.plan)


if __name__ == "__main__":
    main()