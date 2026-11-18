from __future__ import annotations
import copy
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit('PyTorch is required. Install it with `pip install torch` or follow https://pytorch.org/get-started/locally/ for your CUDA version.') from exc
MODEL_NAMES = ('clinical_only', 'hrv_only', 'simple_fusion', 'gated_cross_attention')

@dataclass
class ExperimentConfig:
    subject_col: str = 'subject_id'
    admission_col: str = 'hadm_id'
    label_col: str = 'label'
    hour_col: str = 'hour'
    hrv_valid_col: str = 'hrv_valid'
    timesteps: int = 24
    max_delta_hours: float = 24.0
    n_splits: int = 5
    inner_splits: int = 4
    batch_size: int = 64
    epochs: int = 60
    patience: int = 10
    latent_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.5
    learning_rate: float = 0.0003
    weight_decay: float = 0.005
    gradient_clip: float = 1.0
    threshold_method: str = 'f1'
    min_delta: float = 0.0001
    seed: int = 2026
    device: str = 'auto'
    num_workers: int = 0
    bootstrap_iterations: int = 500
    max_positive_weight: float = 3.0
    selection_brier_weight: float = 0.25

@dataclass
class ArrayBundle:
    subject_ids: np.ndarray
    admission_ids: np.ndarray
    y: np.ndarray
    xc: np.ndarray
    mc: np.ndarray
    dc: np.ndarray
    xh: np.ndarray
    mh: np.ndarray
    dh: np.ndarray
    qh: np.ndarray
    token_valid: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_device(requested: str) -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if suffix in {'.feather', '.ft'}:
        return pd.read_feather(path)
    raise ValueError(f'Unsupported table format: {path}')

def parse_columns(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]

def validate_input_tables(clinical: pd.DataFrame, hrv: pd.DataFrame, clinical_features: Sequence[str], hrv_features: Sequence[str], quality_features: Sequence[str], config: ExperimentConfig) -> None:
    if hrv.empty or 'audit_pass' not in hrv or (not hrv['audit_pass'].eq(1).all()):
        raise ValueError('Only audit-approved HRV rows may enter modeling')
    if not hrv[config.hrv_valid_col].eq(1).all():
        raise ValueError('Rejected HRV must be absent, not available for imputation')
    if hrv.duplicated([config.subject_col, config.admission_col, config.hour_col]).any():
        raise ValueError('Duplicate audited admission-hour rows')
    ckeys = set(map(tuple, clinical[[config.subject_col, config.admission_col]].to_numpy()))
    hkeys = set(map(tuple, hrv[[config.subject_col, config.admission_col]].to_numpy()))
    if ckeys != hkeys:
        raise ValueError('All models require exactly the same audit-approved admission cohort')
    required_clinical = {config.subject_col, config.admission_col, config.label_col, *clinical_features}
    required_hrv = {config.subject_col, config.admission_col, config.hour_col, *hrv_features, *quality_features}
    missing_c = sorted(required_clinical - set(clinical.columns))
    missing_h = sorted(required_hrv - set(hrv.columns))
    if missing_c or missing_h:
        raise ValueError(f"Missing columns. clinical={missing_c or 'none'}; hrv={missing_h or 'none'}")
    if clinical[config.admission_col].duplicated().any():
        duplicates = clinical.loc[clinical[config.admission_col].duplicated(False), config.admission_col].head(10)
        raise ValueError(f'Clinical table must contain one row per admission. Duplicate {config.admission_col}: {duplicates.tolist()}')
    labels = set(pd.Series(clinical[config.label_col]).dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f'{config.label_col} must be binary 0/1; found {labels}')
    if clinical[config.label_col].isna().any():
        raise ValueError(f'{config.label_col} contains missing values')
    hours = pd.to_numeric(hrv[config.hour_col], errors='coerce')
    if hours.isna().any():
        raise ValueError(f'{config.hour_col} contains non-numeric values')
    if ((hours < 0) | (hours >= config.timesteps)).any():
        bad = sorted(hours[(hours < 0) | (hours >= config.timesteps)].unique())[:10]
        raise ValueError(f'{config.hour_col} must be in [0, {config.timesteps - 1}]; found {bad}')
    id_audit = clinical[[config.admission_col, config.subject_col]].merge(hrv[[config.admission_col, config.subject_col]].drop_duplicates(), on=config.admission_col, how='inner', suffixes=('_clinical', '_hrv'))
    mismatch = id_audit[f'{config.subject_col}_clinical'].astype(str) != id_audit[f'{config.subject_col}_hrv'].astype(str)
    if mismatch.any():
        bad = id_audit.loc[mismatch, config.admission_col].head(10).tolist()
        raise ValueError(f'subject_id mismatch between clinical and HRV tables for admissions {bad}')

class FoldPreprocessor:
    """Fold-local imputation, scaling, padding, masks, and delta creation."""

    def __init__(self, clinical_features: Sequence[str], hrv_features: Sequence[str], quality_features: Sequence[str], config: ExperimentConfig) -> None:
        self.clinical_features = list(clinical_features)
        self.hrv_features = list(hrv_features)
        self.quality_features = list(quality_features)
        self.config = config
        self.c_median: Optional[np.ndarray] = None
        self.c_mean: Optional[np.ndarray] = None
        self.c_std: Optional[np.ndarray] = None
        self.h_median: Optional[np.ndarray] = None
        self.h_mean: Optional[np.ndarray] = None
        self.h_std: Optional[np.ndarray] = None

    @staticmethod
    def _fit_stats(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[np.ndarray, ...]:
        array = frame.loc[:, columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if array.shape[0] == 0:
            zeros = np.zeros(len(columns), dtype=np.float32)
            ones = np.ones(len(columns), dtype=np.float32)
            return (zeros, zeros.copy(), ones)
        med = np.nanmedian(array, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        q25 = np.nanpercentile(array, 25, axis=0)
        q75 = np.nanpercentile(array, 75, axis=0)
        robust_scale = (q75 - q25) / 1.349
        filled = np.where(np.isfinite(array), array, med)
        fallback_std = filled.std(axis=0)
        scale = np.where(np.isfinite(robust_scale) & (robust_scale > 1e-08), robust_scale, fallback_std)
        scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, 1.0)
        return (med.astype(np.float32), med.astype(np.float32), scale.astype(np.float32))

    def fit(self, clinical_train: pd.DataFrame, hrv_train: pd.DataFrame) -> 'FoldPreprocessor':
        self.c_median, self.c_mean, self.c_std = self._fit_stats(clinical_train, self.clinical_features)
        hrv_for_stats = hrv_train
        if self.config.hrv_valid_col in hrv_for_stats:
            valid_flag = pd.to_numeric(hrv_for_stats[self.config.hrv_valid_col], errors='coerce').fillna(0)
            valid_rows = hrv_for_stats.loc[valid_flag >= 0.5]
            if not len(valid_rows):
                raise ValueError('Training fold has no audited HRV rows')
            hrv_for_stats = valid_rows
        self.h_median, self.h_mean, self.h_std = self._fit_stats(hrv_for_stats, self.hrv_features)
        return self

    def _check_fitted(self) -> None:
        if any((item is None for item in (self.c_median, self.c_mean, self.c_std, self.h_median, self.h_mean, self.h_std))):
            raise RuntimeError('FoldPreprocessor.fit must be called first')

    def transform(self, clinical_part: pd.DataFrame, hrv_all: pd.DataFrame) -> ArrayBundle:
        self._check_fitted()
        cfg = self.config
        clinical_part = clinical_part.reset_index(drop=True).copy()
        n = len(clinical_part)
        fc, fh, fq, tmax = (len(self.clinical_features), len(self.hrv_features), len(self.quality_features), cfg.timesteps)
        raw_c = clinical_part[self.clinical_features].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        mc = np.isfinite(raw_c).astype(np.float32)
        xc = np.where(np.isfinite(raw_c), raw_c, self.c_median)
        xc = ((xc - self.c_mean) / self.c_std).astype(np.float32)
        xc[mc == 0] = 0.0
        dc = np.empty((n, fc), dtype=np.float32)
        for j, feature in enumerate(self.clinical_features):
            delta_col = f'{feature}__delta'
            if delta_col in clinical_part:
                values = pd.to_numeric(clinical_part[delta_col], errors='coerce').to_numpy(float)
                values = np.where(np.isfinite(values), values, cfg.max_delta_hours)
            else:
                values = np.where(mc[:, j] == 1, 0.0, cfg.max_delta_hours)
            values = np.clip(values, 0.0, cfg.max_delta_hours)
            dc[:, j] = np.log1p(values) / np.log1p(cfg.max_delta_hours)
        xh = np.zeros((n, tmax, fh), dtype=np.float32)
        mh = np.zeros((n, tmax, fh), dtype=np.float32)
        dh = np.ones((n, tmax, fh), dtype=np.float32)
        qh = np.zeros((n, tmax, fq), dtype=np.float32)
        token_valid = np.zeros((n, tmax), dtype=bool)
        admissions = set(clinical_part[cfg.admission_col].tolist())
        hrv_part = hrv_all[hrv_all[cfg.admission_col].isin(admissions)].copy()
        aggregate_cols = [*self.hrv_features, *self.quality_features]
        if cfg.hrv_valid_col in hrv_part:
            aggregate_cols.append(cfg.hrv_valid_col)
        if len(hrv_part):
            hrv_part[aggregate_cols] = hrv_part[aggregate_cols].apply(pd.to_numeric, errors='coerce')
            hrv_part[cfg.hour_col] = pd.to_numeric(hrv_part[cfg.hour_col]).astype(int)
            hrv_part = hrv_part.groupby([cfg.admission_col, cfg.hour_col], as_index=False)[aggregate_cols].mean()
        grouped = {key: value for key, value in hrv_part.groupby(cfg.admission_col)}
        for i, admission_id in enumerate(clinical_part[cfg.admission_col]):
            patient_hrv = grouped.get(admission_id)
            if patient_hrv is None:
                continue
            for _, row in patient_hrv.iterrows():
                hour = int(row[cfg.hour_col])
                raw_h = pd.to_numeric(row[self.hrv_features], errors='coerce').to_numpy(float)
                feature_mask = np.isfinite(raw_h)
                explicitly_valid = True
                if cfg.hrv_valid_col in row.index and pd.notna(row[cfg.hrv_valid_col]):
                    explicitly_valid = float(row[cfg.hrv_valid_col]) >= 0.5
                valid = bool(explicitly_valid and feature_mask.any())
                token_valid[i, hour] = valid
                if valid:
                    mh[i, hour] = feature_mask.astype(np.float32)
                    filled = np.where(feature_mask, raw_h, self.h_median)
                    scaled = ((filled - self.h_mean) / self.h_std).astype(np.float32)
                    scaled[~feature_mask] = 0.0
                    xh[i, hour] = scaled
                    raw_q = pd.to_numeric(row[self.quality_features], errors='coerce').to_numpy(float)
                    quality_scores = np.zeros(fq, dtype=np.float32)
                    for q_index, (quality_name, quality_value) in enumerate(zip(self.quality_features, raw_q)):
                        adverse = any((marker in quality_name.lower() for marker in ('artifact', 'outlier', 'noise', 'invalid')))
                        if not np.isfinite(quality_value):
                            quality_value = 1.0 if adverse else 0.0
                        quality_value = float(np.clip(quality_value, 0.0, 1.0))
                        quality_scores[q_index] = 1.0 - quality_value if adverse else quality_value
                    qh[i, hour] = quality_scores
            for j in range(fh):
                last_valid: Optional[int] = None
                for hour in range(tmax):
                    if mh[i, hour, j] == 1:
                        last_valid = hour
                        delta = 0.0
                    elif last_valid is None:
                        delta = cfg.max_delta_hours
                    else:
                        delta = min(float(hour - last_valid), cfg.max_delta_hours)
                    dh[i, hour, j] = np.log1p(delta) / np.log1p(cfg.max_delta_hours)
        return ArrayBundle(subject_ids=clinical_part[cfg.subject_col].astype(str).to_numpy(), admission_ids=clinical_part[cfg.admission_col].astype(str).to_numpy(), y=clinical_part[cfg.label_col].astype(np.float32).to_numpy(), xc=xc, mc=mc, dc=dc, xh=xh, mh=mh, dh=dh, qh=qh, token_valid=token_valid)

class MultimodalDataset(Dataset):

    def __init__(self, arrays: ArrayBundle) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        a = self.arrays
        return {'subject_id': str(a.subject_ids[index]), 'admission_id': str(a.admission_ids[index]), 'y': torch.tensor(a.y[index], dtype=torch.float32), 'xc': torch.from_numpy(a.xc[index]), 'mc': torch.from_numpy(a.mc[index]), 'dc': torch.from_numpy(a.dc[index]), 'xh': torch.from_numpy(a.xh[index]), 'mh': torch.from_numpy(a.mh[index]), 'dh': torch.from_numpy(a.dh[index]), 'qh': torch.from_numpy(a.qh[index]), 'token_valid': torch.from_numpy(a.token_valid[index])}

def masked_mean(values: Tensor, valid: Tensor, dim: int=1) -> Tensor:
    weights = valid.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator

def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> Tensor:
    position = torch.arange(length, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
    return pe

class ClinicalEncoder(nn.Module):
    """One token per clinical feature; missingness remains an explicit input."""

    def __init__(self, n_features: int, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.value_projection = nn.Sequential(nn.Linear(3, dim), nn.GELU())
        self.feature_embedding = nn.Parameter(torch.randn(1, n_features, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, xc: Tensor, mc: Tensor, dc: Tensor) -> Tensor:
        tokens = torch.stack([xc, mc, dc], dim=-1)
        tokens = self.value_projection(tokens) + self.feature_embedding
        return self.norm(self.encoder(tokens))

class HRVEncoder(nn.Module):
    """Hourly temporal encoder using values, masks, deltas, and quality."""

    def __init__(self, n_hrv: int, n_quality: int, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(n_hrv * 3 + n_quality, dim), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, xh: Tensor, mh: Tensor, dh: Tensor, qh: Tensor, token_valid: Tensor) -> Tensor:
        values = torch.cat([xh, mh, dh, qh], dim=-1)
        tokens = self.input_projection(values)
        tokens = tokens + sinusoidal_position_encoding(tokens.shape[1], tokens.shape[2], tokens.device).unsqueeze(0)
        padding = ~token_valid.bool()
        all_missing = padding.all(dim=1)
        safe_padding = padding.clone()
        safe_padding[all_missing, 0] = False
        encoded = self.norm(self.encoder(tokens, src_key_padding_mask=safe_padding))
        return encoded * token_valid.unsqueeze(-1).to(encoded.dtype)

class QualityBiasedCrossAttention(nn.Module):
    """Multi-head attention with explicit valid-key mask and log-quality bias."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError('latent_dim must be divisible by num_heads')
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_projection = nn.Linear(dim, dim)
        self.k_projection = nn.Linear(dim, dim)
        self.v_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.raw_quality_scale = nn.Parameter(torch.tensor(0.5))

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, key_value: Tensor, key_valid: Tensor, key_quality: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        q = self._split_heads(self.q_projection(query))
        k = self._split_heads(self.k_projection(key_value))
        v = self._split_heads(self.v_projection(key_value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_quality is not None:
            quality = key_quality.clamp(0.0001, 1.0)
            scale = torch.nn.functional.softplus(self.raw_quality_scale)
            scores = scores + scale * torch.log(quality)[:, None, None, :]
        valid = key_valid[:, None, None, :].bool()
        scores = scores.masked_fill(~valid, -10000.0)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.dim)
        return (self.output_projection(context), weights.mean(dim=1))

class PredictionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).squeeze(-1)

class BaseFusionModel(nn.Module):

    def __init__(self, n_clinical: int, n_hrv: int, n_quality: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.clinical_encoder = ClinicalEncoder(n_clinical, d, config.num_heads, config.num_layers, config.dropout)
        self.hrv_encoder = HRVEncoder(n_hrv, n_quality, d, config.num_heads, config.num_layers, config.dropout)

    def encode_clinical(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        tokens = self.clinical_encoder(batch['xc'], batch['mc'], batch['dc'])
        return (tokens, tokens.mean(dim=1))

    def encode_hrv(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        tokens = self.hrv_encoder(batch['xh'], batch['mh'], batch['dh'], batch['qh'], batch['token_valid'])
        pooled = masked_mean(tokens, batch['token_valid'], dim=1)
        return (tokens, pooled)

class ClinicalOnlyModel(BaseFusionModel):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.head = PredictionHead(self.config.latent_dim, self.config.latent_dim, self.config.dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, zc = self.encode_clinical(batch)
        return (self.head(zc), {})

class HRVOnlyModel(BaseFusionModel):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.head = PredictionHead(self.config.latent_dim, self.config.latent_dim, self.config.dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, zh = self.encode_hrv(batch)
        valid_prop = batch['token_valid'].float().mean(dim=1)
        return (self.head(zh), {'hrv_valid_prop': valid_prop})

class SimpleFusionModel(BaseFusionModel):
    """Two encoders followed by plain concatenation; no cross-attention or gate."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.head = PredictionHead(self.config.latent_dim * 2, self.config.latent_dim, self.config.dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, zc = self.encode_clinical(batch)
        _, zh = self.encode_hrv(batch)
        valid_prop = batch['token_valid'].float().mean(dim=1)
        return (self.head(torch.cat([zc, zh], dim=-1)), {'hrv_valid_prop': valid_prop})

class GatedCrossAttentionModelV1(BaseFusionModel):
    """Bidirectional cross-attention plus an admission-level reliability gate."""

    def __init__(self, n_clinical: int, n_hrv: int, n_quality: int, config: ExperimentConfig) -> None:
        super().__init__(n_clinical, n_hrv, n_quality, config)
        d = config.latent_dim
        self.clinical_from_hrv = QualityBiasedCrossAttention(d, config.num_heads, config.dropout)
        self.hrv_from_clinical = QualityBiasedCrossAttention(d, config.num_heads, config.dropout)
        self.clinical_norm = nn.LayerNorm(d)
        self.hrv_norm = nn.LayerNorm(d)
        quality_context_dim = n_quality + 3
        self.quality_encoder = nn.Sequential(nn.Linear(quality_context_dim, d), nn.GELU(), nn.Dropout(config.dropout))
        self.gate = nn.Linear(d * 3, d)
        self.head = PredictionHead(d, d, config.dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        hc, _ = self.encode_clinical(batch)
        hh, _ = self.encode_hrv(batch)
        token_valid = batch['token_valid'].bool()
        clinical_valid = torch.ones(hc.shape[:2], dtype=torch.bool, device=hc.device)
        if batch['qh'].shape[-1] > 0:
            attention_quality = batch['qh'][:, :, 0]
        else:
            attention_quality = token_valid.float()
        c_context, attention_c_from_h = self.clinical_from_hrv(query=hc, key_value=hh, key_valid=token_valid, key_quality=attention_quality)
        h_context, attention_h_from_c = self.hrv_from_clinical(query=hh, key_value=hc, key_valid=clinical_valid)
        hc_star = self.clinical_norm(hc + c_context)
        hh_star = self.hrv_norm(hh + h_context)
        hh_star = hh_star * token_valid.unsqueeze(-1).to(hh_star.dtype)
        zc = hc_star.mean(dim=1)
        zh = masked_mean(hh_star, token_valid, dim=1)
        quality_mean = masked_mean(batch['qh'], token_valid, dim=1)
        valid_prop = token_valid.float().mean(dim=1, keepdim=True)
        clinical_available = batch['mc'].float().mean(dim=1, keepdim=True)
        clinical_delta = batch['dc'].float().mean(dim=1, keepdim=True)
        quality_context = torch.cat([quality_mean, valid_prop, clinical_available, clinical_delta], dim=-1)
        q = self.quality_encoder(quality_context)
        gate = torch.sigmoid(self.gate(torch.cat([zc, zh, q], dim=-1)))
        has_hrv = token_valid.any(dim=1, keepdim=True)
        gate = gate * has_hrv.to(gate.dtype)
        fused = gate * zh + (1.0 - gate) * zc
        logits = self.head(fused)
        return (logits, {'gate_mean': gate.mean(dim=1), 'gate_vector': gate, 'hrv_valid_prop': valid_prop.squeeze(1), 'attention_c_from_h': attention_c_from_h, 'attention_h_from_c': attention_h_from_c})

def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

def safe_auroc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float('nan')

def safe_auprc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float('nan')

def select_threshold(y: np.ndarray, probability: np.ndarray, method: str='f1') -> float:
    """Choose a classification threshold using validation predictions only."""
    y = np.asarray(y).astype(int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.01, np.linspace(0.02, 0.98, 193), probability, 0.99])
    best_score, best_threshold = (-np.inf, 0.5)
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        if method == 'f1':
            score = f1_score(y, prediction, zero_division=0)
        elif method == 'youden':
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            score = sensitivity + specificity - 1.0
        else:
            raise ValueError("threshold_method must be 'f1' or 'youden'")
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold
