"""Nested patient-exclusive GCA experiments. No historical predictions are reused."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import copy, hashlib, json, platform, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix
from encoders import ExperimentConfig, FoldPreprocessor, MultimodalDataset, BaseFusionModel, ClinicalOnlyModel, HRVOnlyModel, SimpleFusionModel, GatedCrossAttentionModelV1, QualityBiasedCrossAttention, PredictionHead, masked_mean, set_seed, resolve_device, move_batch_to_device, validate_input_tables, select_threshold
CLINICAL = ['age_at_admit', 'prior_admissions', 'comorbidity_count', 'heart_rate_mean', 'sbp_mean', 'resp_rate_mean', 'spo2_mean', 'lab_glucose_mean', 'lab_creatinine_mean']
HRV = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_CVNN', 'HRV_MedianNN']
QUALITY = ['sqi', 'rr_valid_fraction', 'artifact_fraction']
BASELINES = ['clinical_only', 'hrv_only', 'simple_fusion']

@dataclass(frozen=True)
class Candidate:
    family: str
    dim: int = 32
    heads: int = 4
    dropout: float = 0.15
    lr: float = 0.0003
    weight_decay: float = 0.0001
    weighted_bce: bool = True
    residual: bool = False
    reliability: bool = False
    warm_start: bool = False
    aux_weight: float = 0.0
    modality_dropout: float = 0.0
    freeze_epochs: int = 0
    encoder_lr_ratio: float = 1.0
    pretrain_epochs: int = 15
    bidirectional: bool = True
    scalar_gate: bool = False

    @property
    def key(self):
        return self.family + '_' + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

@dataclass
class Protocol:
    seeds: tuple = (1, 11, 21)
    outer_folds: int = 5
    inner_folds: int = 3
    max_epochs: int = 80
    patience: int = 15
    min_delta: float = 0.0001
    batch_size: int = 16
    bootstrap_iterations: int = 1000
    device: str = 'auto'
    max_positive_weight: float = 4.0
    run_ablations: bool = True
    save_checkpoints: bool = True
    calibrate: bool = True

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_inputs(clinical_path, labels_path, audit_dir):
    """Strict audit provenance and admission/hour checks. Never use raw HRV."""
    audit_dir = Path(audit_dir)
    report = json.loads((audit_dir / 'audit_run.json').read_text())
    hp = audit_dir / 'audited_hrv_hourly.csv'
    if report.get('status') != 'complete' or report.get('schema_version') != 'ecg_audit_v1':
        raise ValueError('Complete the ECG audit before modeling.')
    if sha256(hp) != report.get('approved_hrv_sha256'):
        raise ValueError('HRV checksum mismatch. Rerun audit; do not bypass this check.')
    h = pd.read_csv(hp)
    c = pd.read_csv(clinical_path)
    y = pd.read_csv(labels_path)
    keys = ['subject_id', 'hadm_id']
    for df in (c, y, h):
        for k in keys:
            nums = pd.to_numeric(df[k].astype(str).str.replace('^p', '', regex=True), errors='raise')
            if nums.isna().any() or not np.equal(nums, np.floor(nums)).all():
                raise ValueError('Patient/admission identifiers must be nonmissing integers.')
            df[k] = nums.astype('int64')
    if h.empty or len(h) != report.get('accepted_hours'):
        raise ValueError('Empty or incomplete approved HRV export.')
    if not h.audit_pass.eq(1).all() or not h.hrv_valid.eq(1).all():
        raise ValueError('Rejected HRV rows must never enter modeling.')
    if h.hadm_id.nunique() != report.get('accepted_admissions'):
        raise ValueError('Audit admission-count mismatch.')
    if h.duplicated(keys + ['hour']).any():
        raise ValueError('Duplicate admission-hour.')
    if h.groupby('hadm_id').subject_id.nunique().max() != 1:
        raise ValueError('An admission maps to more than one patient.')
    hours = pd.to_numeric(h.hour, errors='raise')
    if not np.isfinite(hours).all() or not np.equal(hours, np.floor(hours)).all() or (not hours.between(0, 23).all()):
        raise ValueError('Approved hours must be integer positions 0..23.')
    for col in QUALITY + ['coverage_fraction']:
        if col not in h or not np.isfinite(h[col]).all() or (not h[col].between(0, 1).all()):
            raise ValueError(f'Invalid or absent quality column: {col}')
    if (h.coverage_fraction + 1e-09 < report['policy']['min_coverage']).any():
        raise ValueError('Approved export violates recorded coverage policy.')
    missing = set(CLINICAL) - set(c)
    if missing:
        raise ValueError(f'Clinical feature schema changed: {sorted(missing)}')
    if c.duplicated(keys).any() or y.duplicated(keys).any():
        raise ValueError('Duplicate clinical or label key.')
    eligible = h[keys].drop_duplicates()
    joined = eligible.merge(c[keys + CLINICAL], on=keys, how='left', validate='one_to_one', indicator=True)
    no_clinical = int(joined['_merge'].ne('both').sum())
    joined = joined.loc[joined['_merge'].eq('both')].drop(columns='_merge')
    joined = joined.merge(y[keys + ['y_30day']], on=keys, how='left', validate='one_to_one', indicator=True)
    if joined['_merge'].ne('both').any() or joined.y_30day.isna().any():
        raise ValueError('Audit-matched clinical admissions lack outcome labels.')
    joined = joined.drop(columns='_merge').rename(columns={'y_30day': 'label'})
    h = h.merge(joined[keys], on=keys, how='inner', validate='many_to_one')
    validate_input_tables(joined, h, CLINICAL, HRV, QUALITY, ExperimentConfig(timesteps=24))
    provenance = dict(audit=report, clinical_sha256=sha256(clinical_path), labels_sha256=sha256(labels_path), unmatched_audit_admissions=no_clinical, eligible_admissions=len(joined), subjects=int(joined.subject_id.nunique()), positive=int(joined.label.sum()), approved_hours=len(h))
    return (joined.reset_index(drop=True), h.reset_index(drop=True), provenance)

class CoverageDataset(MultimodalDataset):
    """Coverage stays separate: G0 and all encoders retain the original 3 SQIs."""

    def __init__(self, arrays, hrv):
        super().__init__(arrays)
        lookup = {(str(r.hadm_id), int(r.hour)): float(r.coverage_fraction) for r in hrv.itertuples()}
        self.coverage = np.zeros(arrays.token_valid.shape, dtype=np.float32)
        for i, admission in enumerate(arrays.admission_ids):
            for hour in np.flatnonzero(arrays.token_valid[i]):
                self.coverage[i, hour] = lookup[str(admission), int(hour)]

    def __getitem__(self, index):
        b = super().__getitem__(index)
        b['coverage'] = torch.from_numpy(self.coverage[index])
        return b

def configuration(candidate, protocol):
    return ExperimentConfig(timesteps=24, batch_size=protocol.batch_size, latent_dim=candidate.dim, num_heads=candidate.heads, num_layers=1, dropout=candidate.dropout, learning_rate=candidate.lr, weight_decay=candidate.weight_decay, device=protocol.device, max_positive_weight=protocol.max_positive_weight)

class ResidualGCA(BaseFusionModel):
    """Preserves pre-attention clinical base. Reliability is an experimental proxy."""

    def __init__(self, nc, nh, nq, cfg, candidate):
        super().__init__(nc, nh, nq, cfg)
        self.candidate = candidate
        d = cfg.latent_dim
        self.c_from_h = QualityBiasedCrossAttention(d, cfg.num_heads, cfg.dropout)
        self.h_from_c = QualityBiasedCrossAttention(d, cfg.num_heads, cfg.dropout) if candidate.bidirectional else None
        self.c_norm = nn.LayerNorm(d)
        self.h_norm = nn.LayerNorm(d)
        context_size = nq + 3 + (2 if candidate.reliability else 0)
        self.quality_encoder = nn.Sequential(nn.Linear(context_size, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.gate = nn.Linear(3 * d, 1 if candidate.scalar_gate else d)
        self.correction = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), nn.Tanh())
        self.alpha_logit = nn.Parameter(torch.tensor(float(np.log(0.1 / 0.9))))
        self.head = PredictionHead(d, d, cfg.dropout)
        self.hrv_head = PredictionHead(d, d, cfg.dropout)

    def forward(self, b):
        hc, zc = self.encode_clinical(b)
        hh, zh0 = self.encode_hrv(b)
        valid = b['token_valid'].bool()
        cv = torch.ones(hc.shape[:2], device=hc.device, dtype=torch.bool)
        cc, _ = self.c_from_h(hc, hh, valid, b['qh'][..., 0])
        zcx = self.c_norm(hc + cc).mean(1)
        if self.h_from_c is not None:
            hcxt, _ = self.h_from_c(hh, hc, cv)
            zh = masked_mean(self.h_norm(hh + hcxt), valid)
        else:
            zh = zh0
        quality = masked_mean(b['qh'], valid)
        prop = valid.float().mean(1, keepdim=True)
        context = [quality, prop, b['mc'].mean(1, keepdim=True), b['dc'].mean(1, keepdim=True)]
        coverage = masked_mean(b['coverage'], valid).unsqueeze(1)
        available = masked_mean(b['mh'].mean(-1), valid).unsqueeze(1)
        if self.candidate.reliability:
            context.extend([coverage, available])
        q = self.quality_encoder(torch.cat(context, dim=1))
        learned = torch.sigmoid(self.gate(torch.cat([zcx, zh, q], dim=1)))
        reliability = (quality[:, 0:1] * coverage * prop.sqrt()).clamp(0, 1)
        has = valid.any(1, keepdim=True).to(zc.dtype)
        effective = learned * has
        if self.candidate.reliability:
            effective = effective * reliability
        alpha = torch.sigmoid(self.alpha_logit)
        correction = self.correction(torch.cat([zc, zcx, zh], dim=1))
        fused = zc + alpha * effective * correction
        return (self.head(fused), dict(gate_mean=effective.mean(1), learned_gate=learned.mean(1), signal_reliability=reliability.squeeze(1), residual_scale=alpha.expand(len(zc)), hrv_valid_prop=prop.squeeze(1), clinical_logits=self.head(zc), hrv_logits=self.hrv_head(zh0), gate_vector=effective))

def make_model(c, protocol):
    cfg = configuration(c, protocol)
    classes = {'clinical_only': ClinicalOnlyModel, 'hrv_only': HRVOnlyModel, 'simple_fusion': SimpleFusionModel}
    if c.family in classes:
        return classes[c.family](len(CLINICAL), len(HRV), len(QUALITY), cfg)
    if not c.residual:
        return GatedCrossAttentionModelV1(len(CLINICAL), len(HRV), len(QUALITY), cfg)
    return ResidualGCA(len(CLINICAL), len(HRV), len(QUALITY), cfg, c)

def dropout_hrv(b, probability):
    if probability == 0:
        return b
    dropped = torch.rand(len(b['y']), device=b['y'].device) < probability
    out = dict(b)
    for k in ['xh', 'mh', 'dh', 'qh', 'token_valid', 'coverage']:
        out[k] = b[k].clone()
        out[k][dropped] = 0
    return out

def loader(ds, p, shuffle, seed):
    return DataLoader(ds, batch_size=p.batch_size, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))

def predict(model, ds, p, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader(ds, p, False, 0):
            b = move_batch_to_device(batch, device)
            logit, aux = model(b)
            row = pd.DataFrame(dict(subject_id=batch['subject_id'], admission_id=batch['admission_id'], y_true=b['y'].cpu().numpy().astype(int), logit=logit.cpu().numpy(), probability=torch.sigmoid(logit).cpu().numpy()))
            for k in ['gate_mean', 'learned_gate', 'signal_reliability', 'residual_scale', 'hrv_valid_prop']:
                if k in aux:
                    row[k] = aux[k].detach().cpu().numpy()
            if 'gate_vector' in aux:
                arr = aux['gate_vector'].detach().cpu().numpy()
                for j in range(arr.shape[1]):
                    row[f'gate_{j:02d}'] = arr[:, j]
            rows.append(row)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result.probability).all():
        raise FloatingPointError('Nonfinite prediction')
    return result

def metric_values(y, prob):
    return dict(auprc=float(average_precision_score(y, prob)), auroc=float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float('nan'), brier=float(brier_score_loss(y, prob)))

def loss_fn(y, c, p, device):
    weight = min(float((y == 0).sum()) / max(float((y == 1).sum()), 1), p.max_positive_weight) if c.weighted_bce else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))

def train_epoch(model, ds, c, p, device, optimizer, criterion, epoch, seed):
    model.train()
    frozen = c.warm_start and epoch <= c.freeze_epochs
    frozen_modules = [model.clinical_encoder, model.hrv_encoder]
    if c.warm_start:
        frozen_modules.append(model.head)
    if frozen:
        for module in frozen_modules:
            module.eval()
    for module in frozen_modules:
        for par in module.parameters():
            par.requires_grad_(not frozen)
    losses = []
    for batch in loader(ds, p, True, seed + epoch):
        b = move_batch_to_device(batch, device)
        b = dropout_hrv(b, c.modality_dropout)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(b)
        loss = criterion(logits, b['y'])
        if c.aux_weight:
            loss = loss + c.aux_weight * criterion(aux['clinical_logits'], b['y'])
            has = b['token_valid'].any(1)
            if has.any():
                loss = loss + c.aux_weight * criterion(aux['hrv_logits'][has], b['y'][has])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))

def fit(train, validation, c, p, seed, fixed_epochs=None):
    """Validation chooses epoch only in inner folds. Outer refit has NO validation."""
    set_seed(seed)
    device = resolve_device(p.device)
    model = make_model(c, p).to(device)
    if c.warm_start:
        for family in ['clinical_only', 'hrv_only']:
            pre = replace(c, family=family, residual=False, reliability=False, warm_start=False, aux_weight=0.0, modality_dropout=0.0, freeze_epochs=0, encoder_lr_ratio=1.0)
            source, _, _ = fit(train, None, pre, p, seed, fixed_epochs=c.pretrain_epochs)
            attr = 'clinical_encoder' if family == 'clinical_only' else 'hrv_encoder'
            getattr(model, attr).load_state_dict(getattr(source, attr).state_dict())
            if family == 'clinical_only':
                model.head.load_state_dict(source.head.state_dict())
            elif hasattr(model, 'hrv_head'):
                model.hrv_head.load_state_dict(source.head.state_dict())
            del source
    low = []
    other = []
    for name, par in model.named_parameters():
        (low if c.warm_start and name.startswith(('clinical_encoder.', 'hrv_encoder.', 'head.')) else other).append(par)
    groups = [{'params': other, 'lr': c.lr}]
    if low:
        groups.append({'params': low, 'lr': c.lr * c.encoder_lr_ratio})
    optimizer = torch.optim.AdamW(groups, weight_decay=c.weight_decay)
    criterion = loss_fn(train.arrays.y, c, p, device)
    history = []
    best = None
    best_ap = -np.inf
    best_brier = np.inf
    best_epoch = 0
    stale = 0
    limit = int(fixed_epochs) if fixed_epochs is not None else p.max_epochs
    if limit < 1:
        raise ValueError('Epoch budget must be positive')
    for epoch in range(1, limit + 1):
        loss = train_epoch(model, train, c, p, device, optimizer, criterion, epoch, seed)
        row = {'epoch': epoch, 'train_loss': loss}
        if validation is not None:
            pred = predict(model, validation, p, device)
            met = metric_values(pred.y_true, pred.probability)
            row.update(met)
            better = met['auprc'] > best_ap + p.min_delta or (abs(met['auprc'] - best_ap) <= p.min_delta and met['brier'] < best_brier - p.min_delta)
            if better:
                best_ap = met['auprc']
                best_brier = met['brier']
                best_epoch = epoch
                best = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= p.patience:
            break
    if validation is not None:
        if best is None:
            raise RuntimeError('No checkpoint selected')
        model.load_state_dict(best)
    else:
        best_epoch = limit
    return (model, pd.DataFrame(history), best_epoch)

def checked_splits(frame, n, seed):
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, frame.label, frame.subject_id))
    for tr, va in splits:
        a, b = (frame.iloc[tr], frame.iloc[va])
        if set(a.subject_id) & set(b.subject_id):
            raise AssertionError('Patient leakage')
        if a.label.nunique() != 2 or b.label.nunique() != 2:
            raise ValueError('A predefined split lacks a class. Revise protocol before running, not after seeing scores.')
    return splits

def datasets(train, validation, h, c, p):
    cfg = configuration(c, p)
    train_h = h[h.hadm_id.isin(train.hadm_id)]
    prep = FoldPreprocessor(CLINICAL, HRV, QUALITY, cfg).fit(train, train_h)
    a = CoverageDataset(prep.transform(train, h), h)
    b = CoverageDataset(prep.transform(validation, h), h)
    return (a, b, prep)

def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def refit_predict(outer_train, test, h, c, p, selection, seed, root, label):
    train_ds, test_ds, prep = datasets(outer_train, test, h, c, p)
    model, hist, _ = fit(train_ds, None, c, p, seed + 9000, fixed_epochs=selection['refit_epochs'])
    pred = predict(model, test_ds, p, resolve_device(p.device))
    pred['model'] = label
    pred['candidate_id'] = c.key
    pred['threshold'] = selection['threshold']
    pred['prediction'] = (pred.probability >= pred.threshold).astype(int)
    if p.calibrate:
        from scipy.special import expit
        pred['calibrated_probability'] = expit(selection['calibration_slope'] * pred.logit + selection['calibration_intercept'])
        pred['calibrated_threshold'] = selection['calibrated_threshold']
        pred['calibrated_prediction'] = (pred.calibrated_probability >= pred.calibrated_threshold).astype(int)
    atomic_csv(hist, root / f'{label}_refit_history.csv')
    if p.save_checkpoints:
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'candidate': asdict(c), 'protocol': asdict(p), 'preprocessor': prep, 'threshold': selection['threshold'], 'selection': selection, 'clinical_features': CLINICAL, 'hrv_features': HRV, 'quality_features': QUALITY}, root / f'{label}_checkpoint.pt')
    del model
    return pred
