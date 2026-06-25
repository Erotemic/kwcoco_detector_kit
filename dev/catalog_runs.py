#!/usr/bin/env python3
"""
Build a comprehensive catalog of all training runs from log files and eval directories.

Scans slurm_logs/*.out and kcd_sealion/*.log for files containing a
'=== run config ===' block (the universal signal that a training pipeline
ran).  For each log it extracts:

  * Kit-level config (from === run config === block)
  * DEIMv2 resolved config (from the 'cfg: {...}' dict line) — used for
    cross-validation against the kit config
  * Training dynamics (per-epoch memory, in-loop mAP, outcome, failures)
  * Tiled eval AP (from detect_metrics.json in the run directory)

IMPORTANT: this script accepts only path arguments.  No algorithm-affecting
settings are overridden here so that every log record maps cleanly onto the
script that launched it.

Usage:
    python3 dev/catalog_runs.py [--slurm-log-dir PATH] [--kcd-data-dir PATH]
                                [--out-json PATH] [--out-md PATH]
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Filename prefixes that are NOT training logs
SKIP_PREFIXES = (
    'build_',
    'gen002_data_prep',
    'rescore_',
    'aiq_tiles_',
)

# Known GPU models -> VRAM in MB
GPU_VRAM_MB = {
    'NVIDIA RTX PRO 6000 Blackwell Max-Q': 98304,   # 96 GB
    'NVIDIA RTX A6000': 49152,                        # 48 GB
    'NVIDIA GeForce RTX 3090': 24576,                 # 24 GB
}

# Eval type priority for summary table: first match wins
EVAL_TYPE_PRIORITY = ['tiled', 'standard', 'wholeimage', 'per_checkpoint', 'other']


# ---------------------------------------------------------------------------
# Log discovery
# ---------------------------------------------------------------------------

def discover_logs(slurm_log_dir: Path, kcd_data_dir: Path) -> list:
    candidates = []

    if slurm_log_dir.exists():
        for p in sorted(slurm_log_dir.glob('*.out')):
            if not any(p.stem.startswith(pfx) for pfx in SKIP_PREFIXES):
                candidates.append(p)

    if kcd_data_dir.exists():
        for p in sorted(kcd_data_dir.glob('*.log')):
            if not any(p.stem.startswith(pfx) for pfx in SKIP_PREFIXES):
                candidates.append(p)

    return [p for p in candidates if _has_run_config(p)]


def _has_run_config(p: Path) -> bool:
    try:
        with open(p, errors='replace') as fh:
            for line in fh:
                if '=== run config ===' in line:
                    return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Context block parsing  (=== Slurm context === / === standalone docker run ===)
# ---------------------------------------------------------------------------

def parse_context_block(text: str) -> dict:
    result = {
        'is_slurm': '=== Slurm context ===' in text,
        'slurm_job_id': None,
        'host': None,
        'run_name_ctx': None,
        'gpu_model': None,
        'n_gpus_ctx': None,
    }
    gpu_models = []

    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'SLURM_JOB_ID=(\d+)', line)
        if m:
            result['slurm_job_id'] = int(m.group(1))
        m = re.match(r'HOSTNAME=(\S+)', line)
        if m:
            result['host'] = m.group(1)
        m = re.match(r'RUN_NAME=(\S+)', line)
        if m:
            result['run_name_ctx'] = m.group(1)
        m = re.match(r'GPU\s+\d+:\s+(.+?)\s+\(UUID:', line)
        if m:
            gpu_models.append(m.group(1).strip())

    m = re.search(r'SCHEME=(\S+)\s+VARIANT=(\S+)\s+GPUS=(\d+)', text)
    if m:
        result['n_gpus_ctx'] = int(m.group(3))

    result['gpu_model'] = gpu_models[0] if gpu_models else None
    result['n_gpu_lines'] = len(gpu_models)
    return result


# ---------------------------------------------------------------------------
# Run config block parsing  (=== run config ===)
# ---------------------------------------------------------------------------

def parse_run_config_block(text: str) -> dict:
    kv = {}
    in_block = False
    for line in text.splitlines():
        if '=== run config ===' in line:
            in_block = True
            continue
        if in_block:
            if re.match(r'\s*===', line):
                break
            m = re.match(r'\s{2,}(\w+):\s+(.*)', line)
            if m:
                kv[m.group(1).strip()] = m.group(2).strip()

    result = {}
    result['run_name'] = kv.get('run_name')
    result['scheme'] = kv.get('scheme')
    result['variant'] = kv.get('variant')
    cats_str = kv.get('categories', '')
    result['categories'] = [c.strip() for c in cats_str.split(',') if c.strip()]

    hw_nums = re.findall(r'\d+', kv.get('input_hw', ''))
    result['input_hw'] = [int(x) for x in hw_nums] if hw_nums else None
    result['resolution'] = max(result['input_hw']) if result['input_hw'] else None

    tile_m = re.search(r'scheme-agnostic cache key\s*=\s*([0-9a-f]+)', text)
    result['tile_cache_hash'] = tile_m.group(1) if tile_m else None

    result['kcd_root'] = kv.get('kcd_root')

    m = re.match(r'(\d+)', kv.get('gpus', ''))
    result['n_gpus_cfg'] = int(m.group(1)) if m else None

    batch_str = kv.get('batch', '')
    m = re.search(r'total=(\d+)', batch_str)
    result['total_batch'] = int(m.group(1)) if m else None
    m = re.search(r'per_gpu=(\d+)', batch_str)
    result['per_gpu_batch'] = int(m.group(1)) if m else None

    m = re.match(r'(\d+)', kv.get('epochs', ''))
    result['n_epochs_planned'] = int(m.group(1)) if m else None

    lr_str = kv.get('lr', '')
    m = re.search(r'head=([\d.e+-]+)', lr_str)
    result['lr_head'] = float(m.group(1)) if m else None
    m = re.search(r'backbone=([\d.e+-]+)', lr_str)
    result['lr_backbone'] = float(m.group(1)) if m else None

    result['use_amp'] = kv.get('use_amp', '').lower() == 'true'

    variant = result['variant'] or ''
    m = re.match(r'deimv2_(.*)', variant)
    result['backbone'] = m.group(1) if m else variant

    bal_m = re.search(r'Class-balance[^(]+\(mode=(\w+)\)', text)
    result['balance_mode'] = bal_m.group(1) if bal_m else 'none'

    return result


# ---------------------------------------------------------------------------
# DEIMv2 cfg dict parsing  (the resolved config dict line)
# ---------------------------------------------------------------------------

def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def parse_deimv2_cfg(text: str) -> dict:
    m = re.search(r'\bcfg:\s+(\{.+)', text)
    if not m:
        return {}
    try:
        raw = ast.literal_eval(m.group(1))
    except Exception:
        return {}

    yc = raw.get('yaml_cfg', {})
    opt_params = _get(yc, 'optimizer', 'params') or []

    result = {
        'cfg_epoches': raw.get('epoches'),
        'cfg_flat_epoch': raw.get('flat_epoch'),
        'cfg_no_aug_epoch': raw.get('no_aug_epoch'),
        'cfg_lr_gamma': raw.get('lr_gamma'),
        'cfg_lrscheduler': raw.get('lrsheduler'),
        'cfg_use_amp': raw.get('use_amp'),
        'cfg_use_ema': raw.get('use_ema'),
        'cfg_num_classes': yc.get('num_classes'),
        'cfg_total_batch': _get(yc, 'train_dataloader', 'total_batch_size'),
        'cfg_eval_spatial_size': yc.get('eval_spatial_size'),
        'cfg_optimizer_lr': _get(yc, 'optimizer', 'lr'),
        'cfg_backbone_lr': opt_params[0].get('lr') if (opt_params and isinstance(opt_params[0], dict)) else None,
        'cfg_aug_policy_epochs': _get(yc, 'train_dataloader', 'dataset', 'transforms', 'policy', 'epoch'),
        'cfg_mixup_epochs': _get(yc, 'train_dataloader', 'collate_fn', 'mixup_epochs'),
        'cfg_copyblend_epochs': _get(yc, 'train_dataloader', 'collate_fn', 'copyblend_epochs'),
    }
    return result


# ---------------------------------------------------------------------------
# Training dynamics parsing
# ---------------------------------------------------------------------------

_RE_EPOCH = re.compile(r'Epoch:\s*\[(\d+)\]')
_RE_STEP_MEM = re.compile(
    r'\[(\d+)/(\d+)\].*?cur mem:\s+([\d.]+)\s+max mem:\s+([\d.]+)')
_RE_EVAL_MEM = re.compile(
    r'Test:.*?cur mem:\s+([\d.]+)\s+max mem:\s+([\d.]+)')
_RE_MAP = re.compile(
    r'Average Precision.*?IoU=0\.50:0\.95\s*\|\s*area=\s*all[^\]]*\]\s*=\s*([\d.]+)')
_RE_TS1 = re.compile(r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]')
_RE_TS2 = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')

_NAN_PATS = [
    re.compile(r'tensor\(\[\[\[nan'),
    re.compile(r'nan,\s+nan,\s+nan'),
]
_OOM_PAT = re.compile(r'CUDA out of memory')
_KILL_PATS = [
    # Slurm time-limit and cancel lines (specific enough to avoid false positives)
    re.compile(r'slurmstepd[^:]*:\s*error:.*\b(CANCELLED|KILLED|TIMEOUT)\b', re.IGNORECASE),
    re.compile(r'DUE TO TIME LIMIT'),
]
_EVAL_DONE_PAT = re.compile(r'wrote .*/detect_metrics\.json')
_EXPORT_FAIL_PAT = re.compile(r'\[sweep\].*FAILED at export')
_SWEEP_PAT = re.compile(r'prior train crashed|best_\*\.pth present but no completion marker')


def parse_training_dynamics(text: str, n_epochs_planned) -> dict:
    epoch_mem_max = {}    # epoch -> max max_mem
    epoch_map = {}        # epoch -> last seen mAP
    eval_mem_peaks = []
    failure_events = []
    timestamps = []
    current_epoch = None
    n_epochs_completed = 0

    for line in text.splitlines():
        # Timestamps
        m = _RE_TS1.search(line) or _RE_TS2.search(line)
        if m:
            timestamps.append(m.group(1))

        # Epoch header
        m = _RE_EPOCH.search(line)
        if m:
            current_epoch = int(m.group(1))
            n_epochs_completed = max(n_epochs_completed, current_epoch + 1)

        # Training step memory
        m = _RE_STEP_MEM.search(line)
        if m and current_epoch is not None:
            max_mem = float(m.group(4))
            epoch_mem_max[current_epoch] = max(
                epoch_mem_max.get(current_epoch, 0.0), max_mem)

        # Eval step memory
        m = _RE_EVAL_MEM.search(line)
        if m:
            eval_mem_peaks.append(float(m.group(2)))

        # In-loop mAP
        m = _RE_MAP.search(line)
        if m and current_epoch is not None:
            epoch_map[current_epoch] = float(m.group(1))

        # NaN divergence
        for pat in _NAN_PATS:
            if pat.search(line):
                if 'nan_loss' not in failure_events:
                    failure_events.append('nan_loss')
                break

        # OOM
        if _OOM_PAT.search(line) and 'oom' not in failure_events:
            failure_events.append('oom')

        # Export failure (post-training)
        if _EXPORT_FAIL_PAT.search(line) and 'export_failed' not in failure_events:
            failure_events.append('export_failed')

        # Kill signals
        for pat in _KILL_PATS:
            if pat.search(line) and 'killed' not in failure_events:
                failure_events.append('killed')
                break

        # Sweep noted a prior crash
        if _SWEEP_PAT.search(line) and 'prior_crash_noted' not in failure_events:
            failure_events.append('prior_crash_noted')

    # Memory summary per epoch
    mem_by_epoch = [
        {'epoch': ep, 'peak_max_mem_mb': v}
        for ep, v in sorted(epoch_mem_max.items())
    ]
    global_peak = max(epoch_mem_max.values()) if epoch_mem_max else None
    if eval_mem_peaks:
        global_peak = max(global_peak or 0.0, max(eval_mem_peaks))

    # In-loop mAP list
    in_loop_map = [{'epoch': ep, 'map': v} for ep, v in sorted(epoch_map.items())]

    # Wall time
    wall_start = wall_end = wall_hours = None
    if timestamps:
        wall_start = timestamps[0]
        wall_end = timestamps[-1]
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
            try:
                t0 = datetime.strptime(wall_start, fmt)
                t1 = datetime.strptime(wall_end, fmt)
                wall_hours = (t1 - t0).total_seconds() / 3600
                break
            except ValueError:
                continue

    # Outcome
    completed = bool(_EVAL_DONE_PAT.search(text)) or (
        n_epochs_planned and n_epochs_completed >= n_epochs_planned)

    if completed:
        outcome = 'completed'
    elif 'nan_loss' in failure_events and n_epochs_completed < (n_epochs_planned or 9999):
        outcome = 'nan_diverge'
    elif 'oom' in failure_events and n_epochs_completed < (n_epochs_planned or 9999):
        outcome = 'oom'
    elif 'killed' in failure_events:
        outcome = 'killed'
    elif n_epochs_completed > 0:
        outcome = 'partial'
    elif failure_events:
        outcome = 'error'
    else:
        outcome = 'unknown'

    # had_failures: any event other than export_failed (that's non-training)
    training_failures = [e for e in failure_events if e != 'export_failed']
    had_failures = bool(training_failures)
    if outcome != 'completed' and outcome not in ('unknown',):
        had_failures = True

    return {
        'outcome': outcome,
        'n_epochs_completed': n_epochs_completed,
        'had_failures': had_failures,
        'failure_events': failure_events,
        'global_peak_mem_mb': global_peak,
        'mem_by_epoch': mem_by_epoch,
        'in_loop_map': in_loop_map,
        'wall_time_start': wall_start,
        'wall_time_end': wall_end,
        'wall_time_hours': round(wall_hours, 2) if wall_hours is not None else None,
    }


# ---------------------------------------------------------------------------
# Eval directory parsing
# ---------------------------------------------------------------------------

_EVAL_TYPE_PATTERNS = [
    (re.compile(r'/tiled_compare/eval/tiled/'), 'tiled'),
    (re.compile(r'/tiled_compare/eval/wholeimage/'), 'wholeimage'),
    (re.compile(r'/per_checkpoint_eval/'), 'per_checkpoint'),
    (re.compile(r'/eval/[^/]+/eval/detect_metrics\.json$'), 'standard'),
]


def _classify_eval_path(p: Path) -> str:
    s = str(p)
    for pat, label in _EVAL_TYPE_PATTERNS:
        if pat.search(s):
            return label
    return 'other'


def _parse_detect_metrics(p: Path):
    try:
        with open(p) as fh:
            data = json.load(fh)
    except Exception:
        return None

    bucket = data.get('area_range=all,iou_thresh=0.5') or {}
    if not bucket:
        for v in data.values():
            if isinstance(v, dict) and 'nocls_measures' in v:
                bucket = v
                break

    nocls = bucket.get('nocls_measures', {})
    ovr = bucket.get('ovr_measures', {})
    meta = bucket.get('meta', {})

    return {
        'nocls_ap': nocls.get('ap'),
        'realpos_total': nocls.get('realpos_total'),
        'per_class_ap': {
            cls: vals.get('ap')
            for cls, vals in ovr.items()
            if isinstance(vals, dict)
        },
        'eval_timestamp': meta.get('timestamp'),
    }


def find_eval_results(run_dir: Path) -> list:
    results = []
    for p in sorted(run_dir.rglob('detect_metrics.json')):
        data = _parse_detect_metrics(p)
        if data:
            results.append({
                'path': str(p),
                'eval_type': _classify_eval_path(p),
                **data,
            })
    return results


# ---------------------------------------------------------------------------
# Config cross-validation
# ---------------------------------------------------------------------------

def validate_config(run_cfg: dict, deimv2_cfg: dict) -> dict:
    mismatches = []

    def check(field, kit_val, cfg_val):
        if kit_val is None or cfg_val is None:
            return
        # List comparison
        if isinstance(kit_val, list) or isinstance(cfg_val, list):
            if list(kit_val) != list(cfg_val):
                mismatches.append({'field': field, 'kit': kit_val, 'deimv2': cfg_val})
            return
        try:
            if abs(float(kit_val) - float(cfg_val)) > 1e-9:
                mismatches.append({'field': field, 'kit': kit_val, 'deimv2': cfg_val})
        except (TypeError, ValueError):
            if str(kit_val) != str(cfg_val):
                mismatches.append({'field': field, 'kit': kit_val, 'deimv2': cfg_val})

    check('total_batch', run_cfg.get('total_batch'), deimv2_cfg.get('cfg_total_batch'))
    check('n_epochs', run_cfg.get('n_epochs_planned'), deimv2_cfg.get('cfg_epoches'))
    check('lr_head', run_cfg.get('lr_head'), deimv2_cfg.get('cfg_optimizer_lr'))
    check('lr_backbone', run_cfg.get('lr_backbone'), deimv2_cfg.get('cfg_backbone_lr'))
    check('eval_spatial_size', run_cfg.get('input_hw'), deimv2_cfg.get('cfg_eval_spatial_size'))

    return {'match': len(mismatches) == 0, 'mismatches': mismatches}


# ---------------------------------------------------------------------------
# Generation / job-id helpers
# ---------------------------------------------------------------------------

def _extract_generation(run_name) -> str:
    if not run_name:
        return ''
    # gen006, gen005, ... (newer naming)
    m = re.search(r'_(gen\d+)(?:_|$)', run_name)
    if m:
        return m.group(1)
    # v1, v2, ..., v6 (older naming — standalone segment after last host token)
    m = re.search(r'_(v\d+)(?:_|$)', run_name)
    if m:
        return m.group(1)
    return ''


def _job_id_from_filename(p: Path):
    m = re.search(r'-(\d+)\.(out|log)$', p.name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-log record assembly
# ---------------------------------------------------------------------------

def process_log(log_path: Path, kcd_data_dir: Path) -> dict:
    text = log_path.read_text(errors='replace')

    ctx = parse_context_block(text)
    run_cfg = parse_run_config_block(text)
    dcfg = parse_deimv2_cfg(text)

    n_epochs_planned = run_cfg.get('n_epochs_planned')
    dynamics = parse_training_dynamics(text, n_epochs_planned)

    # Job ID: filename has priority over context block (more reliable for retries)
    job_id = _job_id_from_filename(log_path) or ctx.get('slurm_job_id')

    # run_name
    run_name = (
        run_cfg.get('run_name')
        or ctx.get('run_name_ctx')
        or re.sub(r'-\d+$', '', log_path.stem)
    )

    # GPU VRAM
    gpu_model = ctx.get('gpu_model')
    gpu_vram_mb = None
    for model_str, vram in GPU_VRAM_MB.items():
        if gpu_model and model_str in gpu_model:
            gpu_vram_mb = vram
            break

    # n_gpus: run config block is most authoritative
    n_gpus = run_cfg.get('n_gpus_cfg') or ctx.get('n_gpus_ctx') or ctx.get('n_gpu_lines')

    # Find eval results
    eval_results = []
    kcd_root = run_cfg.get('kcd_root')
    if kcd_root:
        rd = Path(kcd_root)
        if not rd.exists():
            rd = kcd_data_dir / 'runs' / run_name
        if rd.exists():
            eval_results = find_eval_results(rd)

    config_validation = validate_config(run_cfg, dcfg)

    return {
        # Identity
        'log_file': str(log_path),
        'run_name': run_name,
        'slurm_job_id': job_id,
        'generation': _extract_generation(run_name),
        'is_slurm': ctx.get('is_slurm', True),
        'host': ctx.get('host'),
        'gpu_model': gpu_model,
        'gpu_vram_mb': gpu_vram_mb,
        'n_gpus': n_gpus,
        # Kit config
        'scheme': run_cfg.get('scheme'),
        'variant': run_cfg.get('variant'),
        'backbone': run_cfg.get('backbone'),
        'categories': run_cfg.get('categories', []),
        'input_hw': run_cfg.get('input_hw'),
        'resolution': run_cfg.get('resolution'),
        'tile_cache_hash': run_cfg.get('tile_cache_hash'),
        'per_gpu_batch': run_cfg.get('per_gpu_batch'),
        'total_batch': run_cfg.get('total_batch'),
        'n_epochs_planned': n_epochs_planned,
        'lr_head': run_cfg.get('lr_head'),
        'lr_backbone': run_cfg.get('lr_backbone'),
        'use_amp': run_cfg.get('use_amp'),
        'balance_mode': run_cfg.get('balance_mode'),
        'kcd_root': kcd_root,
        # DEIMv2 resolved config
        'flat_epoch': dcfg.get('cfg_flat_epoch'),
        'no_aug_epoch': dcfg.get('cfg_no_aug_epoch'),
        'lr_gamma': dcfg.get('cfg_lr_gamma'),
        'lrscheduler': dcfg.get('cfg_lrscheduler'),
        'use_ema': dcfg.get('cfg_use_ema'),
        'num_classes': dcfg.get('cfg_num_classes'),
        'aug_policy_epochs': dcfg.get('cfg_aug_policy_epochs'),
        # Training dynamics
        'outcome': dynamics['outcome'],
        'n_epochs_completed': dynamics['n_epochs_completed'],
        'had_failures': dynamics['had_failures'],
        'failure_events': dynamics['failure_events'],
        'global_peak_mem_mb': dynamics['global_peak_mem_mb'],
        'mem_by_epoch': dynamics['mem_by_epoch'],
        'in_loop_map': dynamics['in_loop_map'],
        'wall_time_start': dynamics['wall_time_start'],
        'wall_time_end': dynamics['wall_time_end'],
        'wall_time_hours': dynamics['wall_time_hours'],
        # Eval results
        'eval_results': eval_results,
        # Config cross-validation
        'config_validation': config_validation,
    }


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------

def _best_eval(records: list, run_name: str) -> tuple:
    """Return (eval_type, nocls_ap, per_class_ap) for the best eval result."""
    # Collect all eval results across all jobs for this run_name
    all_evals = defaultdict(list)
    for r in records:
        if r['run_name'] != run_name:
            continue
        for ev in r.get('eval_results', []):
            all_evals[ev['eval_type']].append(ev)

    for ev_type in EVAL_TYPE_PRIORITY:
        evs = all_evals.get(ev_type, [])
        if not evs:
            continue
        # Among evals of this type, pick highest nocls_ap
        best = max(evs, key=lambda e: e.get('nocls_ap') or 0.0)
        return ev_type, best.get('nocls_ap'), best.get('per_class_ap', {})
    return None, None, {}


def _outcome_sort_key(r: dict) -> tuple:
    order = {'completed': 0, 'partial': 1, 'killed': 2,
             'nan_diverge': 3, 'oom': 4, 'error': 5, 'unknown': 6}
    return (order.get(r['outcome'], 9), -(r.get('n_epochs_completed') or 0))


def _fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        if v >= 1000:
            return f'{v:.0f}'
        elif v < 0.001:
            return f'{v:.3e}'
        else:
            return f'{v:.4f}'
    return str(v)


def build_summary_table(records: list) -> str:
    by_run = defaultdict(list)
    for r in records:
        by_run[r['run_name']].append(r)

    # Collect all class names seen across all eval results
    all_cls = set()
    for r in records:
        for ev in r.get('eval_results', []):
            all_cls.update(ev.get('per_class_ap', {}).keys())
    all_cls = sorted(all_cls)

    header = (
        ['run_name', 'gen', 'scheme', 'backbone', 'res', 'n_gpu',
         'batch', 'lr_head', 'balance', 'outcome', 'clean',
         'failures', 'n_jobs', 'ep_done', 'peak_MB',
         'best_mAP', 'nocls_AP', 'eval_src', 'wall_h']
        + [f'cls:{c}' for c in all_cls]
    )

    rows = []
    for run_name in sorted(by_run.keys()):
        jobs = by_run[run_name]
        best = sorted(jobs, key=_outcome_sort_key)[0]

        ev_type, nocls_ap, per_cls = _best_eval(records, run_name)

        all_events = sorted({e for j in jobs for e in j.get('failure_events', [])})
        any_failure = any(j.get('had_failures') for j in jobs)

        best_map = max(
            (e['map'] for e in best.get('in_loop_map', [])), default=None)

        row = [
            run_name,
            best.get('generation', ''),
            best.get('scheme', ''),
            best.get('backbone', ''),
            _fmt(best.get('resolution')),
            _fmt(best.get('n_gpus')),
            _fmt(best.get('total_batch')),
            _fmt(best.get('lr_head')),
            best.get('balance_mode', ''),
            best.get('outcome', ''),
            'N' if any_failure else 'Y',   # clean=Y means no failures
            ', '.join(all_events) if all_events else '',
            _fmt(len(jobs)),
            _fmt(best.get('n_epochs_completed')),
            _fmt(best.get('global_peak_mem_mb')),
            _fmt(best_map),
            _fmt(nocls_ap),
            ev_type or '',
            _fmt(best.get('wall_time_hours')),
        ] + [_fmt(per_cls.get(c)) for c in all_cls]
        rows.append(row)

    sep = ['---'] * len(header)
    lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(sep) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--slurm-log-dir',
                    default='/data/users/jon.crall/slurm_logs',
                    help='Directory containing *.out slurm training logs')
    ap.add_argument('--kcd-data-dir',
                    default='/data/users/jon.crall/kcd_sealion',
                    help='kcd_sealion root (contains *.log files and runs/)')
    ap.add_argument('--out-json',
                    default='dev/run_catalog.json',
                    help='Output JSON (per-job records)')
    ap.add_argument('--out-md',
                    default='projects/viame_sealions_2026/docs/journals/data/run_catalog.md',
                    help='Output markdown summary table')
    args = ap.parse_args()

    slurm_log_dir = Path(args.slurm_log_dir)
    kcd_data_dir = Path(args.kcd_data_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    print(f'Discovering logs...', file=sys.stderr)
    log_paths = discover_logs(slurm_log_dir, kcd_data_dir)
    print(f'Found {len(log_paths)} training logs.', file=sys.stderr)

    records = []
    errors = []
    for i, p in enumerate(log_paths, 1):
        print(f'  [{i:3}/{len(log_paths)}] {p.name}', file=sys.stderr, end='')
        try:
            rec = process_log(p, kcd_data_dir)
            records.append(rec)
            print(f'  {rec["outcome"]}  ep={rec["n_epochs_completed"]}', file=sys.stderr)
        except Exception as exc:
            errors.append((str(p), str(exc)))
            print(f'  ERROR: {exc}', file=sys.stderr)

    if errors:
        print(f'\n{len(errors)} errors:', file=sys.stderr)
        for path, msg in errors:
            print(f'  {path}: {msg}', file=sys.stderr)

    n_runs = len({r['run_name'] for r in records})
    print(f'\n{len(records)} records, {n_runs} unique run names.', file=sys.stderr)

    # JSON output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    catalog = {
        'catalog_generated': '2026-06-21',
        'n_logs': len(records),
        'n_unique_run_names': n_runs,
        'runs': records,
    }
    with open(out_json, 'w') as fh:
        json.dump(catalog, fh, indent=2, default=str)
    print(f'Wrote {out_json}', file=sys.stderr)

    # Markdown summary
    out_md.parent.mkdir(parents=True, exist_ok=True)
    table = build_summary_table(records)
    with open(out_md, 'w') as fh:
        fh.write(f'# Run catalog\n\n'
                 f'Generated 2026-06-21. '
                 f'{len(records)} job records across {n_runs} unique run names.\n\n'
                 f'Columns: `clean=Y` means no training failures (export failures '
                 f'are post-training and do not affect the clean flag).\n\n')
        fh.write(table)
        fh.write('\n')
    print(f'Wrote {out_md}', file=sys.stderr)


if __name__ == '__main__':
    main()
