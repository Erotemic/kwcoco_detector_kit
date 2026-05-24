#!/usr/bin/env python3
"""
CLI for the sea-lion detector training-run registry.

The registry lives at docs/training_runs.yaml. Class schemes are loaded
read-only from docs/class_schemes.yaml. Run this on any host the registry has
been rsync'd to; edits are line-stable so diffs stay small.

Subcommands:

    list                Print a one-row-per-run summary table.
    show <id>           Print the full YAML for a single run.
    schemes             List class-collapse schemes from class_schemes.yaml.
    add ...             Add a new planned run. Auto-fills created/updated/host.
    update <id> ...     Mutate fields on an existing run (status, metrics, paths, notes).
    summary             Group runs by (scheme, phase, status) and show counts + best metrics.

Examples:

    python3 scripts/training_registry.py list
    python3 scripts/training_registry.py schemes
    python3 scripts/training_registry.py add \\
        --scheme single_sealion --variant deimv2_dinov3_s \\
        --phase baseline_server --host arisia \\
        --kcd-root /data/users/jon.crall/dvc-repos/viame_sealions_2026/kcd_root \\
        --train-kwcoco training_ready_v1/train.kwcoco.zip \\
        --vali-kwcoco  training_ready_v1/vali.kwcoco.zip \\
        --test-kwcoco  training_ready_v1/test.kwcoco.zip \\
        --train-policy multiscale_512_768 --input-hw 640 640 \\
        --num-epochs 30 --batch-size 16 \\
        --note "first server baseline"
    python3 scripts/training_registry.py update <id> --status running
    python3 scripts/training_registry.py update <id> --status done \\
        --metric vali_map=0.412 --metric vali_map50=0.78 \\
        --artifact detect_metrics_json=kcd_root/eval/<cand>/eval/detect_metrics.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import socket
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO / 'docs' / 'training_runs.yaml'
SCHEMES_PATH = REPO / 'docs' / 'class_schemes.yaml'

STATUS_VALUES = ('planned', 'queued', 'running', 'done', 'failed', 'abandoned')
PHASE_VALUES = ('smoke', 'baseline_mobile', 'baseline_server',
                'operational_pup', 'operational_agesex', 'ablation')

# Canonical field order — used when re-writing the registry so diffs stay tidy.
RUN_FIELD_ORDER = [
    'id', 'scheme', 'variant', 'phase', 'host', 'status',
    'created', 'updated',
    'kcd_root', 'train_kwcoco', 'vali_kwcoco', 'test_kwcoco',
    'train_policy', 'input_hw', 'num_epochs', 'batch_size',
    'metrics', 'artifacts', 'notes',
]


# ---------- I/O ----------

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {'runs': []}
    data = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    data.setdefault('runs', [])
    return data


def load_schemes() -> dict:
    if not SCHEMES_PATH.exists():
        return {}
    data = yaml.safe_load(SCHEMES_PATH.read_text()) or {}
    return data.get('schemes', {})


def _ordered_run(run: dict) -> dict:
    out = {}
    for k in RUN_FIELD_ORDER:
        if k in run:
            out[k] = run[k]
    for k, v in run.items():
        if k not in out:
            out[k] = v
    return out


def save_registry(data: dict) -> None:
    data['runs'] = [_ordered_run(r) for r in data.get('runs', [])]
    text = yaml.safe_dump(
        data, sort_keys=False, default_flow_style=False, width=100,
        allow_unicode=True,
    )
    REGISTRY_PATH.write_text(text)


# ---------- helpers ----------

def now_iso() -> str:
    return dt.date.today().isoformat()


def current_host() -> str:
    name = socket.gethostname()
    return name.split('.')[0]


def find_run(data: dict, run_id: str) -> dict:
    for run in data['runs']:
        if run.get('id') == run_id:
            return run
    raise SystemExit(f'error: no run with id={run_id!r}')


def make_id(scheme: str, variant: str, host: str, suffix: str | None = None) -> str:
    base = f'{now_iso()}-{host}-{variant}-{scheme}'
    if suffix:
        base = f'{base}-{suffix}'
    return base


def parse_kv_pairs(items: list[str], coerce_numbers: bool = True) -> dict:
    out = {}
    for raw in items or []:
        if '=' not in raw:
            raise SystemExit(f'error: expected key=value, got: {raw!r}')
        k, v = raw.split('=', 1)
        k = k.strip()
        v = v.strip()
        if coerce_numbers and v not in ('', 'null', 'None'):
            try:
                if '.' in v or 'e' in v.lower():
                    out[k] = float(v)
                    continue
                out[k] = int(v)
                continue
            except ValueError:
                pass
        if v in ('', 'null', 'None'):
            out[k] = None
        else:
            out[k] = v
    return out


# ---------- commands ----------

def cmd_list(args) -> int:
    data = load_registry()
    runs = data['runs']
    if args.scheme:
        runs = [r for r in runs if r.get('scheme') == args.scheme]
    if args.host:
        runs = [r for r in runs if r.get('host') == args.host]
    if args.status:
        runs = [r for r in runs if r.get('status') == args.status]
    if not runs:
        print('(no runs)')
        return 0
    cols = [
        ('id', 50), ('scheme', 16), ('variant', 22), ('host', 10),
        ('status', 9), ('vali_map', 8), ('vali_map50', 10), ('updated', 10),
    ]
    header = '  '.join(f'{c:<{w}}' for c, w in cols)
    print(header)
    print('-' * len(header))
    for r in runs:
        metrics = r.get('metrics') or {}
        def fmt(v):
            if v is None:
                return '-'
            if isinstance(v, float):
                return f'{v:.3f}'
            return str(v)
        row = {
            'id': r.get('id', '?'),
            'scheme': r.get('scheme', '?'),
            'variant': r.get('variant', '?'),
            'host': r.get('host', '?'),
            'status': r.get('status', '?'),
            'vali_map': fmt(metrics.get('vali_map')),
            'vali_map50': fmt(metrics.get('vali_map50')),
            'updated': r.get('updated', '-'),
        }
        print('  '.join(f'{row[c]:<{w}}' for c, w in cols))
    return 0


def cmd_show(args) -> int:
    data = load_registry()
    run = find_run(data, args.id)
    buf = io.StringIO()
    yaml.safe_dump(_ordered_run(run), buf, sort_keys=False, default_flow_style=False, width=100)
    print(buf.getvalue().rstrip())
    return 0


def cmd_schemes(args) -> int:
    schemes = load_schemes()
    if not schemes:
        print(f'(no schemes in {SCHEMES_PATH})')
        return 0
    for name, info in schemes.items():
        n_cls = info.get('num_classes', '?')
        desc = (info.get('description') or '').strip()
        print(f'{name}  ({n_cls} cls)')
        if desc:
            print(f'    {desc}')
        mapping = info.get('mapping') or {}
        if mapping:
            inv = {}
            for src, tgt in mapping.items():
                inv.setdefault(tgt, []).append(src)
            for tgt in sorted(inv):
                print(f'    {tgt:<20} <- {", ".join(sorted(inv[tgt]))}')
        drop = info.get('drop') or []
        if drop:
            print(f'    drop: {", ".join(drop)}')
        print()
    return 0


def cmd_add(args) -> int:
    schemes = load_schemes()
    if schemes and args.scheme not in schemes:
        print(f'warning: scheme {args.scheme!r} not in {SCHEMES_PATH.name}; known: {list(schemes)}',
              file=sys.stderr)

    data = load_registry()
    host = args.host or current_host()
    run_id = args.id or make_id(args.scheme, args.variant, host, args.suffix)
    if any(r.get('id') == run_id for r in data['runs']):
        raise SystemExit(f'error: id {run_id!r} already exists')

    run = {
        'id': run_id,
        'scheme': args.scheme,
        'variant': args.variant,
        'phase': args.phase,
        'host': host,
        'status': args.status,
        'created': now_iso(),
        'updated': now_iso(),
    }
    if args.kcd_root:
        run['kcd_root'] = args.kcd_root
    if args.train_kwcoco:
        run['train_kwcoco'] = args.train_kwcoco
    if args.vali_kwcoco:
        run['vali_kwcoco'] = args.vali_kwcoco
    if args.test_kwcoco:
        run['test_kwcoco'] = args.test_kwcoco
    if args.train_policy:
        run['train_policy'] = args.train_policy
    if args.input_hw:
        run['input_hw'] = list(args.input_hw)
    if args.num_epochs is not None:
        run['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        run['batch_size'] = args.batch_size
    if args.note:
        run['notes'] = args.note

    data['runs'].append(run)
    save_registry(data)
    print(f'added: {run_id}')
    return 0


def cmd_update(args) -> int:
    data = load_registry()
    run = find_run(data, args.id)

    if args.status:
        run['status'] = args.status
    if args.phase:
        run['phase'] = args.phase
    if args.kcd_root:
        run['kcd_root'] = args.kcd_root
    if args.train_kwcoco:
        run['train_kwcoco'] = args.train_kwcoco
    if args.vali_kwcoco:
        run['vali_kwcoco'] = args.vali_kwcoco
    if args.test_kwcoco:
        run['test_kwcoco'] = args.test_kwcoco
    if args.train_policy:
        run['train_policy'] = args.train_policy
    if args.input_hw:
        run['input_hw'] = list(args.input_hw)
    if args.num_epochs is not None:
        run['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        run['batch_size'] = args.batch_size

    if args.metric:
        metrics = run.setdefault('metrics', {})
        new_metrics = parse_kv_pairs(args.metric)
        per_class = {k: v for k, v in new_metrics.items() if k.startswith('per_class.')}
        for k in per_class:
            new_metrics.pop(k)
        if per_class:
            pc = metrics.setdefault('per_class', {})
            for k, v in per_class.items():
                pc[k.removeprefix('per_class.')] = v
        metrics.update(new_metrics)

    if args.artifact:
        artifacts = run.setdefault('artifacts', {})
        artifacts.update(parse_kv_pairs(args.artifact, coerce_numbers=False))

    if args.note:
        prior = (run.get('notes') or '').rstrip()
        stamp = now_iso()
        new_line = f'[{stamp}] {args.note}'
        run['notes'] = f'{prior}\n{new_line}'.lstrip() if prior else new_line

    run['updated'] = now_iso()
    save_registry(data)
    print(f'updated: {run["id"]}')
    return 0


def cmd_summary(args) -> int:
    data = load_registry()
    runs = data['runs']
    if not runs:
        print('(no runs)')
        return 0
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r.get('scheme', '?'), r.get('phase', '?'), r.get('status', '?'))
        buckets.setdefault(key, []).append(r)
    print(f"{'scheme':<16}  {'phase':<22}  {'status':<10}  {'count':>5}  {'best vali_map':>14}")
    print('-' * 80)
    for key in sorted(buckets):
        scheme, phase, status = key
        rs = buckets[key]
        best = None
        for r in rs:
            v = (r.get('metrics') or {}).get('vali_map')
            if isinstance(v, (int, float)) and (best is None or v > best):
                best = v
        best_str = '-' if best is None else f'{best:.3f}'
        print(f'{scheme:<16}  {phase:<22}  {status:<10}  {len(rs):>5}  {best_str:>14}')
    return 0


# ---------- argparse plumbing ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    p_list = sub.add_parser('list', help='List runs.')
    p_list.add_argument('--scheme')
    p_list.add_argument('--host')
    p_list.add_argument('--status', choices=STATUS_VALUES)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser('show', help='Show one run in full.')
    p_show.add_argument('id')
    p_show.set_defaults(func=cmd_show)

    p_sch = sub.add_parser('schemes', help='List class-collapse schemes.')
    p_sch.set_defaults(func=cmd_schemes)

    p_sum = sub.add_parser('summary', help='Summary grouped by (scheme, phase, status).')
    p_sum.set_defaults(func=cmd_summary)

    p_add = sub.add_parser('add', help='Add a new run.')
    p_add.add_argument('--id', help='Override the auto-generated id.')
    p_add.add_argument('--suffix', help='Optional suffix appended to the auto-generated id.')
    p_add.add_argument('--scheme', required=True)
    p_add.add_argument('--variant', required=True, help='kit trainer variant, e.g. deimv2_dinov3_s')
    p_add.add_argument('--phase', choices=PHASE_VALUES)
    p_add.add_argument('--host', help='Defaults to current hostname.')
    p_add.add_argument('--status', default='planned', choices=STATUS_VALUES)
    p_add.add_argument('--kcd-root')
    p_add.add_argument('--train-kwcoco')
    p_add.add_argument('--vali-kwcoco')
    p_add.add_argument('--test-kwcoco')
    p_add.add_argument('--train-policy')
    p_add.add_argument('--input-hw', nargs=2, type=int, metavar=('H', 'W'))
    p_add.add_argument('--num-epochs', type=int)
    p_add.add_argument('--batch-size', type=int)
    p_add.add_argument('--note', help='Free-form note (stored in notes).')
    p_add.set_defaults(func=cmd_add)

    p_upd = sub.add_parser('update', help='Update an existing run.')
    p_upd.add_argument('id')
    p_upd.add_argument('--status', choices=STATUS_VALUES)
    p_upd.add_argument('--phase', choices=PHASE_VALUES)
    p_upd.add_argument('--kcd-root')
    p_upd.add_argument('--train-kwcoco')
    p_upd.add_argument('--vali-kwcoco')
    p_upd.add_argument('--test-kwcoco')
    p_upd.add_argument('--train-policy')
    p_upd.add_argument('--input-hw', nargs=2, type=int, metavar=('H', 'W'))
    p_upd.add_argument('--num-epochs', type=int)
    p_upd.add_argument('--batch-size', type=int)
    p_upd.add_argument('--metric', action='append',
                       help='key=value, repeatable. Use per_class.<name>=ap for per-class entries.')
    p_upd.add_argument('--artifact', action='append',
                       help='key=value, repeatable. Values are kept as strings.')
    p_upd.add_argument('--note', help='Appends a dated line to notes.')
    p_upd.set_defaults(func=cmd_update)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
