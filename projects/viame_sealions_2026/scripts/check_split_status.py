#!/usr/bin/env python3
"""
Check the status of the training-ready splits.

Run on arisia after rsync'ing this script over:

    python3 scripts/check_split_status.py
    python3 scripts/check_split_status.py --repo /data/users/jon.crall/dvc-repos/viame_sealions_2026

Reports on:
  - presence and size of train / vali / test / all_collapsed kwcoco zips
  - per-split image and annotation counts (from prepare_report.json if present,
    or by loading the kwcoco files directly)
  - category histogram per split and overall
  - per-year image counts and which years are in test (vs train+vali)
  - split overlap sanity checks (no shared image names across splits)
  - presence of the splits_v1.json manifest in unpacked/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_NAMES = ('train', 'vali', 'test')
YEAR_RE = re.compile(r'(?<!\d)(20\d{2})(?!\d)')


def human_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} {unit}'
        n /= 1024
    return f'{n} B'


def load_kwcoco_zip(path: Path) -> dict | None:
    """Return the kwcoco json dict from a .kwcoco.zip, without needing kwcoco installed."""
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as zf:
        json_members = [n for n in zf.namelist() if n.endswith('.json')]
        if not json_members:
            return None
        # Prefer one matching the zip stem.
        stem = path.name.replace('.kwcoco.zip', '')
        preferred = [n for n in json_members if stem in n]
        member = preferred[0] if preferred else json_members[0]
        with zf.open(member) as fp:
            return json.load(fp)


def summarize_dataset(dset: dict) -> dict:
    imgs = dset.get('images', [])
    anns = dset.get('annotations', [])
    cats_by_id = {c['id']: c['name'] for c in dset.get('categories', [])}
    cat_hist = Counter(cats_by_id.get(a['category_id'], '?') for a in anns)
    year_hist = Counter()
    image_names = []
    for img in imgs:
        name = img.get('name') or Path(img.get('file_name', '')).stem
        image_names.append(name)
        # Try to extract a year from the file path or name.
        haystack = f"{img.get('file_name', '')} {img.get('name', '')}"
        m = YEAR_RE.search(haystack)
        if m:
            year_hist[m.group(1)] += 1
        else:
            year_hist['unknown'] += 1
    return {
        'n_images': len(imgs),
        'n_annotations': len(anns),
        'cat_hist': dict(cat_hist),
        'year_hist': dict(year_hist),
        'image_names': image_names,
    }


def fmt_hist(hist: dict, total_label: str = 'total') -> str:
    if not hist:
        return '  (empty)'
    width = max(len(str(k)) for k in hist)
    lines = []
    total = sum(hist.values())
    for k in sorted(hist, key=lambda x: (-hist[x], str(x))):
        lines.append(f'    {str(k):<{width}}  {hist[k]:>8}')
    lines.append(f'    {total_label:<{width}}  {total:>8}')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='Repo root (default: cwd). On arisia: /data/users/jon.crall/dvc-repos/viame_sealions_2026')
    parser.add_argument('--training-ready', type=Path, default=None,
                        help='Override path to training_ready_v1 dir (default: <repo>/training_ready_v1)')
    parser.add_argument('--unpacked', type=Path, default=None,
                        help='Override path to unpacked dir (default: <repo>/unpacked)')
    args = parser.parse_args()

    repo = args.repo.resolve()
    tr_dir = (args.training_ready or (repo / 'training_ready_v1')).resolve()
    unpacked = (args.unpacked or (repo / 'unpacked')).resolve()

    print(f'Repo:           {repo}')
    print(f'Training-ready: {tr_dir}')
    print(f'Unpacked:       {unpacked}')
    print()

    if not tr_dir.exists():
        print(f'ERROR: training_ready dir does not exist: {tr_dir}', file=sys.stderr)
        return 2

    # --- Files on disk ---
    expected = {
        'all':   tr_dir / 'all_collapsed.kwcoco.zip',
        'train': tr_dir / 'train.kwcoco.zip',
        'vali':  tr_dir / 'vali.kwcoco.zip',
        'test':  tr_dir / 'test.kwcoco.zip',
    }
    report_fpath = tr_dir / 'prepare_report.json'

    print('Files:')
    for label, p in expected.items():
        if p.exists():
            print(f'  [OK]   {label:<5} {p.name:<30} {human_bytes(p.stat().st_size):>10}')
        else:
            print(f'  [MISS] {label:<5} {p.name:<30}     missing')
    if report_fpath.exists():
        print(f'  [OK]   report {report_fpath.name}')
    else:
        print(f'  [MISS] report {report_fpath.name}')
    print()

    # --- Compare against prepare_report.json if it exists ---
    if report_fpath.exists():
        try:
            report = json.loads(report_fpath.read_text())
            print('prepare_report.json split_stats:')
            for split, stats in (report.get('split_stats') or {}).items():
                print(f"  {split:<5}  images={stats.get('images'):>5}  annotations={stats.get('annotations'):>7}")
            if 'stats' in report:
                src = report['stats'].get('source_category_histogram') or {}
                print('  source category histogram:')
                print(fmt_hist(src))
            print()
        except Exception as exc:  # noqa: BLE001
            print(f'  (could not parse {report_fpath}: {exc})')
            print()

    # --- Load each split and summarize ---
    summaries = {}
    for split in SPLIT_NAMES + ('all',):
        path = expected[split]
        if not path.exists():
            continue
        print(f'Loading {path.name} ...', flush=True)
        dset = load_kwcoco_zip(path)
        if dset is None:
            print(f'  (failed to read {path})')
            continue
        summaries[split] = summarize_dataset(dset)

    if not summaries:
        print('No split files could be loaded.', file=sys.stderr)
        return 3

    print()
    print('Per-split summary:')
    print(f"  {'split':<6}  {'images':>7}  {'anns':>8}  {'#cats':>6}  {'years':<30}")
    for split in SPLIT_NAMES + ('all',):
        s = summaries.get(split)
        if not s:
            continue
        years = sorted(y for y in s['year_hist'] if y != 'unknown')
        years_str = ','.join(years) if years else '(none)'
        if 'unknown' in s['year_hist']:
            years_str += f" +unknown({s['year_hist']['unknown']})"
        print(f"  {split:<6}  {s['n_images']:>7}  {s['n_annotations']:>8}  {len(s['cat_hist']):>6}  {years_str:<30}")
    print()

    # --- Category histogram per split ---
    print('Category histograms per split:')
    for split in SPLIT_NAMES:
        s = summaries.get(split)
        if not s:
            continue
        print(f'  {split}:')
        print(fmt_hist(s['cat_hist']))
    print()

    # --- Year breakdown per split ---
    print('Year breakdown (images) per split:')
    all_years = set()
    for s in summaries.values():
        all_years.update(s['year_hist'].keys())
    all_years = sorted(all_years)
    header = '  {:<6}'.format('year') + ''.join(f'  {sp:>7}' for sp in SPLIT_NAMES)
    print(header)
    for year in all_years:
        row = f'  {year:<6}'
        for sp in SPLIT_NAMES:
            s = summaries.get(sp, {})
            row += f"  {s.get('year_hist', {}).get(year, 0):>7}"
        print(row)
    print()

    # --- Overlap sanity check ---
    print('Overlap check (image names shared between splits):')
    name_sets = {sp: set(summaries[sp]['image_names']) for sp in SPLIT_NAMES if sp in summaries}
    overlaps_found = False
    pairs = [('train', 'vali'), ('train', 'test'), ('vali', 'test')]
    for a, b in pairs:
        if a in name_sets and b in name_sets:
            shared = name_sets[a] & name_sets[b]
            status = 'OK' if not shared else f'OVERLAP ({len(shared)})'
            print(f'  {a:<5} ∩ {b:<5} : {status}')
            if shared:
                overlaps_found = True
                for s in list(shared)[:5]:
                    print(f'      e.g. {s}')

    if 'all' in summaries:
        union = set().union(*name_sets.values()) if name_sets else set()
        all_names = set(summaries['all']['image_names'])
        only_in_splits = union - all_names
        only_in_all = all_names - union
        print(f'  union(train,vali,test) vs all: '
              f"only_in_splits={len(only_in_splits)}, only_in_all={len(only_in_all)}")
    print()

    # --- splits_v1.json manifest ---
    splits_manifest = unpacked / 'splits_v1.json'
    if splits_manifest.exists():
        try:
            sm = json.loads(splits_manifest.read_text())
            print(f'splits_v1.json: seed={sm.get("seed")}  '
                  f'test_years={sm.get("test_years")}  test_frac={sm.get("test_frac")}  '
                  f'sources={len(sm.get("sources", []))}  '
                  f'assignments={len(sm.get("assignments", []))}')
            split_counts = Counter(a.get('train_vali_split') or a.get('split') for a in sm.get('assignments', []))
            print(f'  assignment split counts: {dict(split_counts)}')
        except Exception as exc:  # noqa: BLE001
            print(f'  (could not parse {splits_manifest}: {exc})')
    else:
        print(f'splits_v1.json: not found at {splits_manifest}')

    return 1 if overlaps_found else 0


if __name__ == '__main__':
    sys.exit(main())
