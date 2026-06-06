#!/usr/bin/env python3
"""
Build splits_v2 sea lion kwcoco bundles from all_norm + splits_v2.yaml.

This is the *project-side* splits tool. It does NOT touch the upstream
corpus build (which lives at /data/Public/VIAME/viame_sealions_2026/
scripts/sealion_pipeline.py); it consumes the upstream's all_norm
kwcoco and applies a hard-coded split definition from
docs/splits_v2.yaml.

Why a separate tool: v1 splits were defined inline in the upstream
pipeline (test_years=2009/2019/2024, 25% contiguous chunk per year).
That definition missed every NFS image in the corpus. v2 fixes class
coverage with hand-picked windows recorded in version control.

Output:
  <out>/{train,vali,test,learn}_norm_v2.kwcoco.zip
  <out>/splits_v2.json     - per-image assignment (full audit trail)
  <out>/splits_v2_report.json - class coverage + size + assertion results

Run:
  python3 scripts/build_splits.py
  python3 scripts/build_splits.py --dry-run
  python3 scripts/build_splits.py --all-norm <path> --manifest <yaml> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import kwcoco
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "splits_v2.yaml"
DEFAULT_ALL_NORM = Path("/data/Public/VIAME/viame_sealions_2026/unpacked/all_norm.kwcoco.zip")
DEFAULT_OUT_DIR = Path("/data/Public/VIAME/viame_sealions_2026/unpacked")

EXPECTED_CATEGORIES = {
    'bull', 'subadult_male', 'female', 'juvenile', 'pup',
    'northern_fur_seal', 'dead_pup', 'dead_nonpup', 'negative',
}


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def sort_images_by_year(dset: kwcoco.CocoDataset) -> dict[str, list[tuple[str, int]]]:
    """Group dataset's images by year, sort by name within each year."""
    by_year = collections.defaultdict(list)
    for img in dset.dataset['images']:
        if 'year' not in img:
            raise SystemExit(f"image {img.get('id')} missing 'year' field")
        by_year[img['year']].append((img['name'], img['id']))
    for y in by_year:
        by_year[y].sort(key=lambda x: x[0])
    return dict(by_year)


def locate_chunk(year_imgs: list, first_name: str, last_name: str,
                 expected_size: int, year: str) -> tuple[int, int]:
    """Find the [first_name..last_name] window in sorted year_imgs.

    Asserts the resulting size equals expected_size; otherwise the
    upstream image set has drifted from when this manifest was
    written. Either fix the manifest or fix the upstream — DO NOT
    silently shift.
    """
    names = [name for name, _ in year_imgs]
    if first_name not in names:
        raise SystemExit(
            f"[{year}] first_image {first_name!r} not in year's images. "
            f"Manifest may be stale or upstream image set changed."
        )
    if last_name not in names:
        raise SystemExit(
            f"[{year}] last_image {last_name!r} not in year's images. "
            f"Manifest may be stale or upstream image set changed."
        )
    i0 = names.index(first_name)
    i1 = names.index(last_name) + 1
    if i1 <= i0:
        raise SystemExit(
            f"[{year}] last_image {last_name!r} (idx {i1-1}) precedes "
            f"first_image {first_name!r} (idx {i0}). Sort order issue?"
        )
    actual = i1 - i0
    if actual != expected_size:
        raise SystemExit(
            f"[{year}] chunk size mismatch: {first_name!r}..{last_name!r} = "
            f"{actual} images, expected {expected_size}. "
            f"Upstream corpus may have added/removed images in this year. "
            f"Inspect and either update splits_v2.yaml or rebuild corpus."
        )
    return i0, i1


def assign_test(dset: kwcoco.CocoDataset, manifest: dict) -> tuple[set, dict]:
    """Apply test_chunks from manifest to produce test gid set + per-chunk audit."""
    by_year = sort_images_by_year(dset)
    test_gids = set()
    chunk_audit = []
    for chunk in manifest['test_chunks']:
        year = chunk['year']
        if year not in by_year:
            raise SystemExit(f"manifest references year {year!r} not in dataset")
        i0, i1 = locate_chunk(
            by_year[year],
            chunk['first_image'], chunk['last_image'],
            chunk['expected_size'], year,
        )
        chunk_gids = [gid for _, gid in by_year[year][i0:i1]]
        test_gids.update(chunk_gids)
        chunk_audit.append({
            'year': year,
            'start_index': i0,
            'end_index': i1,
            'size': i1 - i0,
            'first_image': chunk['first_image'],
            'last_image': chunk['last_image'],
            'gids': chunk_gids,
        })
    return test_gids, chunk_audit


def assign_train_vali(dset: kwcoco.CocoDataset, test_gids: set,
                      manifest: dict) -> tuple[set, set]:
    """Year-stratified random split of (all - test) into train/vali.

    Each year's learn pool is shuffled with a deterministic per-year
    seed (manifest.vali.seed XOR year-string-hash mod 2**32) and the
    first vali.frac fraction goes to vali. Order-independent.
    """
    by_year = sort_images_by_year(dset)
    base_seed = manifest['vali']['seed']
    vali_frac = manifest['vali']['frac']
    train_gids: set = set()
    vali_gids: set = set()
    for year, name_gid_pairs in by_year.items():
        learn_gids = sorted(gid for _, gid in name_gid_pairs if gid not in test_gids)
        if not learn_gids:
            continue
        # Per-year deterministic seed: stable under year iteration order.
        per_year_seed = (base_seed ^ (hash(('year', year)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = random.Random(per_year_seed)
        shuffled = list(learn_gids)
        rng.shuffle(shuffled)
        n_vali = max(1, round(len(shuffled) * vali_frac))
        vali_gids.update(shuffled[:n_vali])
        train_gids.update(shuffled[n_vali:])
    return train_gids, vali_gids


def per_class_image_counts(dset: kwcoco.CocoDataset, gids: set) -> dict[str, int]:
    """For each category, count distinct images in `gids` containing it."""
    cls_imgs = collections.defaultdict(set)
    for ann in dset.dataset['annotations']:
        if ann['image_id'] in gids:
            cname = dset.cats[ann['category_id']]['name']
            cls_imgs[cname].add(ann['image_id'])
    return {c: len(s) for c, s in cls_imgs.items()}


def per_class_ann_counts(dset: kwcoco.CocoDataset, gids: set) -> dict[str, int]:
    cnt: collections.Counter = collections.Counter()
    for ann in dset.dataset['annotations']:
        if ann['image_id'] in gids:
            cname = dset.cats[ann['category_id']]['name']
            cnt[cname] += 1
    return dict(cnt)


def validate_test(dset: kwcoco.CocoDataset, test_gids: set, manifest: dict) -> tuple[list[str], dict]:
    """Check test set against manifest assertions. Return (failures, audit)."""
    failures: list[str] = []
    cls_img = per_class_image_counts(dset, test_gids)
    cls_ann = per_class_ann_counts(dset, test_gids)

    bounds = manifest.get('test_size_bounds', {})
    if 'min_images' in bounds and len(test_gids) < bounds['min_images']:
        failures.append(
            f"test size {len(test_gids)} < min {bounds['min_images']}"
        )
    if 'max_images' in bounds and len(test_gids) > bounds['max_images']:
        failures.append(
            f"test size {len(test_gids)} > max {bounds['max_images']}"
        )

    coverage = manifest.get('test_class_coverage_required') or {}
    if coverage.get('all_9_classes_present'):
        missing = EXPECTED_CATEGORIES - set(cls_img)
        if missing:
            failures.append(f"test missing classes: {sorted(missing)}")
    for cname, min_count in coverage.items():
        if cname == 'all_9_classes_present':
            continue
        got = cls_img.get(cname, 0)
        if got < min_count:
            failures.append(
                f"test class {cname!r} has {got} imgs, manifest requires >= {min_count}"
            )

    return failures, {
        'test_n_images': len(test_gids),
        'test_class_image_counts': cls_img,
        'test_class_ann_counts': cls_ann,
    }


def validate_source(dset: kwcoco.CocoDataset, manifest: dict) -> list[str]:
    failures: list[str] = []
    src = manifest.get('source', {})
    if 'expected_total_images' in src and dset.n_images != src['expected_total_images']:
        failures.append(
            f"source image count mismatch: {dset.n_images} vs "
            f"expected {src['expected_total_images']}"
        )
    if 'expected_total_annotations' in src and dset.n_annots != src['expected_total_annotations']:
        failures.append(
            f"source annotation count mismatch: {dset.n_annots} vs "
            f"expected {src['expected_total_annotations']}"
        )
    return failures


def write_bundle(src: kwcoco.CocoDataset, gids: set, out_path: Path,
                 split_name: str, manifest_summary: dict) -> kwcoco.CocoDataset:
    """Subset src to gids and write as kwcoco.zip."""
    sub = src.subset(sorted(gids))
    sub.dataset['info'] = list(src.dataset.get('info', [])) + [{
        'description': f'splits_v2 {split_name} subset of all_norm',
        'stage': 'splits_v2',
        'split': split_name,
        'manifest': manifest_summary,
        'n_images': sub.n_images,
        'n_annotations': sub.n_annots,
    }]
    sub.fpath = str(out_path)
    sub.dump(str(out_path), newlines=True)
    return sub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all-norm', type=Path, default=DEFAULT_ALL_NORM,
                        help='source all_norm.kwcoco.zip (default: %(default)s)')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='splits_v2.yaml manifest path (default: %(default)s)')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                        help='where to write the v2 bundles (default: %(default)s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='compute splits and validate, but do not write bundles')
    parser.add_argument('--allow-coverage-failure', action='store_true',
                        help='write bundles even if class coverage assertions fail')
    args = parser.parse_args()

    print(f"loading manifest {args.manifest}")
    manifest = load_manifest(args.manifest)

    print(f"loading source {args.all_norm}")
    dset = kwcoco.CocoDataset(args.all_norm)
    print(f"  source: {dset.n_images} images, {dset.n_annots} annotations, "
          f"{dset.n_cats} categories")

    src_failures = validate_source(dset, manifest)
    if src_failures:
        for f in src_failures: print(f"  SOURCE WARNING: {f}", file=sys.stderr)

    print("\nassigning test chunks...")
    test_gids, chunk_audit = assign_test(dset, manifest)
    print(f"  test: {len(test_gids)} images across {len(chunk_audit)} chunks")
    for c in chunk_audit:
        print(f"    {c['year']} [{c['start_index']}:{c['end_index']}] "
              f"size={c['size']}")

    print("\nassigning train/vali (year-stratified)...")
    train_gids, vali_gids = assign_train_vali(dset, test_gids, manifest)
    print(f"  train: {len(train_gids)} images")
    print(f"  vali:  {len(vali_gids)} images")

    # Disjointness assertions
    assert not (test_gids & vali_gids), "test/vali overlap"
    assert not (test_gids & train_gids), "test/train overlap"
    assert not (vali_gids & train_gids), "vali/train overlap"
    assert (test_gids | vali_gids | train_gids) == set(dset.imgs), \
        "splits don't partition all images"

    print("\nvalidating test class coverage...")
    failures, test_audit = validate_test(dset, test_gids, manifest)
    print(f"  test class img counts: {test_audit['test_class_image_counts']}")
    if failures:
        print("\n!!! COVERAGE FAILURES !!!", file=sys.stderr)
        for f in failures: print(f"  {f}", file=sys.stderr)
        if not args.allow_coverage_failure:
            print("Aborting. Use --allow-coverage-failure to override.", file=sys.stderr)
            return 1
    else:
        print("  all coverage assertions pass ✓")

    # Per-split per-class summary
    summary = {
        'manifest_version': manifest['version'],
        'source': {
            'all_norm_path': str(args.all_norm),
            'n_images': dset.n_images,
            'n_annotations': dset.n_annots,
            'n_categories': dset.n_cats,
        },
        'chunk_audit': [
            {k: v for k, v in c.items() if k != 'gids'} for c in chunk_audit
        ],
        'splits': {},
        'test_class_image_counts': test_audit['test_class_image_counts'],
        'test_class_ann_counts': test_audit['test_class_ann_counts'],
        'coverage_failures': failures,
    }
    for sname, sgids in [('train', train_gids), ('vali', vali_gids), ('test', test_gids)]:
        summary['splits'][sname] = {
            'n_images': len(sgids),
            'class_image_counts': per_class_image_counts(dset, sgids),
            'class_ann_counts': per_class_ann_counts(dset, sgids),
        }

    print("\nper-split per-class image counts:")
    cats = sorted(EXPECTED_CATEGORIES)
    print(f"  {'class':<18} " + " ".join(f"{s:>7}" for s in ('train','vali','test')))
    for c in cats:
        row = [c]
        for sname in ('train','vali','test'):
            row.append(str(summary['splits'][sname]['class_image_counts'].get(c, 0)))
        print(f"  {row[0]:<18} {row[1]:>7} {row[2]:>7} {row[3]:>7}")

    if args.dry_run:
        print("\nDRY RUN — no bundles written.")
        # Still write report so the user can audit
        report_path = args.out_dir / 'splits_v2_report.dryrun.json'
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"  wrote dry-run report to {report_path}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_summary = {
        'manifest_path': str(args.manifest),
        'manifest_version': manifest['version'],
        'vali_seed': manifest['vali']['seed'],
        'vali_frac': manifest['vali']['frac'],
    }
    learn_gids = train_gids | vali_gids
    print("\nwriting bundles...")
    for sname, sgids in [
        ('train', train_gids),
        ('vali',  vali_gids),
        ('test',  test_gids),
        ('learn', learn_gids),
    ]:
        out = args.out_dir / f'{sname}_norm_v2.kwcoco.zip'
        sub = write_bundle(dset, sgids, out, sname, manifest_summary)
        print(f"  {out}: {sub.n_images} imgs, {sub.n_annots} annots")

    # Per-image audit trail
    gid_to_split = {}
    for gid in train_gids: gid_to_split[gid] = 'train'
    for gid in vali_gids:  gid_to_split[gid] = 'vali'
    for gid in test_gids:  gid_to_split[gid] = 'test'
    splits_json = {
        'version': 2,
        'manifest': manifest_summary,
        'assignments': [
            {
                'image_name': img['name'],
                'year': img['year'],
                'split': gid_to_split[img['id']],
            }
            for img in dset.dataset['images']
        ],
    }
    (args.out_dir / 'splits_v2.json').write_text(json.dumps(splits_json, indent=2))
    print(f"  {args.out_dir/'splits_v2.json'}: per-image audit trail")

    (args.out_dir / 'splits_v2_report.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print(f"  {args.out_dir/'splits_v2_report.json'}: build report")

    return 0


if __name__ == '__main__':
    sys.exit(main())
