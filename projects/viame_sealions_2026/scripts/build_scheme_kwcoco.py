#!/usr/bin/env python3
"""
Remap the all_collapsed kwcoco splits to a target class scheme.

The existing training_ready_v1/{train,vali,test}.kwcoco.zip bundles
collapse every positive class to "sealion" but preserve the original
class name on each annotation in the ``source_category`` field
(e.g. 'B', 'F', 'J', 'P', 'NFS', ...). This script reads a scheme from
docs/class_schemes.yaml, applies its source -> target name mapping to
each annotation's ``source_category``, drops annotations whose source
class is in the scheme's ``drop`` list, and writes new kwcoco files
under ``training_ready_v1/by_scheme/<scheme>/``.

The output kwcoco bundles use 1-indexed integer category IDs in the
order the scheme declared the target classes (see
``docs/class_schemes.yaml`` — kwcoco_detector_kit's tile/MSCOCO
pipeline preserves that order, and the trained model's class index
matches it).

Usage:

    python3 scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup
    python3 scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup --dry-run
    python3 scripts/build_scheme_kwcoco.py --scheme lifestage_6cls \\
        --src-dir training_ready_v1 --out-dir training_ready_v1/by_scheme/lifestage_6cls
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import yaml


DEFAULT_REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMES = DEFAULT_REPO / "docs" / "class_schemes.yaml"
DEFAULT_SRC = DEFAULT_REPO / "training_ready_v1"
SPLITS = ("train", "vali", "test")


def load_scheme(schemes_fpath: Path, name: str) -> dict:
    data = yaml.safe_load(schemes_fpath.read_text()) or {}
    schemes = data.get("schemes") or {}
    if name not in schemes:
        raise SystemExit(
            f"error: scheme {name!r} not in {schemes_fpath}; known: {sorted(schemes)}"
        )
    scheme = schemes[name]
    scheme.setdefault("mapping", {})
    scheme.setdefault("drop", [])
    return scheme


def ordered_target_names(scheme: dict) -> list[str]:
    """Return target class names in their canonical order.

    Prefers ``target_order`` (explicit list). Falls back to first-seen
    iteration order over ``mapping.values()`` if the scheme is missing
    ``target_order`` — useful for older schemes, but fragile against
    YAML key-order shuffling so new schemes should always set it.
    """
    explicit = scheme.get("target_order")
    if explicit:
        order = list(explicit)
        mapped_targets = set(scheme["mapping"].values())
        missing = mapped_targets - set(order)
        if missing:
            raise ValueError(
                f"scheme target_order={order!r} is missing classes "
                f"referenced in mapping: {sorted(missing)}"
            )
        return order
    seen, out = set(), []
    for target in scheme["mapping"].values():
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


def remap_split(
    src_fpath: Path,
    dst_fpath: Path,
    scheme: dict,
    *,
    dry_run: bool = False,
) -> dict:
    """Remap one kwcoco split per the scheme. Returns a stats dict."""
    import kwcoco

    src = kwcoco.CocoDataset(str(src_fpath))
    target_names = ordered_target_names(scheme)
    name_to_new_cid = {name: i + 1 for i, name in enumerate(target_names)}
    mapping = dict(scheme["mapping"])
    drop = set(scheme.get("drop", []))

    dst = kwcoco.CocoDataset()
    dst.dataset["info"] = [{
        "stage": "scheme_remap",
        "scheme": scheme.get("description", ""),
        "source_kwcoco": str(src_fpath),
        "mapping": mapping,
        "drop": sorted(drop),
        "target_classes": target_names,
    }]
    for name in target_names:
        dst.add_category(name=name, id=name_to_new_cid[name])

    # Copy images, preserving IDs so debugging across the source/scheme
    # bundles stays simple. Rewrite file_name to the absolute on-disk
    # path because the scheme bundles live one level deeper than the
    # source (training_ready_v1/by_scheme/<scheme>/), so any relative
    # file_name in the source would resolve to a nonexistent directory.
    for img in src.dataset.get("images", []):
        new_img = {k: v for k, v in img.items()}
        try:
            abs_fpath = src.get_image_fpath(img["id"])
            new_img["file_name"] = str(abs_fpath)
        except Exception:
            pass
        dst.add_image(**new_img)

    stats = collections.Counter()
    dropped = collections.Counter()
    kept_per_target = collections.Counter()
    n_unknown = 0
    for ann in src.dataset.get("annotations", []):
        src_cat = ann.get("source_category")
        if src_cat in drop:
            dropped[src_cat] += 1
            continue
        if src_cat not in mapping:
            # Not in mapping AND not in drop = silently dropped, but
            # count for visibility.
            n_unknown += 1
            stats["_unknown"] += 1
            continue
        new_ann = {k: v for k, v in ann.items() if k != "id"}
        new_ann["category_id"] = name_to_new_cid[mapping[src_cat]]
        # Keep source_category for traceability.
        new_ann["source_category"] = src_cat
        dst.add_annotation(**new_ann)
        stats[mapping[src_cat]] += 1
        kept_per_target[mapping[src_cat]] += 1

    stats["_dropped_total"] = sum(dropped.values())
    stats["_dropped_by_source"] = dict(dropped)
    stats["_unknown_source_categories"] = n_unknown

    if not dry_run:
        dst_fpath.parent.mkdir(parents=True, exist_ok=True)
        dst.fpath = str(dst_fpath)
        dst.dump(str(dst_fpath), newlines=True)

    return {
        "src": str(src_fpath),
        "dst": str(dst_fpath),
        "n_images": dst.n_images,
        "n_annotations": dst.n_annots,
        "per_target_class": dict(kept_per_target),
        "dropped_by_source": dict(dropped),
        "n_unknown_source_categories": n_unknown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme", required=True,
                        help="scheme name from docs/class_schemes.yaml")
    parser.add_argument("--schemes-file", type=Path, default=DEFAULT_SCHEMES,
                        help=f"path to schemes YAML (default: {DEFAULT_SCHEMES})")
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC,
                        help=f"dir with train/vali/test.kwcoco.zip (default: {DEFAULT_SRC})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default: <src-dir>/by_scheme/<scheme>)")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS),
                        choices=list(SPLITS),
                        help="which splits to remap")
    parser.add_argument("--dry-run", action="store_true",
                        help="report stats but don't write outputs")
    args = parser.parse_args()

    if not args.schemes_file.exists():
        raise SystemExit(f"error: schemes file does not exist: {args.schemes_file}")
    scheme = load_scheme(args.schemes_file, args.scheme)

    out_dir = args.out_dir or (args.src_dir / "by_scheme" / args.scheme)
    print(f"scheme:    {args.scheme}")
    print(f"src_dir:   {args.src_dir}")
    print(f"out_dir:   {out_dir}")
    print(f"targets:   {ordered_target_names(scheme)}")
    print(f"drop:      {sorted(scheme.get('drop', []))}")
    print(f"dry_run:   {args.dry_run}")
    print()

    report = {
        "scheme": args.scheme,
        "src_dir": str(args.src_dir),
        "out_dir": str(out_dir),
        "target_classes": ordered_target_names(scheme),
        "splits": {},
    }
    for split in args.splits:
        src_fpath = args.src_dir / f"{split}.kwcoco.zip"
        dst_fpath = out_dir / f"{split}.kwcoco.zip"
        if not src_fpath.exists():
            print(f"  [skip] {split}: source missing ({src_fpath})", file=sys.stderr)
            continue
        result = remap_split(src_fpath, dst_fpath, scheme, dry_run=args.dry_run)
        report["splits"][split] = result
        print(f"  [{split}] images={result['n_images']:>5} "
              f"anns={result['n_annotations']:>7}  "
              f"per_class={result['per_target_class']}")
        if result["dropped_by_source"]:
            print(f"    dropped: {result['dropped_by_source']}")
        if result["n_unknown_source_categories"]:
            print(f"    WARNING: {result['n_unknown_source_categories']} annotations had "
                  f"a source_category not in mapping and not in drop")

    if not args.dry_run:
        report_fpath = out_dir / "scheme_report.json"
        report_fpath.parent.mkdir(parents=True, exist_ok=True)
        report_fpath.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {report_fpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
