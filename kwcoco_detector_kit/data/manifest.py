"""
Data manifest — record the *true* contents of a kwcoco bundle and (optionally)
assert they match what a recipe declared.

Motivation (KCD-DATA-01). The single most expensive shitspotter bug was training
on ``simplified_train_imgs7350_*.kwcoco.zip`` — a file whose name said 7350
images but which actually held 2564. Three experiment generations (v6/v7/v8)
silently trained on a quarter of the data. A filename is not a contract; a
manifest is. This op records true image/annotation/category counts plus a stable
content hash, and ``recipe-run`` can assert a recipe's declared
``data.expect:`` block against it BEFORE any GPU time.

CLI::

    kwcoco-detector-kit data-manifest path/to/train.kwcoco.zip
    kwcoco-detector-kit data-manifest path/to/train.kwcoco.zip --out manifest.json

Programmatic::

    from kwcoco_detector_kit.data.manifest import compute_manifest, assert_expected
    man = compute_manifest("train.kwcoco.zip")
    assert_expected(man, {"n_images": 10671}, source="recipe.data.expect")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import scriptconfig as scfg


def compute_manifest(kwcoco_path) -> Dict[str, Any]:
    """Return a dict of the true contents of a kwcoco bundle.

    Keys: ``path``, ``exists``, ``n_images``, ``n_annots``, ``n_categories``,
    ``categories`` (sorted names), ``per_category`` (name -> annot count), and
    ``content_hash`` (sha256 over the canonical counts — stable across
    re-saves, sensitive to the things that actually matter for training).
    """
    path = Path(str(kwcoco_path)).expanduser()
    man: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return man

    import kwcoco

    dset = kwcoco.CocoDataset.coerce(str(path))
    cats = dset.dataset.get("categories", [])
    cat_names = sorted([c.get("name") for c in cats if c.get("name")])
    id_to_name = {c["id"]: c.get("name") for c in cats}

    per_category: Dict[str, int] = {name: 0 for name in cat_names}
    for ann in dset.dataset.get("annotations", []):
        name = id_to_name.get(ann.get("category_id"))
        if name is not None:
            per_category[name] = per_category.get(name, 0) + 1

    man.update({
        "n_images": int(dset.n_images),
        "n_annots": int(dset.n_annots),
        "n_categories": len(cat_names),
        "categories": cat_names,
        "per_category": per_category,
    })
    # Canonical fingerprint over the things that define the training problem.
    canonical = json.dumps(
        {k: man[k] for k in ("n_images", "n_annots", "categories", "per_category")},
        sort_keys=True,
    )
    man["content_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return man


def assert_expected(manifest: Mapping[str, Any], expect: Mapping[str, Any],
                    *, source: str = "data.expect", strict: bool = True) -> list:
    """Compare a manifest against an ``expect`` mapping.

    Recognized ``expect`` keys: ``n_images``, ``n_annots``, ``n_categories``,
    ``categories`` (exact sorted-set match), ``content_hash``. Returns a list of
    human-readable mismatch strings. Raises ``ValueError`` when ``strict`` and
    there is at least one mismatch (the loud-failure default).
    """
    mismatches = []
    for key in ("n_images", "n_annots", "n_categories", "content_hash"):
        if key in expect and manifest.get(key) != expect[key]:
            mismatches.append(
                f"{key}: expected {expect[key]!r}, manifest has {manifest.get(key)!r}")
    if "categories" in expect:
        want = sorted(expect["categories"])
        got = list(manifest.get("categories", []))
        if want != got:
            mismatches.append(f"categories: expected {want!r}, manifest has {got!r}")
    if mismatches and strict:
        bullet = "\n  - ".join(mismatches)
        raise ValueError(
            f"{source}: kwcoco manifest does not match the declared expectation "
            f"for {manifest.get('path')!r}:\n  - {bullet}\n"
            "A filename is not a contract. Fix the path/bundle, or update the "
            "`expect:` block if the change is intended."
        )
    return mismatches


class DataManifestConfig(scfg.DataConfig):
    """Record true image/annotation/category counts + a content hash for a kwcoco bundle."""

    src = scfg.Value(None, position=1, required=True,
                     help="kwcoco bundle to inspect")
    out = scfg.Value(None, help="optional path to write the manifest JSON")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def run(config) -> Dict[str, Any]:
    man = compute_manifest(config.src)
    text = json.dumps(man, indent=2)
    if config.out:
        Path(str(config.out)).expanduser().write_text(text)
        print(f"wrote manifest -> {config.out}")
    print(text)
    if not man.get("exists"):
        print(f"WARNING: {config.src} does not exist")
    return man


__cli__ = DataManifestConfig
