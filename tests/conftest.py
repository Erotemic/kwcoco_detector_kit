"""
Shared pytest fixtures.

Conventions:

- `synthetic_kwcoco` is a hand-built fixture (1 category, a handful of
  images with small bboxes) that does NOT require kwcoco.demo() — that
  upstream API has had compat churn and the fixture must be stable across
  kwcoco 0.7 / 0.8.

- `tmp_workdir` returns a per-test workdir under pytest's `tmp_path`.

- Independent of the prior project's conftest; follows the same general
  shape of "hand-built synthetic fixture so we don't depend on
  kwcoco.demo()'s churn across kwcoco 0.7 / 0.8."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    return workdir


def _make_synthetic_bundle(
    bundle_dpath: Path,
    *,
    num_images: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    boxes_per_image: int = 1,
    category_names=("widget",),
    seed: int = 0,
) -> Path:
    """Build a small kwcoco bundle on disk with synthetic JPEG assets.

    Returns the path of the written .kwcoco.zip. When multiple category
    names are given, boxes are round-robin-assigned to each category so
    every class has annotations.
    """
    import kwcoco
    import kwimage

    if isinstance(category_names, str):
        category_names = (category_names,)
    category_names = tuple(category_names)
    assert category_names, "need at least one category name"

    rng = np.random.RandomState(seed)
    asset_dpath = bundle_dpath / "synth_assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "synth.kwcoco.zip")
    cids = [dset.add_category(name=name) for name in category_names]

    H, W = image_size
    box_counter = 0
    for k in range(num_images):
        img = (rng.rand(H, W, 3) * 255).astype(np.uint8)
        boxes = []
        for _ in range(boxes_per_image):
            bw = rng.randint(20, W // 4)
            bh = rng.randint(20, H // 4)
            bx = int(rng.randint(0, W - bw))
            by = int(rng.randint(0, H - bh))
            cid = cids[box_counter % len(cids)]
            boxes.append((bx, by, bw, bh, cid))
            box_counter += 1
            # Burn an obvious bright square into the image so a real
            # detector has signal.
            img[by:by + bh, bx:bx + bw] = (255, 50, 50)

        fname = f"synth_{k:04d}.jpg"
        fpath = asset_dpath / fname
        kwimage.imwrite(str(fpath), img)

        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=W, height=H, name=f"synth_{k:04d}",
        )
        for (bx, by, bw, bh, cid) in boxes:
            dset.add_annotation(
                image_id=gid, category_id=cid,
                bbox=[float(bx), float(by), float(bw), float(bh)],
                area=float(bw * bh), iscrowd=0,
            )

    dset.dump()
    return Path(dset.fpath)


@pytest.fixture
def synthetic_kwcoco_factory(tmp_path: Path):
    """Returns a factory that builds named synthetic kwcoco bundles.

    Each call writes a fresh bundle under tmp_path/<name>/ so callers can
    have multiple bundles (e.g. train + vali + test) within one test.
    """
    def _make(name: str, **kwargs) -> Path:
        bundle_dpath = tmp_path / name
        bundle_dpath.mkdir(parents=True, exist_ok=True)
        return _make_synthetic_bundle(bundle_dpath, **kwargs)
    return _make


@pytest.fixture
def synthetic_kwcoco(synthetic_kwcoco_factory) -> Path:
    """Default 4-image / 1-category bundle for tile + merge tests."""
    return synthetic_kwcoco_factory("default", num_images=4)
