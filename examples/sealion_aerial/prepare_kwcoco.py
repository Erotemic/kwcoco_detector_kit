#!/usr/bin/env python3
"""
Convert the NOAA Steller Sea Lion dataset (Kaggle 2017) into a kwcoco
bundle. The Kaggle distribution gives us raw images, dotted overlays
where each animal is marked with a colored circular blob, and a per-
image count CSV.

This script:

1. Reads each (raw, dotted) pair and computes the per-pixel diff
   to recover the dot locations.
2. Classifies each dot by color → one of 5 NOAA classes (red, magenta,
   brown, blue, green). With ``--single_class``, collapses to a single
   ``sealion`` category.
3. Emits a 32×32 fixed-size bbox centered on each dot.
4. Drops files listed in MismatchedTrainImages.txt (count mismatches).
5. Writes a kwcoco bundle with the original images linked into
   ``<dst-stem>_assets/`` so the bundle is self-contained.

The 32-px bbox size is conventional for sealion detection (the sea
lions in the NOAA imagery are typically 20–60 pixels across at full
resolution). Tune via ``--bbox_size``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import scriptconfig as scfg


# NOAA class -> approximate (R, G, B) of the marker dot.
# These are the canonical Kaggle-challenge colors.
NOAA_COLOR_CLASSES: Dict[str, Tuple[int, int, int]] = {
    "adult_males":    (244, 8,   8),    # red
    "subadult_males": (244, 8,   244),  # magenta
    "adult_females":  (84,  42,  0),    # brown
    "juveniles":      (33,  20,  244),  # blue
    "pups":           (33,  244, 20),   # green
}


class PrepareConfig(scfg.DataConfig):
    """NOAA Steller Sea Lion -> kwcoco."""

    train_dpath = scfg.Value(None, required=True, help="Train/ directory of raw .jpg images")
    dotted_dpath = scfg.Value(None, required=True, help="TrainDotted/ directory of dotted .jpg images")
    counts_csv = scfg.Value(None, required=True, help="Train.csv with per-image counts")
    mismatched = scfg.Value(None, help="MismatchedTrainImages.txt — image_ids to drop")
    dst = scfg.Value(None, required=True, help="output kwcoco bundle path")

    bbox_size = scfg.Value(32, help="fixed bbox side length (px) centered on each dot")
    color_tol = scfg.Value(40, help="L-inf color tolerance when matching dot color to a class")
    single_class = scfg.Value(True, isflag=True, help="collapse the 5 classes into one 'sealion' category")
    category_name = scfg.Value("sealion", help="category name when --single_class")
    asset_strategy = scfg.Value(
        "symlink", choices=["symlink", "copy"],
        help="how to materialise asset files under the bundle",
    )
    limit = scfg.Value(0, help="limit to N images (0 = all). Useful for smoke tests.")
    progress = scfg.Value(True, isflag=True)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _load_mismatched(fpath: Optional[Path]) -> set:
    if not fpath:
        return set()
    fpath = Path(str(fpath))
    if not fpath.exists():
        return set()
    return {line.strip() for line in fpath.read_text().splitlines() if line.strip()}


def _classify_dot_color(rgb: Tuple[int, int, int], tol: int) -> Optional[str]:
    """Return the class name whose canonical color is within L_inf tol of rgb."""
    best_name = None
    best_dist = tol + 1
    for name, (r, g, b) in NOAA_COLOR_CLASSES.items():
        d = max(abs(int(rgb[0]) - r), abs(int(rgb[1]) - g), abs(int(rgb[2]) - b))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= tol else None


def _find_dots(raw_img, dotted_img, color_tol: int) -> List[Tuple[int, int, str]]:
    """Locate (x, y, class_name) for each dot via raw-vs-dotted diff.

    Returns a list of (x_center, y_center, class_name) tuples. Skips
    diffs that don't match any canonical NOAA color.
    """
    import numpy as np
    from skimage import measure

    if raw_img.shape != dotted_img.shape:
        return []
    diff = (dotted_img.astype(np.int16) - raw_img.astype(np.int16))
    # A dot has high diff magnitude in at least one channel.
    diff_mag = np.max(np.abs(diff), axis=-1)
    mask = diff_mag > 30
    labels = measure.label(mask, connectivity=2)
    out: List[Tuple[int, int, str]] = []
    for region in measure.regionprops(labels):
        if region.area < 4 or region.area > 80:
            continue
        cy, cx = region.centroid
        # Sample the dotted-image color at the centroid.
        ry, rx = int(cy), int(cx)
        rgb = tuple(int(c) for c in dotted_img[ry, rx])
        klass = _classify_dot_color(rgb, color_tol)
        if klass is None:
            continue
        out.append((int(round(cx)), int(round(cy)), klass))
    return out


def run(config):
    import kwcoco
    import kwimage
    import ubelt as ub

    train_dpath = Path(str(config.train_dpath)).expanduser().resolve()
    dotted_dpath = Path(str(config.dotted_dpath)).expanduser().resolve()
    if not train_dpath.is_dir():
        raise FileNotFoundError(train_dpath)
    if not dotted_dpath.is_dir():
        raise FileNotFoundError(dotted_dpath)

    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    asset_dpath = dst_fpath.parent / (dst_fpath.stem.replace(".kwcoco", "") + "_assets")
    asset_dpath.mkdir(parents=True, exist_ok=True)

    mismatched = _load_mismatched(Path(str(config.mismatched)) if config.mismatched else None)

    dset = kwcoco.CocoDataset()
    dset.fpath = str(dst_fpath)
    if bool(config.single_class):
        cat_ids = {name: dset.add_category(name=str(config.category_name))
                   for name in NOAA_COLOR_CLASSES}
    else:
        cat_ids = {name: dset.add_category(name=name) for name in NOAA_COLOR_CLASSES}

    image_files = sorted(p for p in train_dpath.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if int(config.limit) > 0:
        image_files = image_files[: int(config.limit)]

    bbox_half = int(config.bbox_size) // 2
    iterator = ub.ProgIter(image_files, desc="sealion -> kwcoco", enabled=bool(config.progress))
    n_kept = 0
    n_drops = 0
    n_anns = 0
    for raw_fpath in iterator:
        stem = raw_fpath.stem
        if stem in mismatched:
            n_drops += 1
            continue
        dotted_fpath = dotted_dpath / raw_fpath.name
        if not dotted_fpath.exists():
            n_drops += 1
            continue
        try:
            raw_img = kwimage.imread(str(raw_fpath))
            dotted_img = kwimage.imread(str(dotted_fpath))
        except Exception as ex:
            print(f"  warn: failed to read {stem}: {ex}")
            n_drops += 1
            continue
        H, W = raw_img.shape[:2]
        dots = _find_dots(raw_img, dotted_img, int(config.color_tol))

        # Materialize the asset file (symlink by default).
        asset_fname = raw_fpath.name
        asset_fpath = asset_dpath / asset_fname
        if not asset_fpath.exists():
            if str(config.asset_strategy) == "copy":
                shutil.copy2(raw_fpath, asset_fpath)
            else:
                asset_fpath.symlink_to(raw_fpath)

        gid = dset.add_image(
            file_name=str(asset_fpath.relative_to(dst_fpath.parent)),
            width=int(W), height=int(H), name=stem,
        )
        for (cx, cy, klass) in dots:
            x = max(0, cx - bbox_half)
            y = max(0, cy - bbox_half)
            bw = min(int(config.bbox_size), W - x)
            bh = min(int(config.bbox_size), H - y)
            dset.add_annotation(
                image_id=gid, category_id=cat_ids[klass],
                bbox=[float(x), float(y), float(bw), float(bh)],
                area=float(bw * bh), iscrowd=0,
            )
            n_anns += 1
        n_kept += 1

    dset.dump()
    print(
        f"wrote {dset.fpath}: {n_kept} images kept, {n_drops} dropped, "
        f"{n_anns} annotations"
    )


__cli__ = PrepareConfig


if __name__ == "__main__":
    __cli__.main()
