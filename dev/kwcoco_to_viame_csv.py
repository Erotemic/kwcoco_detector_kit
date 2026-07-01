#!/usr/bin/env python3
"""
Convert a kwcoco ground-truth bundle into a matched (image_list.txt, truth.csv)
pair for scoring a detector in VIAME.

Why a pair
----------
VIAME's ``score_results.py`` aligns the computed detections against truth by
**frame id** (column 3 of the VIAME CSV) — i.e. the image's position in the list
fed to the pipeline, NOT its filename. So the truth CSV's frame ids must match
the exact order VIAME numbers frames. The only robust way to guarantee that is
to emit the image list and the truth CSV together, from the same ordering:

  1. run VIAME on ``image_list.txt``               -> computed_detections.csv
  2. score computed_detections.csv vs ``truth.csv`` (frame ids line up by
     construction).

Ground-truth boxes are in whole-image pixel coordinates; run the detector tiled
(VIAME ``ocv_windowed`` wrapper) so its stitched detections are in the same
whole-image frame. See docs/planning/viame_integration_runbook.md.

Usage
-----
    python dev/kwcoco_to_viame_csv.py \
        --kwcoco .../scheme_applied/test.kwcoco.zip \
        --out-dir /data/users/jon.crall/kcd_sealion/viame_score/pup_gen007 \
        --limit 25 --require-annots \
        --drop-classes northern_fur_seal
"""
from __future__ import annotations

from pathlib import Path

import kwconf


class Kwcoco2ViameConfig(kwconf.Config):
    kwcoco = kwconf.Value(None, position=1, required=True, help="ground-truth kwcoco path")
    out_dir = kwconf.Value(None, required=True, help="output dir for image_list.txt + truth.csv")
    limit = kwconf.Value(None, parser=int, help="use only the first N (selected) images")
    require_annots = kwconf.Value(False, isflag=True,
                                  help="include only images with >=1 (kept) annotation")
    drop_classes = kwconf.Value("", help="comma-separated category names to drop from GT "
                                         "(e.g. northern_fur_seal — NFS always counts as negative)")
    image_root = kwconf.Value(None, help="optional path prefix to remap image file paths into "
                                         "(e.g. the mount point seen inside the VIAME container)")


# VIAME CSV column contract (see any detector_*.pipe header):
#   1 track/det id, 2 image name, 3 frame id, 4-7 ltrb, 8 confidence,
#   9 target length (-1 = n/a), 10.. repeated (class, confidence) pairs
_HEADER = (
    "# 1: Detection or Track-id, 2: Video or Image Identifier, "
    "3: Unique Frame Identifier, 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y), "
    "8: Detection or Length Confidence, 9: Target Length (0 or -1 if invalid), "
    "10-11+: Repeated Species, Confidence Pairs or Attributes\n"
    "# metadata, exported_by: kwcoco_to_viame_csv.py\n"
)


def _remap(fpath: str, image_root) -> str:
    if not image_root:
        return fpath
    # Replace everything up to the shared suffix with image_root by matching the
    # basename directory tail is fragile; simplest useful behavior: swap the
    # leading dir with image_root, keeping the basename. Callers who need exact
    # container paths should bind-mount the original path instead (recommended).
    return str(Path(image_root) / Path(fpath).name)


def main(argv=None) -> int:
    config = Kwcoco2ViameConfig.cli(argv=argv)
    import kwcoco

    out_dir = Path(config.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = {c.strip() for c in str(config.drop_classes).split(",") if c.strip()}

    dset = kwcoco.CocoDataset(config.kwcoco)
    id2name = {cid: c["name"] for cid, c in dset.cats.items()}

    # Deterministic image order (sorted by gid) so a rerun reproduces frame ids.
    gids = sorted(dset.images())

    def kept_aids(gid):
        return [a for a in dset.gid_to_aids[gid]
                if id2name.get(dset.anns[a]["category_id"]) not in drop]

    if config.require_annots:
        gids = [g for g in gids if kept_aids(g)]
    if config.limit is not None:
        gids = gids[: int(config.limit)]

    list_fpath = out_dir / "image_list.txt"
    truth_fpath = out_dir / "truth.csv"

    n_img = n_box = n_unresolved = 0
    with open(list_fpath, "w") as lf, open(truth_fpath, "w") as tf:
        tf.write(_HEADER)
        det_id = 0
        for frame_id, gid in enumerate(gids):
            fpath = dset.get_image_fpath(gid)
            listed = _remap(str(fpath), config.image_root)
            if not Path(fpath).exists():
                n_unresolved += 1
            lf.write(listed + "\n")
            n_img += 1
            for aid in kept_aids(gid):
                ann = dset.anns[aid]
                x, y, w, h = ann["bbox"]
                name = id2name[ann["category_id"]]
                # ltrb, confidence 1.0 (GT), target length -1, (class, 1.0)
                tf.write(
                    f"{det_id},{listed},{frame_id},"
                    f"{x:.2f},{y:.2f},{x + w:.2f},{y + h:.2f},"
                    f"1.0,-1,{name},1.0\n"
                )
                det_id += 1
                n_box += 1

    print(f"[kwcoco->viame] images     : {n_img}  -> {list_fpath}")
    print(f"[kwcoco->viame] gt boxes   : {n_box}  -> {truth_fpath}")
    if drop:
        print(f"[kwcoco->viame] dropped classes: {sorted(drop)}")
    if n_unresolved:
        print(f"[kwcoco->viame] WARNING: {n_unresolved} listed image path(s) do not exist on "
              "this host — bind-mount them into the VIAME container at the SAME path, or use "
              "--image-root.")
    print("\nNext (inside the VIAME container, after sourcing setup_viame.sh):")
    print(f"  viame <tiled scoring pipe> -s input:video_filename={list_fpath} \\")
    print("       -s detector_writer:file_name=computed_detections.csv")
    print("  python $VIAME_INSTALL/configs/score_results.py \\")
    print(f"       -computed computed_detections.csv -truth {truth_fpath} \\")
    print("       --ignore-classes -iou-thresh 0.5 -det-prc-conf score_out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
