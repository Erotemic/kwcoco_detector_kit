"""Export kwcoco prediction datasets to LabelMe JSON sidecar files.

LabelMe sidecars sit next to each source image and are the standard way to
load/save annotations in the LabelMe annotation tool.  This module converts
a kwcoco prediction dataset (written by :func:`predict_kwcoco`) into one
``.json`` sidecar per image, ready for human review and correction.

Only polygon/segmentation annotations are exported — pure bounding-box
annotations are skipped because LabelMe's native format requires polygon
shapes.  Call :func:`export_to_labelme` after running prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _stage_images(pred_dset, export_dset, copy_dst: Path) -> Path:
    """Copy source images into ``copy_dst`` with stable, collision-free names.

    Image ``file_name`` fields in ``export_dset`` are updated in-place to
    point to the copied files so subsequent LabelMe file generation uses the
    correct relative paths.
    """
    import shutil

    copy_dst = copy_dst.expanduser().resolve()
    if copy_dst.exists() and any(copy_dst.iterdir()):
        raise FileExistsError(
            f"copy_dst {copy_dst} already contains files; "
            "provide an empty or nonexistent directory"
        )
    copy_dst.mkdir(parents=True, exist_ok=True)

    for img in export_dset.images().objs:
        src_fpath = Path(pred_dset.get_image_fpath(img["id"])).expanduser().resolve()
        staged_name = f"{int(img['id']):08d}_{src_fpath.name}"
        staged_fpath = copy_dst / staged_name
        shutil.copy2(src_fpath, staged_fpath)
        img["file_name"] = str(staged_fpath)

    export_dset.reroot(absolute=True)
    return copy_dst


def export_to_labelme(
    pred_kwcoco,
    *,
    score_thresh: float = 0.0,
    only_missing: bool = True,
    copy_dst: Optional[str | Path] = None,
) -> list[Path]:
    """Write LabelMe JSON sidecars for a kwcoco prediction dataset.

    Each image that has at least one polygon annotation above ``score_thresh``
    gets a ``.json`` sidecar written next to its image file.

    Args:
        pred_kwcoco: Path to a kwcoco prediction dataset, or a
            ``kwcoco.CocoDataset`` instance.
        score_thresh: Skip annotations below this detection score.
        only_missing: When ``True`` (default), skip images that already have
            a sidecar on disk — useful for incremental export without
            overwriting manual corrections.
        copy_dst: Optional directory.  When given, source images are copied
            there with id-prefixed names before writing sidecars, keeping
            the annotation review artefacts self-contained.

    Returns:
        Sorted list of written sidecar :class:`pathlib.Path` objects.
    """
    import kwcoco
    import kwimage
    from kwcoco.formats.labelme import LabelMeFile

    pred_dset = kwcoco.CocoDataset.coerce(pred_kwcoco)

    # Build a clean copy containing only export-eligible annotations.
    export_dset = pred_dset.copy()
    export_dset.clear_annotations()

    for coco_img in pred_dset.images().coco_images:
        for ann in coco_img.annots().objs:
            if float(ann.get("score", 1.0)) < score_thresh:
                continue
            seg = ann.get("segmentation")
            if seg is None:
                continue
            try:
                mpoly = kwimage.Segmentation.coerce(seg).to_multi_polygon()
                mpoly = mpoly.simplify(1.0)
            except Exception:
                continue
            if not len(mpoly.data):
                continue
            catname = pred_dset.cats[ann["category_id"]]["name"]
            export_dset.add_annotation(
                image_id=coco_img.img["id"],
                category_id=export_dset.ensure_category(catname),
                bbox=list(mpoly.box().to_coco()),
                segmentation=mpoly.to_coco(style="new"),
                score=float(ann.get("score", 1.0)),
                role=str(ann.get("role", "prediction")),
            )

    if copy_dst is not None:
        _stage_images(pred_dset, export_dset, Path(copy_dst))

    sidecars = list(LabelMeFile.multiple_from_coco(export_dset))
    written: list[Path] = []
    for sidecar in sidecars:
        sidecar.reroot(absolute=False)
        sidecar.fpath = sidecar.fpath.resolve()
        if not sidecar.data["shapes"]:
            continue
        if only_missing and sidecar.fpath.exists():
            continue
        sidecar.dump()
        written.append(sidecar.fpath)
    return sorted(written)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli():
    import kwconf

    class LabelMeExportConfig(kwconf.Config):
        """Export a kwcoco prediction dataset to LabelMe JSON sidecars."""

        pred_kwcoco = kwconf.Value(None, required=True, position=1,
                                 help="source prediction kwcoco dataset")
        score_thresh = kwconf.Value(0.0, parser=float, help="minimum annotation score to export")
        only_missing = kwconf.Value(True, isflag=True,
                                  help="skip images that already have a sidecar")
        copy_dst = kwconf.Value(None, help="optional directory; source images copied here before export")

        @classmethod
        def main(cls, argv=1, **kwargs):
            config = cls.cli(argv=argv, data=kwargs, strict=True)
            written = export_to_labelme(
                config.pred_kwcoco,
                score_thresh=float(config.score_thresh),
                only_missing=bool(config.only_missing),
                copy_dst=config.copy_dst,
            )
            for p in written:
                print(f"wrote: {p}")
            print(f"total: {len(written)} sidecar(s)")
            return 0

    return LabelMeExportConfig


__cli__ = _build_cli()
