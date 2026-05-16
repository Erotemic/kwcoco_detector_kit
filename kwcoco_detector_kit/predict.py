"""Package-aware kwcoco prediction.

The public entry point is :func:`predict_kwcoco`, which accepts either a
package directory, a ``package.yaml`` manifest, or an archive package
(``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``).  The ``src`` argument accepts
either a kwcoco dataset path or a plain image directory (JPEG/PNG/TIFF files
are auto-collected into an in-memory kwcoco dataset).

Package manifest ``pipeline`` field
------------------------------------
``"detector_only"`` (default)
    Run the :class:`DetectorPredictor` and write bounding-box annotations.

``"detector_segmenter"``
    Chain the detector with a :class:`~kwcoco_detector_kit.trainers.sam2.SAM2Segmenter`
    to produce polygon / segmentation annotations.  Requires a ``segmenter``
    section in the package manifest with at minimum either:

    * ``checkpoint_fpath`` + ``config_relpath`` (local checkpoint), **or**
    * ``hf_model_id`` (HuggingFace download on first use).

    The ``postprocess`` section should also include ``crop_padding``,
    ``polygon_simplify``, ``min_component_area``, and
    ``keep_largest_component`` for this pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import scriptconfig as scfg

from kwcoco_detector_kit.export.package import materialize_workdir, open_package

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_rgb(image):
    """Normalise kwcoco-loaded image arrays into HWC uint8-ish RGB."""
    import numpy as np
    arr = image
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _load_labels(package_root: Path, manifest: dict) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    labels_rel = artifacts.get("labels")
    if labels_rel:
        labels_fpath = package_root / labels_rel
        if labels_fpath.exists():
            data = json.loads(labels_fpath.read_text())
            labels = data.get("labels")
            if labels:
                return [str(lbl) for lbl in labels]
    category_name = manifest.get("category_name") or "object"
    return [str(category_name)]


def _build_kwcoco_from_image_dir(image_dpath: Path):
    """Build an in-memory kwcoco dataset from image files in a directory."""
    import kwcoco
    dset = kwcoco.CocoDataset()
    for fpath in sorted(image_dpath.iterdir()):
        if fpath.is_file() and fpath.suffix.lower() in _IMAGE_EXTS:
            dset.add_image(file_name=str(fpath.resolve()))
    return dset


def _clone_for_predictions(true_dset, dst_path=None):
    """Clone image records from a kwcoco dataset, clearing all annotations.

    Used to create the prediction output skeleton so image metadata (width,
    height, name, etc.) is preserved without carrying over ground-truth
    annotations.
    """
    import kwcoco
    pred = kwcoco.CocoDataset()
    if dst_path is not None:
        pred.fpath = str(dst_path)
    for img in true_dset.images().objs:
        new = {k: v for k, v in img.items() if k != "id"}
        try:
            new["file_name"] = str(true_dset.get_image_fpath(img["id"]))
        except Exception:
            pass
        pred.add_image(**new, id=img["id"])
    return pred


def _coerce_src_kwcoco(src: str | Path):
    """Coerce ``src`` to a kwcoco dataset.

    Accepts:
    - A kwcoco path (``.json``, ``.zip``, ``.kwcoco.*``).
    - A directory that contains a ``.kwcoco.*`` file (auto-detected).
    - A plain image directory — all JPEG/PNG/TIFF files become images.
    """
    import kwcoco
    src = Path(src).expanduser()
    if src.is_dir():
        candidates = sorted(src.glob("*.kwcoco.*")) + sorted(src.glob("*.kwcoco"))
        if candidates:
            return kwcoco.CocoDataset.coerce(str(candidates[0]))
        return _build_kwcoco_from_image_dir(src)
    return kwcoco.CocoDataset.coerce(str(src))


def _build_segmenter(manifest: dict, package_root: Path):
    """Construct a SAM2Segmenter from the ``segmenter`` section of a manifest.

    Checkpoint paths that are relative are resolved against ``package_root``
    so packaged segmenters work after archive extraction.  Absolute paths and
    HuggingFace model IDs are used as-is.
    """
    from kwcoco_detector_kit.trainers.sam2 import SAM2Segmenter

    segmenter_cfg = dict(manifest.get("segmenter") or {})
    ckpt = segmenter_cfg.get("checkpoint_fpath")
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            resolved = (package_root / ckpt_path).resolve()
            if resolved.exists():
                segmenter_cfg["checkpoint_fpath"] = str(resolved)
    return SAM2Segmenter(segmenter_cfg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_kwcoco(
    *,
    package: str | Path,
    src: str | Path,
    dst: str | Path,
    device: str = "cpu",
    score_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    workdir: Optional[str | Path] = None,
) -> Path:
    """Run a packaged detector over a kwcoco dataset and write predictions.

    Args:
        package: Package directory, archive, or ``package.yaml`` manifest.
        src: Source kwcoco dataset or plain image directory.
        dst: Destination prediction kwcoco path.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
        score_thresh: Override for manifest ``postprocess.score_thresh``.
        nms_thresh: Override for manifest ``postprocess.nms_iou_thresh``.
        workdir: Optional persistent materialized-package workdir (avoids
            re-extraction on repeated calls).

    Returns:
        The written prediction kwcoco path.
    """
    import tempfile

    from kwcoco_detector_kit.data.postprocess import (
        add_prediction_annotations,
        detector_records_to_anns,
        detector_records_to_bbox_anns,
    )
    from kwcoco_detector_kit.trainers._registry import get_trainer

    dst = Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open_package(package) as (package_root, manifest):
        trainer = get_trainer(str(manifest["trainer"]))
        post_manifest = manifest.get("postprocess", {}) or {}
        if score_thresh is None:
            score_thresh = float(post_manifest.get("score_thresh", 0.30))
        if nms_thresh is None:
            nms_thresh = float(post_manifest.get("nms_iou_thresh", 0.50))

        labels = _load_labels(package_root, manifest)
        label_mapping = {i: name for i, name in enumerate(labels)}
        post_cfg = {
            "score_thresh": score_thresh,
            "nms_thresh": nms_thresh,
            "crop_padding": float(post_manifest.get("crop_padding", 10)),
            "polygon_simplify": float(post_manifest.get("polygon_simplify", 1.0)),
            "min_component_area": float(post_manifest.get("min_component_area", 50.0)),
            "keep_largest_component": bool(post_manifest.get("keep_largest_component", True)),
        }

        pipeline = str(manifest.get("pipeline", "detector_only"))
        segmenter = _build_segmenter(manifest, package_root) if pipeline == "detector_segmenter" else None

        tmp_ctx = None
        if workdir is None:
            tmp_ctx = tempfile.TemporaryDirectory()
            materialized = Path(tmp_ctx.name) / "workdir"
        else:
            materialized = Path(workdir).expanduser()
        try:
            materialize_workdir(package_root, manifest, materialized)
            predictor = trainer.build_predictor(materialized, device=str(device))

            true = _coerce_src_kwcoco(src)
            pred = _clone_for_predictions(true, dst)
            backend_name = str(manifest["trainer"])

            for gid in list(true.images()):
                coco_img = true.coco_image(gid)
                try:
                    arr = _coerce_rgb(coco_img.imdelay().finalize())
                except Exception as ex:
                    print(f"predict: failed to read gid {gid}: {ex}")
                    continue
                W, H = arr.shape[1], arr.shape[0]
                records = predictor.predict_image(arr, (W, H))
                if segmenter is not None:
                    anns = detector_records_to_anns(arr, records, segmenter, post_cfg, label_mapping)
                else:
                    anns = detector_records_to_bbox_anns(records, post_cfg, label_mapping)
                add_prediction_annotations(pred, gid, anns, backend_name)

            pred.dump()
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class PredictConfig(scfg.DataConfig):
    """Run packaged detector inference over a kwcoco dataset or image directory."""

    package = scfg.Value(None, required=True, help="package directory, archive, or package.yaml")
    src = scfg.Value(None, required=True, help="source kwcoco dataset or image directory")
    dst = scfg.Value(None, required=True, help="prediction kwcoco output path")
    device = scfg.Value("cpu", help="torch device, e.g. cpu, cuda, cuda:0")
    score_thresh = scfg.Value(None, type=float, help="override detection score threshold")
    nms_thresh = scfg.Value(None, type=float, help="override NMS IoU threshold")
    workdir = scfg.Value(None, help="optional persistent materialized predictor workdir")
    create_labelme = scfg.Value(False, isflag=True,
                                help="write LabelMe sidecars after prediction (polygon annotations only)")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        out = predict_kwcoco(
            package=config.package,
            src=config.src,
            dst=config.dst,
            device=str(config.device),
            score_thresh=config.score_thresh,
            nms_thresh=config.nms_thresh,
            workdir=config.workdir,
        )
        if config.create_labelme:
            from kwcoco_detector_kit.export.labelme import export_to_labelme
            written = export_to_labelme(out, only_missing=True)
            print(f"wrote {len(written)} LabelMe sidecar(s)")
        print(f"wrote predictions: {out}")
        return 0


__cli__ = PredictConfig
