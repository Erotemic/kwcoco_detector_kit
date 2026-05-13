"""
Package-aware kwcoco prediction.

The public entry point is :func:`predict_kwcoco`, which accepts either a
package directory, a ``package.yaml`` manifest, or an archive package
(``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``). The package manifest names the
trainer plugin and the artifacts needed to reconstruct the trainer workdir.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import scriptconfig as scfg

from kwcoco_detector_kit.export.package import materialize_workdir, open_package


def _coerce_rgb(image):
    """Normalize kwcoco-loaded image arrays into HWC uint8-ish RGB input."""
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
                return [str(label) for label in labels]
    category_name = manifest.get("category_name") or "object"
    return [str(category_name)]


def _add_prediction_categories(pred, labels: Iterable[str]) -> Dict[int, int]:
    label_to_cid = {}
    for label_idx, label in enumerate(labels):
        label_to_cid[label_idx] = pred.add_category(name=str(label))
    if not label_to_cid:
        label_to_cid[0] = pred.add_category(name="object")
    return label_to_cid


def predict_kwcoco(
    *,
    package: str | Path,
    src: str | Path,
    dst: str | Path,
    device: str = "cpu",
    score_thresh: Optional[float] = None,
    workdir: Optional[str | Path] = None,
) -> Path:
    """Run a packaged detector over a kwcoco dataset and write predictions.

    Args:
        package: Package directory, package archive, or ``package.yaml``.
        src: Source kwcoco dataset.
        dst: Destination prediction kwcoco.
        device: Device passed to the trainer predictor.
        score_thresh: Optional override for manifest ``postprocess.score_thresh``.
        workdir: Optional reusable materialized package workdir.

    Returns:
        The written prediction kwcoco path.
    """
    import kwcoco

    from kwcoco_detector_kit.trainers._registry import get_trainer

    dst = Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open_package(package) as (package_root, manifest):
        trainer = get_trainer(str(manifest["trainer"]))
        postprocess = manifest.get("postprocess", {}) or {}
        if score_thresh is None:
            score_thresh = float(postprocess.get("score_thresh", 0.30))
        labels = _load_labels(package_root, manifest)

        tmp_ctx = None
        if workdir is None:
            tmp_ctx = tempfile.TemporaryDirectory()
            materialized = Path(tmp_ctx.name) / "workdir"
        else:
            materialized = Path(workdir).expanduser()
        try:
            materialize_workdir(package_root, manifest, materialized)
            predictor = trainer.build_predictor(materialized, device=str(device))

            true = kwcoco.CocoDataset.coerce(str(src))
            pred = kwcoco.CocoDataset()
            pred.fpath = str(dst)
            label_to_cid = _add_prediction_categories(pred, labels)

            for img in true.images().objs:
                new = {k: v for k, v in img.items() if k != "id"}
                try:
                    new["file_name"] = str(true.get_image_fpath(img["id"]))
                except Exception:
                    pass
                pred.add_image(**new, id=img["id"])

            for gid in list(true.images()):
                coco_img = true.coco_image(gid)
                try:
                    arr = _coerce_rgb(coco_img.imdelay().finalize())
                except Exception as ex:
                    print(f"predict: failed to read gid {gid}: {ex}")
                    continue
                H, W = arr.shape[:2]
                detections = predictor.predict_image(arr, (W, H))
                for det in detections:
                    score = float(det.get("score", 0.0))
                    if score < float(score_thresh):
                        continue
                    label_idx = int(det.get("label", 0))
                    cat_id = label_to_cid.get(label_idx, next(iter(label_to_cid.values())))
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    pred.add_annotation(
                        image_id=gid,
                        category_id=cat_id,
                        bbox=[
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1),
                        ],
                        score=score,
                    )
            pred.dump()
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    return dst


class PredictConfig(scfg.DataConfig):
    """Run packaged detector inference over a kwcoco dataset."""

    package = scfg.Value(None, required=True, help="package directory, archive, or package.yaml")
    src = scfg.Value(None, required=True, help="source kwcoco dataset")
    dst = scfg.Value(None, required=True, help="prediction kwcoco output")
    device = scfg.Value("cpu", help="torch device, e.g. cpu, cuda, cuda:0")
    score_thresh = scfg.Value(None, type=float, help="override detection score threshold")
    workdir = scfg.Value(None, help="optional persistent materialized predictor workdir")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        out = predict_kwcoco(
            package=config.package,
            src=config.src,
            dst=config.dst,
            device=str(config.device),
            score_thresh=config.score_thresh,
            workdir=config.workdir,
        )
        print(f"wrote predictions: {out}")
        return 0


__cli__ = PredictConfig
